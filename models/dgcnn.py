import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Dropout
from torch.nn import BatchNorm1d as BN, GroupNorm as GN, LayerNorm as LN
from itertools import combinations, product
from random import shuffle
from convex_loss import convex_loss
# from torch_cluster import knn_graph
#from convex_loss import ACDLoss
from models.reconstruction import AtlasNet
from models.reconstruction import ChamferDistance


def knn_memory_efficient(x, k):
    batch_size = x.shape[0]
    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            inner = -2 * torch.matmul(x[b:b + 1].transpose(2, 1), x[b:b + 1])
            xx = torch.sum(x[b:b + 1] ** 2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            distances.append(pairwise_distance)
        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        idx = distances.topk(k=k, dim=-1)[1]
        del distances
        torch.cuda.empty_cache()
    return idx


def knn_fast(x, k):
    batch_size = x.shape[0]
    with torch.no_grad():
        inner = -2 * torch.bmm(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        distances = -xx - inner - xx.transpose(2, 1)
        idx = distances.topk(k=k, dim=-1)[1]
        # del distances
        # torch.cuda.empty_cache()
    return idx


def knn_fast_subset(x, k, subset_size=1000):
    all_inds = np.array(list(range(x.shape[-1])))
    inds = np.random.choice(all_inds, size=subset_size, replace=False)
    inds = torch.LongTensor(inds).to(x.device)
    batch_size = x.shape[0]
    with torch.no_grad():
        s = x[:, :, inds]
        inner = -2 * torch.bmm(x.transpose(2, 1), s)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        ss = torch.sum(s ** 2, dim=1, keepdim=True)
        distances = -xx - inner.transpose(2, 1) - ss.transpose(2, 1)
        idx = distances.transpose(1, 2).topk(k=k, dim=-1)[1]
        # del distances
        # torch.cuda.empty_cache()

    return inds[idx]


def knn_torch_cluster(tst, k):
    x = torch.cat([tst[i].t() for i in range(len(tst))]).to('cuda')
    batch = torch.tensor([[i] * tst.shape[2] for i in range(len(tst))]).reshape(-1).to('cuda')
    edges = knn_graph(x, k=k, batch=batch, loop=True)[0]
    idx = edges.reshape((-1, k))
    idx = idx.reshape((len(tst), -1, k))
    idx = idx - tst.shape[2] * batch.reshape((len(tst), -1)).unsqueeze(2)

    return idx


def get_graph_feature(x, k=20, idx=None, knn_func=knn_fast):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn_memory_efficient(x, k=k)  # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


def MLP(channels, bn=True, activation=ReLU()):
    if bn:
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i]), activation, BN(channels[i]))
            for i in range(1, len(channels))])
    else:
        return Seq(*[
            Seq(Lin(channels[i - 1], channels[i]), activation)
            for i in range(1, len(channels))])


class DGCNN(nn.Module):
    def __init__(self, num_classes, emb_dims=1024, k=20):
        super(DGCNN, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.lin1 = MLP([2 * 64 + 128 + 256, 1024])

        self.mlp = Seq(
            MLP([1024, 256]), Dropout(0.5), MLP([256, 128]), Dropout(0.5),
            Lin(128, num_classes))

    def forward(self, x, pos, batch, edge_index):
        # print('x', x)
        # print('pos', pos)

        batch_size = torch.max(batch).item() + 1
        # print(batch_size)
        to_cat = [x[batch == i].t().unsqueeze(0) for i in range(batch_size)]
        x = torch.cat(to_cat, dim=0)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        # print('x shape', x.shape)
        out = [x[i].t() for i in range(batch_size)]
        out = torch.cat(out, dim=0)

        # print(out.shape)
        out = self.lin1(out)
        out = self.mlp(out)

        # print(out.shape)
        return F.log_softmax(out, dim=1)


class DGCNNVanillaVer2(nn.Module):
    def __init__(self, num_classes, k=20, dropout=0.3):
        super(DGCNNVanillaVer2, self).__init__()
        self.drop = dropout
        self.k = k
        self.num_classes = num_classes
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.lin1 = MLP([2 * 64 + 128 + 256, 1024])

        self.mlp1 = nn.Conv1d(256, 1024, 1)
        self.bn_fin = nn.BatchNorm1d(1024)

        self.seg_conv1 = torch.nn.Conv1d(1024 + 256, 512, 1)
        self.seg_conv2 = torch.nn.Conv1d(512, 256, 1)
        self.seg_conv3 = torch.nn.Conv1d(256, 128, 1)
        self.seg_conv4 = torch.nn.Conv1d(128, num_classes, 1)

        self.seg_bn1 = nn.BatchNorm1d(512)
        self.seg_bn2 = nn.BatchNorm1d(256)
        self.seg_bn3 = nn.BatchNorm1d(128)

        self.fin_mlp = torch.nn.Linear(128, num_classes)

    def forward(self, x, pos, batch, edge_index):
        # print('x', x)
        # print('pos', pos)

        batch_size = torch.max(batch).item() + 1
        # print(batch_size)
        to_cat = [x[batch == i].t().unsqueeze(0) for i in range(batch_size)]
        x = torch.cat(to_cat, dim=0)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        x_features = torch.cat((x1, x2, x3), dim=1)

        x = F.relu(self.bn_fin(self.mlp1(x_features)))

        x_max = x.max(dim=2)[0]
        # x_avg = x.mean(dim=2)

        x_max = x_max.view(batch_size, 1024, 1).repeat(1, 1, x.shape[2])
        # x_avg = x_avg.view(batch_size, 1024, 1).repeat(1, 1, x.shape[2])
        x = torch.cat([x_max, x_features], 1)

        x = F.dropout(F.relu(self.seg_bn1(self.seg_conv1(x))), self.drop)
        x = F.dropout(F.relu(self.seg_bn2(self.seg_conv2(x))), self.drop)
        x = F.dropout(F.relu(self.seg_bn3(self.seg_conv3(x))), self.drop)
        x = self.seg_conv4(x)

        out = [x[i].t() for i in range(batch_size)]
        out = torch.cat(out, dim=0)

        # print(out.shape)

        # print(out.shape)
        return F.log_softmax(out, dim=1)

class DGCNNVanillaVer3(nn.Module):
    def __init__(self, num_classes, k=20, dropout=0.3):
        super(DGCNNVanillaVer3, self).__init__()
        self.drop = dropout
        self.k = k
        self.num_classes = num_classes
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.lin1 = MLP([2 * 64 + 128 + 256 + 512, 1024])

        self.mlp1 = nn.Conv1d(512, 1024, 1)
        self.bn_fin = nn.BatchNorm1d(1024)

        self.seg_conv1 = torch.nn.Conv1d(1024 + 512, 512, 1)
        self.seg_conv2 = torch.nn.Conv1d(512, 256, 1)
        self.seg_conv3 = torch.nn.Conv1d(256, 128, 1)
        self.seg_conv4 = torch.nn.Conv1d(128, num_classes, 1)

        self.seg_bn1 = nn.BatchNorm1d(512)
        self.seg_bn2 = nn.BatchNorm1d(256)
        self.seg_bn3 = nn.BatchNorm1d(128)

        self.fin_mlp = torch.nn.Linear(128, num_classes)

    def forward(self, x, pos, batch, edge_index):
        # print('x', x)
        # print('pos', pos)

        batch_size = torch.max(batch).item() + 1
        # print(batch_size)
        to_cat = [x[batch == i].t().unsqueeze(0) for i in range(batch_size)]
        x = torch.cat(to_cat, dim=0)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        x_features = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn_fin(self.mlp1(x_features)))

        x_max = x.max(dim=2)[0]
        # x_avg = x.mean(dim=2)

        x_max = x_max.view(batch_size, 1024, 1).repeat(1, 1, x.shape[2])
        # x_avg = x_avg.view(batch_size, 1024, 1).repeat(1, 1, x.shape[2])
        x = torch.cat([x_max, x_features], 1)

        x = F.dropout(F.relu(self.seg_bn1(self.seg_conv1(x))), self.drop)
        x = F.dropout(F.relu(self.seg_bn2(self.seg_conv2(x))), self.drop)
        x = F.dropout(F.relu(self.seg_bn3(self.seg_conv3(x))), self.drop)
        x = self.seg_conv4(x)

        out = [x[i].t() for i in range(batch_size)]
        out = torch.cat(out, dim=0)

        # print(out.shape)

        # print(out.shape)
        return F.log_softmax(out, dim=1)


class DGCNNVanillaShapeNet(nn.Module):
    def __init__(self, num_classes, k=20, dropout=0.3):
        super(DGCNNVanillaShapeNet, self).__init__()
        self.drop = dropout
        self.k = k
        self.num_classes = num_classes
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.lin1 = MLP([2 * 64 + 128 + 256, 1024])

        self.mlp1 = nn.Conv1d(512, 1024, 1)
        self.bn_fin = nn.BatchNorm1d(1024)

        self.seg_conv1 = torch.nn.Conv1d(1024 + 256, 512, 1)
        self.seg_conv2 = torch.nn.Conv1d(512, 256, 1)
        self.seg_conv3 = torch.nn.Conv1d(256, 128, 1)
        self.seg_conv4 = torch.nn.Conv1d(128, num_classes, 1)

        self.seg_bn1 = nn.BatchNorm1d(512)
        self.seg_bn2 = nn.BatchNorm1d(256)
        self.seg_bn3 = nn.BatchNorm1d(128)

        self.fin_mlp = torch.nn.Linear(128, num_classes)

    class DGCNNVanillaShapeNet(nn.Module):
        def __init__(self, num_classes, k=20, dropout=0.3):
            super(DGCNNVanillaShapeNet, self).__init__()
            self.drop = dropout
            self.k = k
            self.num_classes = num_classes
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)
            self.bn5 = nn.BatchNorm1d(1024)

            self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                       self.bn2,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                       self.bn3,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                       self.bn4,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                       self.bn5,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.lin1 = MLP([2 * 64 + 128 + 256, 1024])

            self.mlp1 = nn.Conv1d(512, 1024, 1)
            self.bn_fin = nn.BatchNorm1d(1024)

            self.seg_conv1 = torch.nn.Conv1d(1024 + 256, 512, 1)
            self.seg_conv2 = torch.nn.Conv1d(512, 256, 1)
            self.seg_conv3 = torch.nn.Conv1d(256, 128, 1)
            self.seg_conv4 = torch.nn.Conv1d(128, num_classes, 1)

            self.seg_bn1 = nn.BatchNorm1d(512)
            self.seg_bn2 = nn.BatchNorm1d(256)
            self.seg_bn3 = nn.BatchNorm1d(128)

            self.fin_mlp = torch.nn.Linear(128, num_classes)

    class DGCNNVanillaShapeNetVer2(nn.Module):
        def __init__(self, num_classes, k=20, dropout=0.3):
            super(DGCNNVanillaShapeNetVer2, self).__init__()
            self.drop = dropout
            self.k = k
            self.num_classes = num_classes
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
            self.bn4 = nn.BatchNorm2d(256)
            self.bn5 = nn.BatchNorm1d(1024)

            self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                       self.bn2,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                       self.bn3,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                       self.bn4,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                       self.bn5,
                                       nn.LeakyReLU(negative_slope=0.2))

            self.lin1 = MLP([2 * 64 + 128 + 256, 1024])

            self.mlp1 = nn.Conv1d(512, 1024, 1)
            self.bn_fin = nn.BatchNorm1d(1024)

            self.seg_conv1 = torch.nn.Conv1d(1024 + 256, 512, 1)
            self.seg_conv2 = torch.nn.Conv1d(512, 256, 1)
            self.seg_conv3 = torch.nn.Conv1d(256, 128, 1)
            self.seg_conv4 = torch.nn.Conv1d(128, num_classes, 1)

            self.seg_bn1 = nn.BatchNorm1d(512)
            self.seg_bn2 = nn.BatchNorm1d(256)
            self.seg_bn3 = nn.BatchNorm1d(128)

            self.fin_mlp = torch.nn.Linear(128, num_classes)

    def forward(self, x, pos, batch, edge_index):
        # print('x', x)
        # print('pos', pos)

        batch_size = torch.max(batch).item() + 1
        # print(batch_size)
        to_cat = [x[batch == i].t().unsqueeze(0) for i in range(batch_size)]
        x = torch.cat(to_cat, dim=0)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        x_features = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn_fin(self.mlp1(x_features)))

        x_max = x.max(dim=2)[0]
        # x_avg = x.mean(dim=2)

        x_max = x_max.view(batch_size, 1024, 1).repeat(1, 1, x.shape[2])
        # x_avg = x_avg.view(batch_size, 1024, 1).repeat(1, 1, x.shape[2])
        x = torch.cat([x_max, x_features], 1)

        x = F.dropout(F.relu(self.seg_bn1(self.seg_conv1(x))), self.drop)
        x = F.dropout(F.relu(self.seg_bn2(self.seg_conv2(x))), self.drop)
        x = F.dropout(F.relu(self.seg_bn3(self.seg_conv3(x))), self.drop)
        x = self.seg_conv4(x)

        out = [x[i].t() for i in range(batch_size)]
        out = torch.cat(out, dim=0)

        # print(out.shape)

        # print(out.shape)
        return F.log_softmax(out, dim=1)


class DGCNNVanillaGroupNorm(nn.Module):
    def __init__(self, num_classes, k=20, dropout=0.0, quantile=0.05, msc_iterations=5, max_num_clusters=25, convex_loss=False, beta=1.0, alpha=0.01, entropy_loss=False, if_acd_loss=False, mode=0, intersect=False, reconstruction=False):
        super(DGCNNVanillaGroupNorm, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.drop = dropout
        self.convex_loss = convex_loss
        self.entropy_loss = entropy_loss
        self.acd_loss = if_acd_loss
        self.mode = mode
        self.intersect = intersect
        self.reconstruction = reconstruction
        if if_acd_loss:
            self.loss = ACDLoss()
        else:
            self.loss = None

        self.quantile = quantile
        self.msc_iterations = msc_iterations
        self.max_num_clusters = max_num_clusters
        self.k = k
        self.num_classes = num_classes
        self.bn1 = nn.GroupNorm(8, 64)
        self.bn2 = nn.GroupNorm(8, 64)
        self.bn3 = nn.GroupNorm(8, 128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        # self.lin1 = MLP([2 * 64 + 128 + 256, 1024])

        self.mlp1 = nn.Conv1d(256, 1024, 1)
        self.bn_fin = nn.GroupNorm(32, 1024)

        self.seg_conv1 = torch.nn.Conv1d(1024 + 256, 512, 1)
        self.seg_conv2 = torch.nn.Conv1d(512, 256, 1)
        self.seg_conv3 = torch.nn.Conv1d(256, 128, 1)
        self.seg_conv4 = torch.nn.Conv1d(128, num_classes, 1)

        self.seg_bn1 = nn.GroupNorm(16, 512)
        self.seg_bn2 = nn.GroupNorm(16, 256)
        self.seg_bn3 = nn.GroupNorm(8, 128)

        if self.mode == 0:
            self.fc_embed = torch.nn.Conv1d(128, 64, 1, bias=False)
        elif self.mode == 1:
            self.fc_embed = torch.nn.Conv1d(256, 64, 1, bias=False)
        elif self.mode == 2:
            self.fc_embed0 = torch.nn.Conv1d(1024 + 256, 512, 1)
            self.fc_embed1 = torch.nn.Conv1d(512, 256, 1)
            self.fc_embed2 = torch.nn.Conv1d(256, 128, 1)
            self.fc_embed3 = torch.nn.Conv1d(128, 128, 1, bias=False)
            self.fc_embed0_bn = nn.GroupNorm(16, 512)
            self.fc_embed1_bn = nn.GroupNorm(16, 256)
            self.fc_embed2_bn = nn.GroupNorm(8, 128)
            self.fc_embed = nn.Sequential(self.fc_embed0, self.fc_embed0_bn, nn.ReLU(),
                                          self.fc_embed1, self.fc_embed1_bn, nn.ReLU(),
                                          self.fc_embed2, self.fc_embed2_bn, nn.ReLU(),
                                          self.fc_embed3)

        if self.reconstruction:
            self.reconst_embed = nn.Conv1d(1024, 128, 1)
            self.atlasnet = AtlasNet(128, 25, 128)
            self.chamferdistance = ChamferDistance()

    def forward(self, x, chamfer_x=0):
        if isinstance(x, list): # if it is a list, that means we need to compute acd loss also.
            x, target = x
        input_points = x
        batch_size = x.shape[0]
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        x_features = torch.cat((x1, x2, x3), dim=1)

        x_before_max_pool = F.relu(self.bn_fin(self.mlp1(x_features)))

        x_max = x_before_max_pool.max(dim=2)[0]
        # x_avg = x.mean(dim=2)

        x_max_expand = x_max.view(batch_size, 1024, 1).repeat(1, 1, x.shape[2])
        # x_avg = x_avg.view(batch_size, 1024, 1).repeat(1, 1, x.shape[2])
        x_combined = torch.cat([x_max_expand, x_features], 1)

        x = F.relu(self.seg_bn1(self.seg_conv1(x_combined)))
        x_second_last = F.relu(self.seg_bn2(self.seg_conv2(x)))
        x_last = F.relu(self.seg_bn3(self.seg_conv3(x_second_last)))

        out_prob = F.log_softmax(self.seg_conv4(x_last), dim=1)
        if self.acd_loss:
            if self.mode == 0:
                embedding = self.fc_embed(x_last)
            elif self.mode == 1:
                embedding = self.fc_embed(x_second_last)

            num_points = embedding.shape[-1]
            random_samples = np.random.choice(num_points, num_points // 2)
            embedding = embedding[:, :, random_samples]
            target = target[:, random_samples]
            loss = self.loss(embedding, target)
            return out_prob, loss

        if self.convex_loss:
            if self.mode == 0:
                embedding = self.fc_embed(x_last)
            elif self.mode == 1:
                embedding = self.fc_embed(x_second_last)
            elif self.mode == 2:
                embedding = self.fc_embed(x_combined)

            if self.beta > 0.005:
                self.beta *= 0.99
            else:
                self.entropy_loss = False
            #print(input_points.shape, embedding.shape)
            #input_points = torch.squeeze(input_points, 3)
            cd = convex_loss(input_points, chamfer_x, embedding,
                             include_intersect_loss=self.intersect,
                             if_cuboid=False,
                             alpha=self.alpha,
                             beta=self.beta,
                             quantile=self.quantile,
                             iterations=self.msc_iterations,
                             max_num_clusters=self.max_num_clusters,
                             include_entropy_loss=self.entropy_loss,
                             include_pruning=False)

            # embedding.register_hook(lambda x: print("Grad norm: ", x.norm().item()))
            return out_prob, cd[0].view(1, 1)

        if self.reconstruction:
            z = torch.mean(x_before_max_pool, 2, keepdim=True) # max pool to get global embedding
            feat = self.reconst_embed(z) # map 1024 dimensional vector to 128 d vec
            feat = torch.squeeze(feat, -1)
            output_points = self.atlasnet(feat)
            input_points = torch.squeeze(input_points, -1).permute(0, 2, 1)
            cd = self.chamferdistance(output_points, input_points).view(1, 1)
            return out_prob, cd
        return out_prob

    
class DGCNNMemEff(nn.Module):
    def __init__(self, num_classes, emb_dims=1024, k=20, dropout=0.5):
        super(DGCNNMemEff, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.lin1 = MLP([2 * 64 + 128 + 256, 1024])

        self.mlp = Seq(
            MLP([1024, 256]), Dropout(0.5), MLP([256, 128]), Dropout(0.5),
            Lin(128, num_classes))

    def forward(self, x, pos, batch, edge_index):
        # print('x', x)
        # print('pos', pos)

        batch_size = torch.max(batch).item() + 1
        # print(batch_size)
        to_cat = [x[batch == i].t().unsqueeze(0) for i in range(batch_size)]
        x = torch.cat(to_cat, dim=0)

        x = get_graph_feature(x, k=self.k, knn_func=knn_memory_efficient)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k, knn_func=knn_memory_efficient)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k, knn_func=knn_memory_efficient)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k, knn_func=knn_memory_efficient)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        # print('x shape', x.shape)
        out = [x[i].t() for i in range(batch_size)]
        out = torch.cat(out, dim=0)

        # print(out.shape)
        out = self.lin1(out)
        out = self.mlp(out)

        # print(out.shape)
        return F.log_softmax(out, dim=1)


class EdgeConvHeadRefactor(nn.Module):
    def __init__(self, num_classes, emb_dims=1024, k=20, channel_dims=[64, 64, 128, 256],
                 feat_dims=3,
                 knn_func=knn_fast, group_norm=None):
        super(EdgeConvHeadRefactor, self).__init__()
        self.k = k
        self.knn_func = knn_func
        self.num_classes = num_classes
        #self.bn5 = nn.BatchNorm1d(emb_dims)
        self.channel_dims = channel_dims
        channel_dims = [feat_dims] + channel_dims
        self.conv_blocks = nn.ModuleList([])

        for i, dim in enumerate(channel_dims[1:]):
            #print(dim, 2*channel_dims[i-1])
            if group_norm is not None:
                print('Using Group Norm for the EdgeconvHead.')
                normalization_layer = nn.GroupNorm(group_norm[i], dim)

            else:
                normalization_layer = nn.BatchNorm2d(dim)

            cur_conv = nn.Sequential(nn.Conv2d(2 * channel_dims[i], dim, kernel_size=1, bias=False),
                                     normalization_layer,
                                     nn.LeakyReLU(negative_slope=0.2))
            self.conv_blocks.append(cur_conv)

        #self.conv5 = nn.Sequential(nn.Conv1d(sum(channel_dims[1:])*2, emb_dims, kernel_size=1, bias=False),
        #                           self.bn5,
        #                          nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, pos=None, batch=None):

        if batch is not None:

            batch_size = torch.max(batch).item() + 1
            to_cat = [x[batch == i].t().unsqueeze(0) for i in range(batch_size)]
            x = torch.cat(to_cat, dim=0)

        reprs = []

        for conv_block in self.conv_blocks:
            x = get_graph_feature(x, k=self.k, knn_func=self.knn_func)
            #print(x.shape)
            x = conv_block(x)
            #print(x.shape, '\n')
            out = x.max(dim=-1, keepdim=False)[0]
            x = out
            reprs += [out]

        #print(len(reprs), reprs[0].shape)
        return reprs

    def get_reps(self, x, pos=None, batch=None):

        x = self.forward(x, pos=pos, batch=batch)
        x = torch.cat(x, dim=1)

        return x


class EdgeConvHeadRefactorFusion(nn.Module):
    def __init__(self, num_classes, emb_dims=1024, k=20, channel_dims=[64, 64, 128, 256],
                 knn_func=knn_fast, group_norm=None):
        super(EdgeConvHeadRefactorFusion, self).__init__()
        self.k = k
        self.knn_func = knn_func
        self.num_classes = num_classes
        #self.bn5 = nn.BatchNorm1d(emb_dims)
        self.channel_dims = channel_dims
        channel_dims = [3] + channel_dims
        self.conv_blocks = nn.ModuleList([])
        self.fuse_blocks = nn.ModuleList([])

        for i, dim in enumerate(channel_dims[1:]):
            # print(dim, 2*channel_dims[i-1])
            if group_norm is not None:
                print('Using Group Norm for the EdgeconvHead.')
                normalization_layer = nn.GroupNorm(group_norm[i], dim)

            else:
                normalization_layer = nn.BatchNorm2d(dim)

            cur_conv = Seq(nn.Conv2d(2 * channel_dims[i], dim, kernel_size=1, bias=False),
                                     normalization_layer,
                                     nn.LeakyReLU(negative_slope=0.2))
            cur_fuse = Seq(nn.Conv1d(2 * dim, dim, kernel_size=1, bias=False),
                           nn.ELU())

            self.conv_blocks.append(cur_conv)
            self.fuse_blocks.append(cur_fuse)

        print(self.fuse_blocks)
        #self.conv5 = nn.Sequential(nn.Conv1d(channel_dims[-1]*2, emb_dims, kernel_size=1, bias=False),
        #                           self.bn5,
        #                          nn.LeakyReLU(negative_slope=0.2))

    def forward_step(self, x, conv_block, fuse_rep=None, fuse_block=None):

        if fuse_rep is not None:
            x = fuse_block(torch.cat((x, fuse_rep), axis=1))

        x = get_graph_feature(x, k=self.k, knn_func=self.knn_func)
        # print(x.shape)
        x = conv_block(x)
        # print(x.shape, '\n')
        out = x.max(dim=-1, keepdim=False)[0]

        return out

    def forward(self, x, pos=None, batch=None, reps_fuse=None):

        if batch is not None:

            batch_size = torch.max(batch).item() + 1
            to_cat = [x[batch == i].t().unsqueeze(0) for i in range(batch_size)]
            x = torch.cat(to_cat, dim=0)

        reprs = []
        if reps_fuse is not None:
            assert len(reps_fuse) == len(self.conv_blocks)

        for i, conv_block in enumerate(self.conv_blocks):
            out = self.forward_step(x, self.conv_blocks[i], fuse_rep=reps_fuse[i], fuse_block=self.fuse_blocks[i])
            x = out
            reprs += [out]

        return reprs
