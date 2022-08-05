import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        batchsize = x.size()[0]
        # print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class AtlasNet(nn.Module):
    def __init__(self, bottleneck_size=128, num_charts=25, num_points=128):
        super(AtlasNet, self).__init__()
       
        self.nb_primitives = num_charts
        self.num_points = num_points

        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=2 + bottleneck_size) for i in range(0, self.nb_primitives)])
        # Computer regular uv grid
        grid_size = int(np.sqrt(self.num_points))
        self.grid_size = grid_size
        grid = np.indices((grid_size, grid_size)).T.reshape(-1, 2).T.astype('float32')
        grid = grid / (grid_size - 1)
        self.reg_grid = torch.from_numpy(grid).unsqueeze(0).cuda()

    def forward(self, z):
        # Latent representation z
        outs = []
        reg_grid = self.reg_grid.to(z.device)
        # Decoder - Use feat to decode
        for i in range(0, self.nb_primitives):
            # rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points//self.nb_primitives))
            # rand_grid.data.uniform_(0, 1)
            rand_grid = reg_grid
            y = z.unsqueeze(2).expand(z.size(0), z.size(1), rand_grid.size(2)).contiguous()
            # y = torch.cat( (rand_grid, y), 1).contiguous()
            y = torch.cat((rand_grid.expand(z.size(0), -1, -1), y), 1).contiguous()
            # outs.append(self.decoder[i](y))

            out = self.decoder[i](y)
            out = out.view(out.size()[0], out.size()[1], self.grid_size, self.grid_size)
            outs.append(out)

        allpts = []
        for i in range(self.nb_primitives):
            allpts.append(outs[i].view(outs[i].size()[0], outs[i].size()[1], -1))
        allpts = torch.cat(allpts, 2).contiguous().transpose(2, 1).contiguous()
        return allpts


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False, l2_norm=False):
        super(get_model, self).__init__()
        self.nb_primitives = 25
        self.bottleneck_size = 128
        self.num_points = 128
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0

        # Encoder
        self.normal_channel = normal_channel
        self.l2_norm = l2_norm
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3 + additional_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4, 0.8], [64, 128], 128 + 128 + 64,
                                             [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3,
                                          mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150 + additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

        # Decoder
        self.atlasnet = AtlasNet(self.bottleneck_size, self.nb_primitives, self.num_points)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B, C, N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)
        l0_points = self.fp1(l0_xyz, l1_xyz,
                             torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1),
                             l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        if self.l2_norm:
            feat = F.normalize(feat, p=2, dim=1)
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        # Latent representation z
        z = feat.mean(dim=2)
        allpts = self.atlasnet(z)
        return x, (l1_points, l2_points, l3_points), feat, allpts


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


@torch.jit.script
def batched_cdist_l2(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1),
                        beta=1, alpha=-2).add_(x1_norm).transpose(1, 2)
    return res


class ChamferDistance(nn.Module):

    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, x, y):
        # d = batch_pairwise_dist(x,y)
        d = batched_cdist_l2(x, y)
        # d = nn_dist_lp(x,y,1)
        return torch.mean(torch.min(d, dim=2)[0]) + torch.mean(torch.min(d, dim=1)[0])


class get_rec_selfsup_loss(nn.Module):
    def __init__(self, margin=0.5, lcont=0.0, lrec=1.0):
        super(get_rec_selfsup_loss, self).__init__()
        self.margin = margin
        self.ch = ChamferDistance()
        self.lcont = lcont
        self.lrec = lrec

    def forward(self, feat, target, pts, gtpts):
        # Contrastive loss
        feat = F.normalize(feat, p=2, dim=1)
        pair_sim = torch.bmm(feat.transpose(1, 2), feat)

        one_hot_target = F.one_hot(target).float()
        pair_target = torch.bmm(one_hot_target, one_hot_target.transpose(1, 2))

        cosine_loss = pair_target * (1. - pair_sim) + (1. - pair_target) * F.relu(pair_sim - self.margin)
        diag_mask = 1 - torch.eye(cosine_loss.shape[-1])  # discard diag elems (always 1)

        with torch.no_grad():
            # balance positive and negative pairs
            pos_fraction = (pair_target.data == 1).float().mean()
            sample_neg = torch.cuda.FloatTensor(*pair_target.shape).uniform_() > 1 - pos_fraction
            sample_mask = (pair_target.data == 1) | sample_neg  # all positives, sampled negatives

        cosine_loss = diag_mask.unsqueeze(0).cuda() * sample_mask.float() * cosine_loss
        total_loss = 0.5 * cosine_loss.mean()  # scale down

        # Reconstruction loss
        d1, d2 = self.ch(pts, gtpts.transpose(1, 2))
        rec_loss = d1.mean() + d2.mean()

        return self.lcont * total_loss + self.lrec * rec_loss


class get_selfsup_loss(nn.Module):
    def __init__(self, margin=0.5):
        super(get_selfsup_loss, self).__init__()
        self.margin = margin

    def forward(self, feat, target):
        feat = F.normalize(feat, p=2, dim=1)
        pair_sim = torch.bmm(feat.transpose(1, 2), feat)

        one_hot_target = F.one_hot(target).float()
        pair_target = torch.bmm(one_hot_target, one_hot_target.transpose(1, 2))

        cosine_loss = pair_target * (1. - pair_sim) + (1. - pair_target) * F.relu(pair_sim - self.margin)
        diag_mask = 1 - torch.eye(cosine_loss.shape[-1])  # discard diag elems (always 1)

        with torch.no_grad():
            # balance positive and negative pairs
            pos_fraction = (pair_target.data == 1).float().mean()
            sample_neg = torch.cuda.FloatTensor(*pair_target.shape).uniform_() > 1 - pos_fraction
            sample_mask = (pair_target.data == 1) | sample_neg  # all positives, sampled negatives

        cosine_loss = diag_mask.unsqueeze(0).cuda() * sample_mask.float() * cosine_loss
        total_loss = 0.5 * cosine_loss.mean()  # scale down

        return total_loss
