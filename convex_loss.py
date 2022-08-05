import numpy as np
from src.VisUtils import visualize_point_cloud
from src.utils import visualize_point_cloud
import torch
from src.fitting_utils import customsvd
from open3d import *
import trimesh
from src.mean_shift import MeanShift
import torch
from torch import nn
from src.guard import guard_exp
from scipy.spatial.transform import Rotation as R
from src.utils import visualize_point_cloud_from_labels
from src.sample_ellipsoid import SampleEllipsoid
from src.ellipsoid_fitting import *
from src.sample_ellipsoid import Loss
# from torch_scatter import scatter_mean
from src.utils import analytic_chamfer_distance
from src.VisUtils import grid_points_lists_visulation


from src.ellipsoid_utils import sample_from_pred_params, clustering, sample_from_pred_params_cuboid
import sys, os

loss_cd = Loss()

def convex_loss(points, chamfer_points, X, batch_id=0, epoch=-1, seed=0, N=500, quantile=0.01, iterations=5, visualize=False, max_num_clusters=25, class_list=[], include_intersect_loss=False, alpha=1, beta=1, if_cuboid=False, include_pruning=False, include_entropy_loss=False, evaluation=False):
    """
    Computes convex approximation loss.
    :param X: per point embedding, size: B x N x 128
    :param points: input points to the model,
    :param N: number of points to sample on each ellipsoid
    :param iterations: number of iterations of mean shift clustering to run.
    :param visualize: whether to save the visualization results.
    """ 
    
    X = X.permute(0, 2, 1)
    points = points.permute(0, 2, 1)

    entropy_loss = torch.zeros(1, requires_grad=True).cuda()
    X = torch.nn.functional.normalize(X, dim=2, p=2)
    if visualize:
        directory1 = 'lasts56q{}/inputs/'.format(quantile)
        directory2 = 'lasts56q{}/embeddings/'.format(quantile)
        # if not os.path.exists(directory1):
        #     os.makedirs(directory1)
        # if not os.path.exists(directory2):
        #     os.makedirs(directory2)
        # for batch_index in range(X.shape[0]):
        #     path1 = os.path.join(directory1, 'batch_{}_{}_{}.xyz'.format(batch_id, batch_index, class_list[batch_index]))
        #     path2 = os.path.join(directory2, 'batch_{}_{}_{}.txt'.format(batch_id, batch_index, class_list[batch_index]))
        #     np.savetxt(path2, X[batch_index, :, :].data.cpu().numpy())
        #     np.savetxt(path1, points[batch_index, :, :].data.cpu().numpy())

    #device_id = torch.cuda.current_device()
    #svd_params = {'epoch_id': epoch, 'batch_id': batch_id, 'device_id': device_id}
    X = torch.nn.functional.normalize(X, dim=2, p=2)

    if include_entropy_loss:
        #print('entr loss')
        sub_sample_indices = np.random.choice(X.shape[1], X.shape[1] // 4, replace=False)
        entropy_loss = entropy(X[:, sub_sample_indices])
        #print('Ent loss: ', entropy_loss.item())
    # Sub-sample points
    torch.cuda.empty_cache()

    # TODO specify the number of samples, to X.shape[1] // 2
    weights_batch, labels = clustering(X, quantile=quantile, iterations=iterations, visualize=visualize, max_num_clusters=max_num_clusters, num_samples=X.shape[1])
   
    ellipse_params_batch = weighted_ellipsoid_fitting_batch(points, weights_batch)

    if not if_cuboid:
        resampled_points_batch = sample_from_pred_params(ellipse_params_batch, N, batch_id=batch_id, seed=seed, visualize=visualize, class_list=class_list, quantile=quantile)
    else:
        print("cuboid")
        resampled_points_batch = sample_from_pred_params_cuboid(ellipse_params_batch, N, batch_id=batch_id, seed=seed, visualize=visualize, class_list=class_list)

    if include_pruning:
        #print('pruning')
        pruned_points = prune_points(resampled_points_batch, ellipse_params_batch)
    else:
        pruned_points = resampled_points_batch

    if evaluation is False:
        chamfer_points = chamfer_points.permute(0, 2, 1)
        # l = loss_cd.loss(chamfer_points, pruned_points)
        #print('anal')

        l = analytic_chamfer_distance(ellipse_params_batch, resampled_points_batch, chamfer_points, cuboid=if_cuboid)
       
    else:
        #print('no chamfer points')
        #l = loss_cd.loss(points, pruned_points)
        l = torch.zeros(1, requires_grad=True).cuda()

    intersection_loss = torch.zeros(1, requires_grad=True).cuda()
    if include_intersect_loss:
        #print('int loss')
        intersection_loss = compute_intersection_loss_volume_3(ellipse_params_batch, chamfer_points - torch.rand(chamfer_points.shape).cuda() * 0.2, cuboid=if_cuboid)
    torch.cuda.empty_cache()
    total_loss = l + (alpha * intersection_loss) + (beta * entropy_loss)
    #print('chamfer loss: {}, int loss: {}'.format(l.item(), intersection_loss.item()))
    return total_loss.view(1, 1), l.view(1, 1), ellipse_params_batch, labels
    #return chamfer_loss + (alpha * intersection_loss) + (beta * entropy_loss), l

def compute_intersection_loss(ellipsoid_params_batch, sampled_points_batch):
    """
    Computes the amount of intersection a collection of ellipsoids have
    among each other. To do this, we use sampled points on all ellipsoids,
    and compute sdf values of each ellipsoid on each point. In this way, we end up
    with K x N matrix, where K is the number of ellipsoids, and N is the number
    of points. min across first dimension gives you the sdf calculated at a point
    with respect to the output shape which is a collection of ellipsoid.
    :param ellipsoid_params_batch: list of list; containing ellipsoid parameters for
    each batch element.
    :param sampled_points_batch: sampled points on ellipsoids surface.
    :param ellipsoid_params_batch: parameters of ellipsoids.
    """
    #batch_size = len(ellipsoid_params_batch)
    batch_size = len(sampled_points_batch)

    if batch_size > 0:
        losses = []
        for b in range(batch_size):
            sdfs = []
            points = sampled_points_batch[b]
            for ellipse_param in ellipsoid_params_batch[b]:
                r, V, center = ellipse_param

                shifted_points = (V.T @ (points - center).T).T

                k0 = torch.norm(shifted_points / (r + 1e-6), p=2, dim=1)
                k1 = torch.norm(shifted_points / (r ** 2 + 1e-6), p=2, dim=1)

                sdf = k0 * (k0 - 1.0) / k1
                sdfs.append(sdf)

            sdfs = torch.stack(sdfs, 1)
        # to compute sdf w.r.t the entire shape, compute min over all ellipsoids.
            sdfs = torch.min(sdfs, 1)[0]

        # sdfs are thresholded because for some reason the above sdf formula
        # is always giving negative sdfs. Anyway this is a truncated sdf.
        # pcd1 = visualize_point_cloud(points[sdfs > -1e-3].data.cpu().numpy(), viz=False)
        # pcd2 = visualize_point_cloud(points[sdfs < -1e-3].data.cpu().numpy(), viz=False)
        #
        # pcd1.paint_uniform_color([0, 1, 0])
        # pcd2.paint_uniform_color([0, 0, 1])
        # visualization.draw_geometries([pcd1, pcd2])

        # sort of like a margin.
            sdfs = torch.clamp_max(sdfs, -1e-3)

            losses.append(torch.mean(sdfs))
        losses = torch.stack(losses) ** 2
        losses = losses.mean()
    # value.
    else:
        losses = torch.zeros(1, requires_grad=True).cuda()
    return losses


def compute_intersection_loss_cuboid(ellipsoid_params_batch, sampled_points_batch):
    """
    Computes intersection of cuboids, follows same strategy as ellipsoid.
    :param sampled_points_batch: sampled points on ellipsoids surface.
    :param ellipsoid_params_batch: parameters of ellipsoids.
    """
    batch_size = len(ellipsoid_params_batch)

    losses = []
    for b in range(batch_size):
        sdfs = []
        points = sampled_points_batch[b]
        for ellipse_param in ellipsoid_params_batch[b]:
            r, V, center = ellipse_param

            shifted_points = (V.T @ (points - center).T).T

            q = torch.abs(shifted_points) - r

            # not exactly correct but will work on the kind of points we have.
            sdf = torch.max(q, 1)[0]
            sdfs.append(sdf)

        sdfs = torch.stack(sdfs, 1)
        # to compute sdf w.r.t the entire shape, compute min over all ellipsoids.
        sdfs = torch.min(sdfs, 1)[0]

        # sdfs are thresholded because for some reason the above sdf formula
        # is always giving negative sdfs. Anyway this is a truncated sdf.
        # pcd1 = visualize_point_cloud(points[sdfs > -1e-3].data.cpu().numpy(), viz=False)
        # pcd2 = visualize_point_cloud(points[sdfs < -1e-3].data.cpu().numpy(), viz=False)
        #
        # pcd1.paint_uniform_color([0, 1, 0])
        # pcd2.paint_uniform_color([0, 0, 1])
        # visualization.draw_geometries([pcd1, pcd2])

        # sort of like a margin.
        sdfs = torch.clamp_max(sdfs, -1e-3)

        losses.append(torch.mean(sdfs))
    losses = torch.stack(losses) ** 2
    losses = losses.mean()
    # value.
    return losses


def entropy(X):
    """
    While training only using convexity loss, the initial embedding for all points
    is same, which leads to extreme local minima which is hard to get out of. Here,
    we define a measure of similarity between point embeddings and minimize it.
    @param X: embedding
    @return: loss
    """
    batch_size = X.shape[0]
    num_points = X.shape[1]
    l = []
    margin = 1.8
    # X = torch.nn.functional.normalize(X, dim=2, p=2)
    for b in range(batch_size):
        D = (1 + X[b] @ X[b].T) ** 2 # N x N
        l.append((torch.sum(D)) / num_points ** 2)
    return torch.relu(torch.stack(l).mean() - margin)

def compute_intersection_loss_volume(ellipsoid_params_batch, sampled_points_batch):
    """
    To minimize the intersection between ellipsoids, sample points inside ellipsoid,
    and then compute sdf at these points inside an ellipsoid wr.t other ellipsoids.
    :param sampled_points_batch: sampled points on ellipsoids surface.
    :param ellipsoid_params_batch: parameters of ellipsoids.
    """
    #batch_size = len(ellipsoid_params_batch)
    batch_size = len(sampled_points_batch)
    
    if batch_size > 0:
        losses = []
        for b in range(batch_size):
            samples = []
            sdfs = []
            if len(ellipsoid_params_batch[b]) <= 1:
                continue
        # sample points inside the ellipsoids along the principal axis.
            for i, ellipse_param in enumerate(ellipsoid_params_batch[b]):
                r, V, center = ellipse_param
                samples.append(sample_axis(r, V, center))

        # visualize_point_cloud(torch.cat(samples, 0).data.cpu().numpy(), viz=True)
            for i in range(len(ellipsoid_params_batch[b])):
                sdf = []
                for j in range(len(ellipsoid_params_batch[b])):
                    if i == j: continue
                    r, V, center = ellipsoid_params_batch[b][i]
                    sdf.append(compute_sdf_ellipsoid(samples[i], center, r, V))
                if len(sdf) == 0:
                    continue

            # minimum value of sdf of a point across all ellipsoid except one.
                sdf = torch.stack(sdf, 0)
                sdf = torch.min(sdf, 0)[0]
            # only use samples that have negative sdfs, that is, for points inside
            # another ellipsoid.
                sdf = torch.clamp_max(sdf, -1e-3)
                sdf = torch.mean(sdf)
                sdfs.append(sdf)

            sdfs = torch.stack(sdfs)
            loss = torch.mean(sdfs ** 2)
            losses.append(loss)

        if losses == []:
            losses = torch.zeros(1, requires_grad=True).cuda()
        else:
            losses = torch.stack(losses)
        
        losses = losses.mean()

    else:
        losses = torch.zeros(1, requires_grad=True).cuda()

    return losses


def sample_axis(r, V, center, num_samples=40):
    """
    Given parameters of ellipsoid, return point samples along principal axis.
    Sample more along longer axis.
    @param r: lengths of principal axis
    @param V: orthogonal principal axis
    @param center: center of ellipsoid
    @return: samples
    """
    scaled_axis = (V * r.view(1, 3)).T

    scaled_axis_1 = scaled_axis[0:1, :]
    scaled_axis_2 = scaled_axis[1:2, :]
    scaled_axis_3 = scaled_axis[2:3, :]
    # sample more along the longer axis
    with torch.no_grad():
        num_samples = r * num_samples / torch.sum(r)
        num_samples = num_samples.int() + 1

        ratios = []
        for i in range(3):
            ratios.append(torch.linspace(-0.9, 0.897, num_samples[i]).view(-1, 1).cuda())

    samples = torch.cat([scaled_axis_1 * ratios[0], scaled_axis_2 * ratios[1], scaled_axis_3 * ratios[2]], 0)

    samples = samples + center.view(1, 3)
    return samples

def compute_sdf_ellipsoid(points, center, r, V):
    """
    SDF of ellipsoid. This is only an approximation of sdf.
    @param points: points at which sdf needs to be calculated
    @param center: center of ellipsoid
    @param r: principal axis lengths
    @param V: principal axis
    @return: sdf at points
    """
    shifted_points = (V.T @ (points - center).T).T

    k0 = torch.norm(shifted_points / (r + 1e-6), p=2, dim=1)
    k1 = torch.norm(shifted_points / (r ** 2 + 1e-6), p=2, dim=1)

    sdf = k0 * (k0 - 1.0) / (k1 + 1e-6)
    return sdf


def compute_sdf_ellipsoids(points, ellipsoids_parameters):
    sdf = []
    for i, params in enumerate(ellipsoids_parameters):
        r, V, center = params
        sdf.append(compute_sdf_ellipsoid(points, center, r, V))
    return sdf


def compute_sdf_ellipsoids_batch(points, ellipsoids_parameters_batch):
    sdfs = []
    for b, ellipsoids_parameters in enumerate(ellipsoids_parameters_batch):
        sdfs.append(compute_sdf_ellipsoids(points[b], ellipsoids_parameters))
    return sdfs


def compute_intersection_loss_volume_2(ellipsoid_params_batch, points):
    """
    To minimize the intersection between ellipsoids, sample points inside ellipsoid,
    and then compute sdf at these points inside an ellipsoid wr.t other ellipsoids.
    :param sampled_points_batch: sampled points on ellipsoids surface.
    :param ellipsoid_params_batch: parameters of ellipsoids.
    """
    sdfs = compute_sdf_ellipsoids_batch(points, ellipsoid_params_batch)
    losses = []
    for b in range(len(sdfs)):
        num_ellipsoids = len(sdfs[b])
        if num_ellipsoids <= 1:continue
        sdf = sdfs[b]
        #sdf = compute_sdf_ellipsoids(points[b], ellipsoid_params_batch[b])
        sdf = torch.stack(sdf, 1)
        sdf = torch.clamp_max(sdf, -1e-3)

        # without detach all embedding collapse
        sdf = sdf - torch.min(sdf, 1, keepdim=True)[0].detach()
        loss = sdf ** 2
        losses.append(loss.mean())
    if losses == []:
        losses = torch.zeros(1, requires_grad=True).cuda()
        return losses  
    else:
        return torch.stack(losses).mean()


def compute_intersection_loss_volume_3(ellipsoid_params_batch, points, cuboid=False):
    """
    To minimize the intersection between ellipsoids, sample points inside ellipsoid,
    and then compute sdf at these points inside an ellipsoid wr.t other ellipsoids.
    :param sampled_points_batch: sampled points on ellipsoids surface.
    :param ellipsoid_params_batch: parameters of ellipsoids.
    """
    if cuboid:
        sdfs = compute_sdf_cuboid_batch(points, ellipsoid_params_batch)
    else:
        sdfs = compute_sdf_ellipsoids_batch(points, ellipsoid_params_batch)
    losses = []
    for b in range(len(sdfs)):
        num_ellipsoids = len(sdfs[b])
        if num_ellipsoids <= 1:continue
        # sdf = compute_sdf_ellipsoids(points[b], ellipsoid_params_batch[b])
        sdf = sdfs[b]
        sdf = torch.stack(sdf, 1)
        sdf = torch.clamp_max(sdf, -1e-3)

        # without detach all embedding collapse
        closest_sdf_index = torch.min(sdf, 1, keepdim=False)[1]

        index = torch.zeros(points[b].shape[0], len(sdfs[b])).long().cuda()
        # set the index of the ellipsoid to which the point belongs to zero
        # so that you can exclude that later.

        index[torch.arange(points[b].shape[0]), closest_sdf_index] = 1
        # import ipdb; ipdb.set_trace()
        sdf = scatter_mean(sdf, index, dim=1)

        sdf = sdf[:, 0] # exclude the 1 index because it contains the sdf of ellipsoid to which the point belongs.
        loss = sdf ** 2
        losses.append(loss.mean())

    if losses == []:
        losses = torch.zeros(1, requires_grad=True).cuda()
        return losses
    else:
        return torch.stack(losses).mean()


def compute_intersection_loss_volume_4(ellipsoid_params_batch, points):
    """
    To minimize the intersection between ellipsoids, sample points inside ellipsoid,
    and then compute sdf at these points inside an ellipsoid wr.t other ellipsoids.
    :param sampled_points_batch: sampled points on ellipsoids surface.
    :param ellipsoid_params_batch: parameters of ellipsoids.
    """
    sdfs = compute_sdf_ellipsoids_batch(points, ellipsoid_params_batch)
    losses = []
    for b in range(len(sdfs)):
        num_ellipsoids = len(sdfs[b])
        if num_ellipsoids == 1:continue
        sdf = sdfs[b]
        #sdf = compute_sdf_ellipsoids(points[b], ellipsoid_params_batch[b])
        sdf = torch.stack(sdf, 1)
        sdf = torch.clamp_max(sdf, -1e-3)

        # without detach all embedding collapse
        sdf = torch.sum(sdf ** 2, 1) - torch.min(sdf, 1, keepdim=True)[0] ** 2
        loss = sdf
        losses.append(loss.mean())
    if losses == []:
        losses = torch.zeros(1, requires_grad=True).cuda()
        return losses
    else:
        return torch.stack(losses).mean()


def prune_points(points, ellipsoid_param_batch, thres=-1e-3):
    """
    For chamfer distance calculation, we only want to include
    predicted points with sdf > threshold. This means not including
    predicted points that are much inside the iso surface formed by
    union of ellipsoids.
    This is also good for visualization, as you are only displaying
    surface points and not inside points.
    @param points:
    @param ellipsoid_param_batch:
    @return:
    """
    pruned_points = []

    # TODO the below sdfs are also computed in the intersection loss.
    # TODO can be inherited.
    with torch.no_grad():
        sdfs = compute_sdf_ellipsoids_batch(points, ellipsoid_param_batch)
    for b in range(len(sdfs)):
        with torch.no_grad():
            sdf = torch.stack(sdfs[b], 1)
            sdf = torch.min(sdf, 1)[0]

            # only use points are are on or near the surface
            indices = sdf > thres
        pruned_points.append(points[b][indices])
    return pruned_points


def compute_sdf_cuboid(points, center, r, V):
    """
    SDF of cuboids.
    @param points: points at which sdf needs to be calculated
    @param center: center of ellipsoid
    @param r: principal axis lengths
    @param V: principal axis
    @return: sdf at points
    """
    shifted_points = (V.T @ (points - center).T).T

    # note that the cuboid has the side of 2 * r
    q = torch.abs(shifted_points) - r
    sdf = torch.norm(torch.relu(q), p=2, dim=1) + torch.clamp_max(torch.max(q, 1)[0], 0.0)
    return sdf


def compute_sdf_cuboids(points, ellipsoids_parameters):
    sdf = []
    for i, params in enumerate(ellipsoids_parameters):
        r, V, center = params
        sdf.append(compute_sdf_cuboid(points, center, r, V))
    return sdf


def compute_sdf_cuboid_batch(points, ellipsoids_parameters_batch):
    sdfs = []
    for b, ellipsoids_parameters in enumerate(ellipsoids_parameters_batch):
        sdfs.append(compute_sdf_cuboids(points[b], ellipsoids_parameters))
    return sdfs
