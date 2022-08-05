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
import sys, os

EPS = 1e-7

def weighted_ellipsoid_fitting(points, weights, batch_id=0, shape_id=0, cluster_id=0):
    """
    Given a point cloud, weighted fitting is done.
    :param points: points of size N x 3
    :param weights: per point weights of size N x 1
    """
    
    #size 3 x 3
    N = points.shape[0]
    sum_weights = torch.sum(weights)
    center = torch.sum(points * weights, 0) / sum_weights
    points = points - center
    
    weighted_var = (points * weights).T @ points / sum_weights    
    #Original
    #U, S, V = customsvd(weighted_var)

    #Patch for SVD convergence!
    l, h = weighted_var.shape
    noise = 1e-4 * weighted_var.mean() * torch.rand(l, h).cuda()
    
    try:
        with torch.no_grad():
            U, S, V = customsvd(weighted_var + noise)
            if S[0] / S[2] > 1e5:                             #original
            #if S[0] / S[2] > 1e4:
                print('SVD high cond no.!')
                sys.stdout.flush()
                return -1
        U, S, V = customsvd(weighted_var + noise)
        s, v = principal_axis_ellipsoid(points, weights, S, V, mode="slow")
        return s, v, center
    
    except RuntimeError as e:
        directory1 = 'SVD_Error_Ellipsoids/points'
        directory2 = 'SVD_Error_Ellipsoids/weights'
        try:
            if not os.path.exists(directory1):
                os.makedirs(directory1)
            if not os.path.exists(directory2):
                os.makedirs(directory2)
        except FileExistsError as f:
            pass 
        # path1 = os.path.join(directory1, 'batch-{}_shape-{}_cluster-{}.npy'.format(batch_id, shape_id, cluster_id))
        # path2 = os.path.join(directory2, 'batch-{}_shape-{}_cluster-{}.npy'.format(batch_id, shape_id, cluster_id))
        # np.save(path1, points.data.cpu().numpy())
        # np.save(path2, weights.data.cpu().numpy())
        #print(weights)
        print('SVD Convergence Error!')
        sys.stdout.flush()
        return -1       

    #s, v = principal_axis_ellipsoid(points, weights, S, V, mode="slow")
    #return s, v, center

def weighted_ellipsoids_fitting(points, weights, batch_id=0, shape_id=0):
    """
     Given a point cloud, weighted fitting is done corresponding to K clusters
    :param points: points of size N x 3
    :param weights: per point weights of size N x K
    """
    params = []
    for i in range(weights.shape[1]):
        # TODO: need to check if the sum of weights along column is not very small.
        # ignore that column if the sum is below a threshold, meaning.
        param = weighted_ellipsoid_fitting(points, weights[:, i:i+1], batch_id=batch_id, shape_id=shape_id, cluster_id=i)
        if param != -1:                             # and param != -2:
            params.append(param)
        #elif param == -1:
        #    directory1 = 'svd_error_embeddings'
        #    directory2 = 'svd_error_inputs'
        #    if not os.path.exists(directory1):
        #        os.makedirs(directory1)
        #    if not os.path.exists(directory2):
        #        os.makedirs(directory2)
        #    path1 = os.path.join(directory1, 'batch_{}_{}_{}_{}.txt'.format(svd_params['epoch_id'], svd_params['batch_id'], svd_params['device_id'], b))
        #    path2 = os.path.join(directory2, 'batch_{}_{}_{}_{}.txt'.format(svd_params['epoch_id'], svd_params['batch_id'], svd_params['device_id'], b))
        #    np.savetxt(path1, X[:, :].data.cpu().numpy())
        #    np.savetxt(path2, points[:, :].data.cpu().numpy())
        #    continue
        else:
            continue
    
    return params

def weighted_ellipsoid_fitting_batch(points, weights, batch_id=0):
    """
    Given a batch of a point cloud, weighted fitting is done.
    :param points: points od size B x N x 3
    :param weights: List of len B, each element is a tensor of size N x K_b
    """
    params = []
    var = 0
    B = points.shape[0]
    
    for b in range(B):
        params.append(weighted_ellipsoids_fitting(points[b], weights[b], batch_id=batch_id, shape_id=b))
   
    return params

def principal_axis_ellipsoid(points, weights, S, V, mode="slow"):
    """
    Given the singular value decomposition of covariance matrix,
    returns the legth of principal_axis.
    """
    if mode == "fast":
        return torch.sqrt(torch.clamp(S, min=1e-7)) * 1.732, V
    if mode == "slow":
        # weight the points
        points = points - torch.sum(points * weights, 0) / torch.sum(weights)
        points = points * weights
        
        # rotate the points in the new basis
        # make sure that the matrix is not a reflection matrix
        if torch.det(V.T) < 0:
           # print ("reflection matrix")
            V = torch.stack([V[:, 0], V[:, 1], -1 * V[:, 2]], 1)
        transformed_points = points @ V
        max_length, max_index = torch.max(transformed_points, 0)
        min_length, min_index = torch.min(transformed_points, 0)
        
        axis_length = torch.abs(max_length - min_length) / 2.0
        return axis_length, V


def create_synthetic_dataset(batch_size):
    points_batch = []
    weights_batch = []
    parameters = []
    centers_batch = []
    rotation_batch = []
    for b in range(batch_size):
        points = []
        weights = []
        params = []
        centers = []
        rotations = []
        for i in range(3):
            
            # generate a random ellipsoid
            ellipsoid = trimesh.creation.icosphere(subdivisions=5)
            a = np.random.choice(np.arange(2, 20), 1)[0]
            b = np.random.choice(np.arange(2, 20), 1)[0]
            c = np.random.choice(np.arange(2, 20), 1)[0]
            ellipsoid.vertices[:, 0] *= a
            ellipsoid.vertices[:, 1] *= b
            ellipsoid.vertices[:, 2] *= c
            
            params.append([a, b, c])
            
            pts, _ = trimesh.sample.sample_surface_even(ellipsoid, count=1000)

            
            r = R.from_euler('z', np.random.random() * 360, degrees=True)

            transformation = r.as_matrix()
            pts = pts[0:500]
            pts = pts @ transformation
            center = np.random.random((1, 3)) * np.max([a, b, c])
            pts = pts + center
            points.append(pts)
            wgt = np.zeros((500, 32), dtype=np.float32)
            wgt[:, i] = 1.0
            weights.append(wgt)
            centers.append(center)
            rotations.append(transformation)
        points = np.concatenate(points)
        weights = np.concatenate(weights)
        rotations = np.stack(rotations, 0)
        points_batch.append(points)
        weights_batch.append(weights)
        parameters.append(params)
        centers_batch.append(centers)
        rotation_batch.append(rotations)
    return np.stack(points_batch, 0), np.stack(weights_batch, 0), parameters, centers_batch, rotation_batch
