from src.utils import visualize_point_cloud

import trimesh
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from open3d import *
from src.utils import chamfer_distance_single_shape, chamfer_distance_kdtree
import sys
from src.guard import guard_acos


class SampleEllipsoid:
    def __init__(self):
        pass
    
    def sample(self, a, b, c, center, transformation, n=500):
        """
        Uniformly samples points on ellipsoid, where a, b, c are the lenghts
        of the principal axis and right eigen vector is transformation.
        1. Sampled points uniformly on an ellipsoid using trimesh
        2. Find u,v parameters of each sampled points.
        3. Compute corresponding points on the ellipsoid surface, but this
            time using torch, with a,b,c as parameter so that we can backprop.
        4. Transform the sampled points using "transformation" matrix.
        :param a, b, c: principal axis
        :param center: center of the ellipsoid
        :param transformation: principal axis of ellipsoid (or its transform),
        3 x 3 matrix.
        """
        ellipsoid = trimesh.creation.icosphere(subdivisions=5)
        ellipsoid.vertices[:, 0] *= a.item()
        ellipsoid.vertices[:, 1] *= b.item()
        ellipsoid.vertices[:, 2] *= c.item()
        points, _ = trimesh.sample.sample_surface_even(ellipsoid, count=n)

        # Sample Points from ellipsoid
        # x = a cos(u) sin(v)
        # y = b sin(u) sin(v)
        # z = c cos(v)

        # find parameters
        points = torch.from_numpy(points.astype(np.float32)).cuda() #(device=transformation.device)
        
        V = guard_acos(points[:, 2] / (c + 1e-6))
        U = torch.atan2(points[:, 1] / (b + 1e-6), points[:, 0] / (a + 1e-6))
        
        V = V.detach()
        U = U.detach()
        sampled_points = self.uniform_sample_points_on_ellipsoid(U, V, a, b, c)
        sampled_points = sampled_points @ transformation.T
        sampled_points = sampled_points + center
        return sampled_points, ellipsoid
    
    def uniform_sample_points_on_ellipsoid(self, U, V, a, b, c):
        """
        Given the parameters of a points from ellipsoid, assuming that the axis 
        of ellipsoid are aligned with standard basis, it returns corresponding points
        """
        x = a * torch.cos(U) * torch.sin(V)
        y = b * torch.sin(U) * torch.sin(V)
        z = c * torch.cos(V)
        return torch.stack([x, y, z], 1)

    def sample_cuboid(self, a, b, c, center, transformation, n=500):
        """
        Taking the parameters coming from svd, it samples cuboid uniformly such
        that it is possible to back propagate the gradients.
        @param a: side a
        @param b: side b
        @param c: side c
        @param center: center
        @param transformation: principal axis of cuboid
        @param n: number of points to sample.
        @return:
        """
        ellipsoid = trimesh.creation.box([2.0, 2.0, 2.0])
        sides_numpy = np.array([a.item(), b.item(), c.item()]).reshape((1, 3))
        sides_torch = torch.stack([a, b, c]).view(1, 3)

        # sample uniformly over the surface.
        ellipsoid.vertices = ellipsoid.vertices * sides_numpy

        points, _ = trimesh.sample.sample_surface_even(ellipsoid, count=n)

        # bring the points back to original unit size. Note here that point coordinates act
        # as parameterization of cuboid.
        points = points / (sides_numpy + 1e-6)

        # multiplied the uniformly sampled points with the side length in torch tensor format
        # to allow back propagation.
        sampled_points = torch.from_numpy(points.astype(np.float32)).cuda()  # (device=transformation.device)
        sampled_points = sampled_points * sides_torch
        sampled_points = sampled_points @ transformation.T
        sampled_points = sampled_points + center
        return sampled_points, ellipsoid


class Loss:
    def __init__(self):
        pass
    
    
    def loss(self, gt_points, sampled_points):
        """
        Computes chamfer distance loss between predicted and gt points
        :param gt_points: B x N x 3
        :param sampled_points: list of size B, each element is a tensor of size G x 3.
        """
        # compute chamfer distance both sided
        B = gt_points.shape[0]
        #B = len(sampled_points)
        index = np.random.choice(B)

        pcd1 = visualize_point_cloud(gt_points[index].data.cpu().numpy(), viz=False)
        pcd2 = visualize_point_cloud(sampled_points[index].data.cpu().numpy(), viz=False)

        pcd1.paint_uniform_color([0, 1, 0])
        pcd2.paint_uniform_color([0, 0, 1])
        visualization.draw_geometries([pcd1, pcd2])

        distance = []
        for b in range(B):
            if torch.is_tensor(sampled_points[b]):
                distance.append(chamfer_distance_kdtree(torch.unsqueeze(sampled_points[b], 0), torch.unsqueeze(gt_points[b], 0)))
            else:
                #print('-1')
                continue

        if distance == []:
            #print('No ellipsoid-no loss')
            return torch.zeros(1, requires_grad=True).cuda()
        else:
            distance = torch.stack(distance)
            return torch.mean(distance)
