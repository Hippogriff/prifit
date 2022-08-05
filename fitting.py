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

from src.ellipsoid_utils import sample_from_pred_params, clustering


# In[2]:


# meanshift = MeanShift()
# sampleellipse = SampleEllipsoid()
batch_size = 6
# Number of points sampled per ellipsoid
N = 500

print ("Create synthetic dataset")
points, X, parameters, centers_batch, rotation_batch = create_synthetic_dataset(batch_size)
points = torch.from_numpy(points.astype(np.float32)).cuda()


X_ = torch.from_numpy(X.astype(np.float32)).cuda()
# points.requires_grad = True
# X_ = torch.rand_like(X_)
X_.requires_grad = True

X = torch.nn.functional.normalize(X_, dim=2, p=2)

print ("cluster point embedding")
weights_batch, labels = clustering(X)

pcd1 = visualize_point_cloud_from_labels(points[0].data.cpu().numpy(),
                                         labels=labels[0].data.cpu().numpy(),
                                         viz=True)

print ("ellipsoid fitting")
import ipdb; ipdb.set_trace()
ellipse_params_batch = weighted_ellipsoid_fitting_batch(points, weights_batch)


print ("samle points on the fitted ellipsoids")
resampled_points_batch = sample_from_pred_params(ellipse_params_batch, N)


print ("Compute Loss (Chamfer Distance)")
loss_cd = Loss()
l = loss_cd.loss(points, resampled_points_batch)
l.backward()

print ("visualization")
for batch_index in range(batch_size):
    # Input point cloud
    pcd1 = visualize_point_cloud(points[batch_index].data.cpu().numpy(), viz=False)
    pcd1.paint_uniform_color([1, 0, 0])
    
    # Reconstructed point cloud
    pcd2 = visualize_point_cloud(resampled_points_batch[batch_index].data.cpu().numpy(), viz=False)
    pcd2.paint_uniform_color([0, 0, 0])

    visualization.draw_geometries([pcd2, pcd1])
