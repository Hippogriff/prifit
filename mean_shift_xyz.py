from open3d import *
import torch
import numpy as np
from src.ellipsoid_utils import clustering, sample_from_pred_params
from src.utils import visualize_point_cloud_from_labels, visualize_point_cloud
from convex_loss import weighted_ellipsoid_fitting_batch
from sklearn.cluster import MeanShift
import numpy as np
from src.VisUtils import grid_points_lists_visulation
import copy
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.affines import compose
import time


def update_point_cloud(pcd, p, M):
    pcd.points = p.points
    pcd.normals = p.normals
    pcd.colors = p.colors
    pcd.transform(M)


def grid_points_lists_visulation(pcds, viz=False):
    """
    Every list contains a list of points clouds to be visualized.
    Every element of the list of list is a point cloud in pcd format
    """
    print("ola")
    # First normalize them
    for pcd_list in pcds:
        for index, p in enumerate(pcd_list):
            maxx = np.max(np.array(p.points), 0)
            minn = np.min(np.array(p.points), 0)
            points = np.array(p.points) - np.mean(np.array(p.points), 0).reshape(1, 3)
            points = points / np.max(maxx - minn)
            p.points = utility.Vector3dVector(points)

    new_meshes = []
    for j in range(len(pcds)):
        for i in range(len(pcds[j])):
            p = pcds[j][i]
            shift_y = j * 1.1
            shift_x = i * 1.1
            p.points = utility.Vector3dVector(
                np.array(p.points) + np.array([shift_x, shift_y, 0])
            )
            new_meshes.append(p)
    if viz:
        visualization.draw_geometries(new_meshes)
    return new_meshes


def render_images_from_list_pcds_points(pcds, sleep=1, compute_normals=None):
    """
    Renders meshes/point-clouds. Assumes all meshes/point clouds are normalized
    to be zero center.
    Args:
        pcds: List containing mixture of point cloud or meshes in open3d format
    Returns:
        images: List containing rendered images
    """
    R = euler2mat(60 * 3.14 / 180, -15 * 3.14 / 180, 0)
    M = compose(T=(0, 0, 0), R=R, Z=(1, 1, 1))
    pcd = pcds[0]
    images = []
    vis = visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().load_from_json("renderoptions.json")

    for index, p in enumerate(pcds):
        if index == 0:
            update_point_cloud(pcd, p, M)
            vis.add_geometry(pcd)
            vis.run()
        else:
            update_point_cloud(pcd, p, M)

            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

        time.sleep(sleep)
        image = np.array(vis.capture_screen_float_buffer())
        images.append(image)

        # plt.imsave(path_template.format(index), image[200:-200, 200:-200])
    return images


labels = []
weights_batch = []

points_ = np.load("points.npy")
embedding = np.load("embed.npy")

X = torch.from_numpy(points_).cuda()
X = X.permute(0, 2, 1)
points = torch.from_numpy(points_).cuda()

def cluster(x):
    clustering = MeanShift(bandwidth=0.15).fit(x)
    return clustering.labels_

for i in range(24):
    labels.append(torch.from_numpy(cluster(points_[i].T)))
    temp = torch.zeros(2048, torch.max(labels[i]) + 1).cuda()
    temp[torch.arange(2048), labels[i]] = 1.0
    weights_batch.append(temp)
    print (temp.shape)
ellipse_params_batch = weighted_ellipsoid_fitting_batch(points.permute(0, 2, 1), weights_batch)

weights_batch, labels = clustering(X, 2048, quantile=0.01, iterations=40)
for i in range(len(weights_batch)):
    print (i)
    temp = torch.zeros(2048, torch.max(labels[i]) + 1).cuda()
    temp[torch.arange(2048), labels[i]] = 1.0
    weights_batch[i] = temp

ellipse_params_batch = weighted_ellipsoid_fitting_batch(points.permute(0, 2, 1), weights_batch)
resampled_points_batch = sample_from_pred_params(ellipse_params_batch, N=500)

recon_xyz_pcds = []
recon_embedding_pcds = []
gt_pcds = []
for i in range(24):
    pcd1 = visualize_point_cloud(resampled_points_batch[i].data.cpu().numpy(), viz=True)
    pcd1.paint_uniform_color([1, 0, 0])
    recon_xyz_pcds.append(pcd1)
