"""
This defines a module for all sorts of visualization necessary for debugging and other
final visualization.
"""
import copy
from random import shuffle
from typing import List
import matplotlib.pyplot as plt
from open3d import *
from open3d import utility
from open3d import visualization
import numpy as np

from .utils import visualize_point_cloud

Vector3dVector, Vector3iVector = utility.Vector3dVector, utility.Vector3iVector
from transforms3d.affines import compose
from transforms3d.euler import euler2mat

PointCloud = geometry.PointCloud

# TODO Visualizing input and output in a grid
# TODO look at the meshutils
# TODO Other grid visualization
# TODO Visualize the spline surfaces
# TODO Find some representative shapes that are difficult, and can be used to benchmark algorithms
# TODO see how the surfaces reconstructs after the UV predictions

import numpy as np
from sklearn.manifold import TSNE


def convert_trimesh_to_o3d(mesh):
    o3d_mesh = geometry.TriangleMesh()
    o3d_mesh.vertices = utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh

def tsne(embedding, points):
    X_embedded = TSNE(n_components=3, perplexity=50).fit_transform(embedding)
    pcd = visualize_point_cloud(points)
    color = X_embedded - X_embedded.min()
    color = color / color.max()
    pcd.colors = utility.Vector3dVector(color)
    visualization.draw_geometries([pcd])
    return pcd


def plotall(images, cmap="Greys_r"):
    """
    Awesome function to plot figures in list of list fashion.
    Every list inside the list, is assumed to be drawn in one row.
    :param images: List of list containing images
    :param cmap: color map to be used for all images
    :return: List of figures.
    """
    figures = []
    num_rows = len(images)
    for r in range(num_rows):
        cols = len(images[r])
        f, a = plt.subplots(1, cols)
        for c in range(cols):
            a[c].imshow(images[r][c], cmap=cmap)
            a[c].title.set_text("{}".format(c))
            a[c].axis("off")
            a[c].grid("off")
        figures.append(f)
    return figures


def load_points_from_directory(path, suffix=".xyz", tessalate=False, random=True, max_models=50):
    pcds = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(suffix):
                pcds.append(root + "/" + f)
    if not random:
        pcds.sort()
    else:
        shuffle(pcds)
    pcds = pcds[0:max_models]
    for index, value in enumerate(pcds):
        pcds[index] = np.loadtxt(value)
    return pcds


def visualize_from_directory(path, suffix=".xyz", tessalate=False, random=True, max_models=50):
    pcds = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(suffix):
                pcds.append(root + "/" + f)
    if not random:
        pcds.sort()
    else:
        shuffle(pcds)
    pcds = pcds[0:max_models]
    for index, value in enumerate(pcds):
        pcds[index] = np.loadtxt(value)
    pcds = np.stack(pcds, 0)
    vis_batch_in_grid(pcds, tessalate)


def convert_into_open3d_format(points, tessellate=False):
    if tessellate:
        size_u = int(np.sqrt(points.shape[0]))
        pcd = tessalate_points(points[:, 0:3], size_u, size_u)
    else:
        pcd = PointCloud()
        size = points.shape[1]
        pcd.points = Vector3dVector(points[:, 0:3])
        if size > 3:
            pcd.colors = Vector3dVector(points[:, 3:] / 255.0)
    return pcd


def generate_grid(pcds):
    batch_size = len(pcds)

    height = int(np.sqrt(batch_size))
    width = int(batch_size // height)
    grids = []
    for i in range(int(height)):
        grid = []
        for j in range(int(width)):
            grid.append(pcds[i * width + j])
        grids.append(grid)

    grid = []
    for k in range(height * width, batch_size):
        grid.append(pcds[k])
    grids.append(grid)
    return grids


def visualize_compare_gt_pred(path_gt, path_pred, suffix=".xyz", tessalte=False):
    print(path_gt, path_pred)
    pcds_gt = []
    for root, dirs, files in os.walk(path_gt):
        for f in files:
            if f.endswith(suffix):
                pcds_gt.append(root + "/" + f)
    pcds_gt.sort()

    pcds_pred = []
    for root, dirs, files in os.walk(path_pred):
        for f in files:
            if f.endswith(suffix):
                pcds_pred.append(root + "/" + f)
    pcds_pred.sort()
    print(len(pcds_pred))
    for i in range(min(len(pcds_pred), len(pcds_gt))):
        pcds = []
        print(np.loadtxt(pcds_gt[i])[:, 0:3].shape)
        pts_pred = np.loadtxt(pcds_pred[i])[:, 0:3]
        pts_gt = np.loadtxt(pcds_gt[i])[:, 0:3]
        pcds.append()
        pcds.append()
        pcds = np.stack(pcds, 0)
        vis_batch_in_grid(pcds, tessalate)


def save_xyz(points, root_path, epoch, prefix, color=None):
    os.makedirs(root_path, exist_ok=True)
    batch_size = points.shape[0]
    for i in range(batch_size):
        if isinstance(color, np.ndarray):
            pcd = np.concatenate([points[i], color], 1)
        else:
            pcd = points[i]
        np.savetxt(root_path + "{}_{}_{}.xyz".format(prefix, epoch, i), pcd)


def save_xyz_continuous(points, root_path, id, prefix, color=None):
    """
    Saves xyz in continuous manner used for saving testing.
    """
    os.makedirs(root_path, exist_ok=True)
    batch_size = points.shape[0]
    for i in range(batch_size):
        if isinstance(color, np.ndarray):
            pcd = np.concatenate([points[i], color], 1)
        else:
            pcd = points[i]
        np.savetxt(root_path + "{}_{}.xyz".format(prefix, id * batch_size + i), pcd)


def custom_draw_geometry_load_option(pcds, render=False):
    R = euler2mat(-15 * 3.14 / 180, -35 * 3.14 / 180, 35)
    M = compose(T=(0, 0, 0), R=R, Z=(1, 1, 1))

    vis = visualization.Visualizer()
    vis.create_window()
    for pcd in pcds:
        pcd.transform(M)
        vis.add_geometry(pcd)
    vis.get_render_option().load_from_json("render_options.json")
    vis.run()
    if render:
        image = vis.capture_screen_float_buffer()
        vis.destroy_window()
        return image
    vis.destroy_window()


def save_images_from_list_pcds(pcds: List, pcd, path_template=None, transform=True):
    pcd = copy.deepcopy(pcd)
    R = euler2mat(45 * 3.14 / 180, 30 * 3.14 / 180, 0)
    M = compose(T=(0, 0, 0), R=R, Z=(1, 1, 1))
    vis = visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().load_from_json("render_options.json")

    images = []
    for index, p in enumerate(pcds):
        if index == 0:
            if transform:
                pcd.transform(M)
            vis.add_geometry(pcd)
            vis.run()
        else:
            pcd.points = p.points
            pcd.colors = p.colors
            pcd.normals = p.normals
            if transform:
                pcd.transform(M)
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
        image = vis.capture_screen_float_buffer()
        # plt.imsave(path_template.format(index), image)
        images.append(image)
    return images


def save_images_from_list_pcds_meshes(pcds: List, vis, pcd, path_template=None):
    R = euler2mat(-15 * 3.14 / 180, -35 * 3.14 / 180, 35)
    M = compose(T=(0, 0, 0), R=R, Z=(1, 1, 1))

    for index, p in enumerate(pcds):
        if index == 0:
            pcd.vertices = p.vertices
            pcd.triangles = p.triangles
            pcd.transform(M)
            pcd.compute_vertex_normals()
            vis.add_geometry(pcd)
            vis.run()
        else:
            pcd.vertices = p.vertices
            pcd.triangles = p.triangles
            pcd.transform(M)
            pcd.compute_vertex_normals()
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
        image = np.array(vis.capture_screen_float_buffer())

        plt.imsave(path_template.format(index), image[200:-200, 200:-200])


def save_images_shape_patches_collection(Pcds: List, path_template=None, transform=True):
    """
    Given a list of list, where the inner list containts open3d meshes
    Now, the task is to consider the inner list contains surface patches
    for each segment of the shape. We need to visualize the shape at different
    rotations.
    """
    import os

    # os.makedirs(path_template, exist_ok=True)
    R = euler2mat(45 * 3.14 / 180, 30 * 3.14 / 180, 0)
    M = compose(T=(0, 0, 0), R=R, Z=(1, 1, 1))
    images = []

    for index, shape_list in enumerate(Pcds):
        vis = visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().load_from_json("render_options.json")
        for s in shape_list:
            vis.add_geometry(s)

        for i in range(1):
            for s in shape_list:
                if transform:
                    s.transform(M)
                vis.add_geometry(s)
            vis.poll_events()

            vis.update_renderer()
            vis.run()
            image = np.array(vis.capture_screen_float_buffer())
            images.append(image)
        vis.destroy_window()
    return images


def grid_pcd_visulation_save_images(pcds: List, pcd, vis=None, first=True):
    """
    Assuming the the elements of List are itself point clouds of numpy arrays
    """
    # First normalize them
    R = euler2mat(-75 * 3.14 / 180, -75 * 3.14 / 180, 0)
    M = compose(T=(0, 0, 0), R=R, Z=(5, 5, 5))
    half_length = np.min((len(pcds) // 2, 10))

    for index, p in enumerate(pcds):
        p.points = Vector3dVector(
            p.points - np.mean(np.array(p.points), 0).reshape(1, 3)
        )
        pcds[index] = p

    points = []
    colors = []
    for j in range(2):
        for i in range(half_length):
            p = pcds[j * half_length + i]
            shift_y = j * 1.3
            shift_x = i * 1.3
            temp = np.array(p.points)
            temp = np.matmul(temp, M[0:3, 0:3])
            temp = temp + np.matmul(np.array([shift_x, shift_y, 0]), M[0:3, 0:3])
            points.append(temp)
            colors.append(p.colors)

    points = np.concatenate(points, 0)
    colors = np.concatenate(colors, 0)
    pcd.points = Vector3dVector(points)
    pcd.colors = Vector3dVector(colors)

    if first:
        vis = Visualizer()
        vis.create_window()
        vis.get_render_option().load_from_json("renderoption.json")
        vis.add_geometry(pcd)
        vis.run()
        first = False
    else:
        print("here")
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.run()

    image = vis.capture_screen_float_buffer()
    return image, vis, first


class VizGridAll:
    def __init__(self):
        pass

    def load_file_paths(path, file_type):
        # TODO Use wild card to get files
        retrieved_path = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(file_type):
                    retrieved_path.append(root + "/" + f)
        return retrieved_path

    def load_files(retrieved_path, file_type):
        if file_type == "xyz":
            retrieved_path.sort()
            pcds = []
            for index, value in enumerate(retrieved_path):
                pcds[index] = np.loadtxt(value)
            pcds = np.stack(pcds, 0)
            vis_batch_in_grid(pcds)
        elif file_type == ".ply":
            print("Not Impletementd Yet!")

def grid_points_lists_visulation(pcds: List, viz=False):
    """
    Every list contains a list of points clouds to be visualized.
    Every element of the list of list is a point cloud in pcd format
    """
    # First normalize them
    for pcd_list in pcds:
        for index, p in enumerate(pcd_list):
            maxx = np.max(np.array(p.points), 0)
            minn = np.min(np.array(p.points), 0)
            points = np.array(p.points) - np.mean(np.array(p.points), 0).reshape(1, 3)
            points = points / np.linalg.norm(maxx - minn)
            p.points = Vector3dVector(points)

    new_meshes = []
    for j in range(len(pcds)):
        for i in range(len(pcds[j])):
            p = pcds[j][i]
            shift_y = j * 1.0
            shift_x = i * 1.0
            p.points = Vector3dVector(
                np.array(p.points) + np.array([shift_x, shift_y, 0])
            )
            new_meshes.append(p)
    if viz:
        visualization.draw_geometries(new_meshes)
    return new_meshes


def grid_meshes_lists_visulation(pcds, viz=False) -> None:
    """
    Every list contains a list of points clouds to be visualized.
    Every element of the list of list is a point cloud in pcd format
    """
    # First normalize them
    for pcd_list in pcds:
        for index, p in enumerate(pcd_list):
            maxx = np.max(np.array(p.vertices), 0)
            minn = np.min(np.array(p.vertices), 0)
            points = np.array(p.vertices) - np.mean(np.array(p.vertices), 0).reshape(1, 3)
            points = points / np.linalg.norm(maxx - minn)
            p.vertices = Vector3dVector(points)

    new_meshes = []
    for j in range(len(pcds)):
        for i in range(len(pcds[j])):
            p = pcds[j][i]
            shift_y = j * 1.2
            shift_x = i * 1.2
            p.vertices = Vector3dVector(
                np.array(p.vertices) + np.array([shift_x, shift_y, 0])
            )
            new_meshes.append(p)
    if viz:
        visualization.draw_geometries(new_meshes)
    return new_meshes
