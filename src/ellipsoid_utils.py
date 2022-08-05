from src.ellipsoid_fitting import *
from pathlib import Path
import os, sys, torch, shutil
meanshift = MeanShift()
sampleellipse = SampleEllipsoid()
MAXCLUSTERS = 25  # 50
import ipdb

def guard_mean_shift(embedding, number_samples, quantile, iterations, max_num_clusters, kernel_type="gaussian"):
    """
    Some times if band width is small, number of cluster can be larger than 50, that
    but we would like to keep max clusters 50 as it is the nmax number in our dataset.
    in that case you increase the quantile to increase the band width to decrease
    the number of clusters.
    :param embedding: embedding of a shape, N x D
    :param number_samples: number of samples to use while computing bandwidth
    :param itertations: number of times to do mean shift updates.
    """
    while True:
        center, bandwidth, cluster_ids = meanshift.mean_shift(embedding, number_samples, quantile, iterations, kernel_type=kernel_type)
        #print('BW: {}, Center: {} '.format(bandwidth.item(), center.shape))
        #print('Center: ', center.shape)
        if torch.unique(cluster_ids).shape[0] > max_num_clusters:
            quantile *= 2
        else:
            break
    return center, bandwidth, cluster_ids


# Do clustering and get per point weights
def clustering(X, num_samples=1000, quantile=0.01, iterations=5, visualize=False, max_num_clusters=MAXCLUSTERS):
    """
    Cluster the input embedding.
    :param X: embedding of size B x N x D
    :param num_sampels: number of samples to use to compute the band width, more samples
    leads to better estimate, but slower.
    :param iterations: number of iterations to do mean shift clustering.
    :param visualize: whether to visualize cluster results.
    """
    weights_batch = []
    batch_size = X.shape[0]
    labels = []
    for b in range(batch_size):
        centers, bw, new_labels = guard_mean_shift(X[b], num_samples, quantile, iterations, max_num_clusters=max_num_clusters)
        weights = meanshift.membership(centers, X[b], bw).T
        #ipdb.set_trace()
        #print('before: ',torch.sum(weights, axis=0).data.cpu().numpy())
        if visualize:
            num_clusters = weights.shape[1]

            cluster_ids = torch.max(weights, 1)[1]
            
            eye = torch.eye(num_clusters).cuda()
            weights = eye[cluster_ids].float()
            weight_sum = torch.sum(weights, axis=0).data.cpu().numpy()
            #print('after: ', weight_sum)
            #print('after: {} == {}'.format(weights.shape[1], np.count_nonzero(weight_sum)))
            #if weights.shape[1] != np.count_nonzero(weight_sum):
                #print('Found')
                #print(weight_sum)
            #    weights_np = weights.data.cpu().numpy()
            #    to_delete = np.where(weight_sum == 0)[0]
            #    weights_np_new = np.delete(weights_np, to_delete, axis=1)
            #    #print('Before: ', weights.shape, weights.device)
            #    weights = torch.tensor(weights_np_new, dtype=weights.dtype).cuda()
            #    #print('Now: ', weights.shape, weights.device)
            
            #sys.stdout.flush()

        weights_batch.append(weights)
        labels.append(new_labels)
    # TODO add small epsilon to the weights to make things more stable.
    return weights_batch, labels


def sample_from_pred_params(ellipse_params_batch, N, batch_id = 0, seed = 0, visualize=False, class_list=[], quantile=0.05):

    """
    Sample fixed number of points for each ellipse in
    the batch of shapes.
    """
    resampled_points_batch = []
    batch_size = len(ellipse_params_batch)

    for batch_index in range(batch_size):
        resampled_points, resampled_points_2, resampled_points_3 = [], [], []

        ellipsoids_areas = []
        A, B, C, Vs, centers = [], [], [], [], []
        for i in range(len(ellipse_params_batch[batch_index])):
            a = ellipse_params_batch[batch_index][i][0][0]
            b = ellipse_params_batch[batch_index][i][0][1]
            c = ellipse_params_batch[batch_index][i][0][2]
            V = ellipse_params_batch[batch_index][i][1]
            center = ellipse_params_batch[batch_index][i][2]
            A.append(a)
            B.append(b)
            C.append(c)
            Vs.append(V)
            centers.append(center)
            area = compute_approximate_ellipsoid_area(a, b, c, p=1.585)
            ellipsoids_areas.append(area)
       
        total_sum = np.sum(ellipsoids_areas)
        weights = ellipsoids_areas / total_sum
        num_points = np.round(10000 * weights).astype(int)
        num_points[num_points <= 0] = 100

        for i in range(len(ellipse_params_batch[batch_index])):
            sampled_points, ellipse = sampleellipse.sample(A[i], B[i], C[i], centers[i], Vs[i], n=num_points[i])
            resampled_points.append(sampled_points)
        
        if len(resampled_points) > 0:
            resampled_points = torch.cat(resampled_points, 0)
        else:
            resampled_points = -1
            #continue

        # if visualize and torch.is_tensor(resampled_points):
        #     directory = 'pretrainmodelnet40q{}/ellipsoids'.format(quantile)
        #     if not os.path.exists(directory):
        #         os.makedirs(directory)
        #
        #     path = os.path.join(directory, 'batch_{}_{}_{}.xyz'.format(batch_id, batch_index, class_list[batch_index]))
        #     np.savetxt(path, resampled_points.data.cpu().numpy())

        resampled_points_batch.append(resampled_points)


    return resampled_points_batch


def visualize_fitted_ellipsoids(sampled_points):
    """
    Visualizes a list of points.
    :param sampled_points: list, with each element a point cloud of size N x 3.
    """
    pcds = []
    for s in sampled_points:
        pcd = visualize_point_cloud(s)
        pcd.paint_uniform_color(np.random.random(3))
        pcds.append(pcd)
    visualization.draw_geometries(pcds)


def to_one_hot(target, maxx=50):
    target = torch.from_numpy(target.astype(np.int64)).cuda()
    N = target.shape[0]
    target_one_hot = torch.zeros((N, maxx))

    target_one_hot = target_one_hot.cuda()
    target_t = target.unsqueeze(1)
    target_one_hot = target_one_hot.scatter_(1, target_t.long(), 1)
    return target_one_hot


def compute_approximate_ellipsoid_area(a, b, c, p):
    area = 4 * 3.142 * ((a * b) ** p + (b * c) ** p + (c * a) ** p) ** (1 / p)
    return area.item()


def sample_from_pred_params_cuboid(ellipse_params_batch, N, batch_id=0, seed=0, visualize=False, class_list=[]):
    """
    Sample fixed number of points for each ellipse in
    the batch of shapes.
    """
    resampled_points_batch = []
    batch_size = len(ellipse_params_batch)

    for batch_index in range(batch_size):
        resampled_points, resampled_points_2, resampled_points_3 = [], [], []
        ellipsoids_areas = []
        A, B, C, Vs, centers = [], [], [], [], []
        for i in range(len(ellipse_params_batch[batch_index])):
            a = ellipse_params_batch[batch_index][i][0][0]
            b = ellipse_params_batch[batch_index][i][0][1]
            c = ellipse_params_batch[batch_index][i][0][2]
            V = ellipse_params_batch[batch_index][i][1]
            center = ellipse_params_batch[batch_index][i][2]
            A.append(a)
            B.append(b)
            C.append(c)
            Vs.append(V)
            centers.append(center)

            # Note that the cuboids have sides 2 * a, 2 * b, 2 * c
            area = (8 * (a * b + b * c + c * a)).item()
            ellipsoids_areas.append(area)

        total_sum = np.sum(ellipsoids_areas)
        weights = ellipsoids_areas / total_sum
        num_points = np.round(10000 * weights).astype(int)
        num_points[num_points <= 0] = 100

        for i in range(len(ellipse_params_batch[batch_index])):
            sampled_points, ellipse = sampleellipse.sample_cuboid(A[i], B[i], C[i], centers[i], Vs[i], n=num_points[i])
            resampled_points.append(sampled_points)

        if len(resampled_points) > 0:
            resampled_points = torch.cat(resampled_points, 0)
        else:
            resampled_points = -1

        if visualize and torch.is_tensor(resampled_points):
            # directory = '/home/bbdash/shapes_aft_area{}/'.format(str(seed))
            # if not os.path.exists(directory):
            #     os.mkdir(directory)
            #
            # path = os.path.join(directory, 'batch_{}_{}_{}.xyz'.format(batch_id, batch_index, class_list[batch_index]))
            # np.savetxt(path, resampled_points.data.cpu().numpy())
            pass
        resampled_points_batch.append(resampled_points)

    return resampled_points_batch
