import torch
from open3d import *
from torch.autograd import Function
import numpy as np


Vector3dVector, Vector3iVector = utility.Vector3dVector, utility.Vector3iVector
EPS = float(np.finfo(np.float32).eps)
torch.manual_seed(2)
np.random.seed(2)


class LeastSquares:
    def __init__(self):
        pass

    def lstsq(self, A, Y, lamb=0.0):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """
        cols = A.shape[1]
        if np.isinf(A.data.cpu().numpy()).any():
            import ipdb; ipdb.set_trace()

        # Assuming A to be full column rank
        if cols == torch.matrix_rank(A):
            # Full column rank
            q, r = torch.qr(A)
            x = torch.inverse(r) @ q.transpose(1, 0) @ Y
        else:
            # rank(A) < n, do regularized least square.
            AtA = A.transpose(1, 0) @ A

            # get the smallest lambda that suits our purpose, so that error in
            # results minimized.
            lamb = best_lambda(AtA)
            A_dash = AtA + lamb * torch.eye(cols, device=A.get_device())
            Y_dash = A.transpose(1, 0) @ Y

            # if it still doesn't work, just set the lamb to be very high value.
            x = self.lstsq(A_dash, Y_dash, 1)
        return x


def best_lambda(A):
    """
    Takes an under determined system and small lambda value,
    and comes up with lambda that makes the matrix A + lambda I
    invertible. Assuming A to be square matrix.
    """
    lamb = 1e-6
    cols = A.shape[0]

    for i in range(7):
        A_dash = A + lamb * torch.eye(cols, device=A.get_device())
        if cols == torch.matrix_rank(A_dash):
            # we achieved the required rank
            break
        else:
            # factor by which to increase the lambda. Choosing 10 for performance.
            lamb *= 10
    return lamb


def compute_grad_V(U, S, V, grad_V, grad_S):
    N = S.shape[0]
    K = svd_grad_K(S)
    S = torch.eye(N).cuda(S.get_device()) * S.reshape((N, 1))
    inner = K.T * (V.T @ grad_V)
    inner = (inner + inner.T) / 2.0
    out = 2 * U @ S @ inner @ V.T

    # now compute the first, assuming partial derivative w.r.t to U is zero.
    grad_S = torch.eye(N).cuda(grad_S.get_device()) * grad_S.reshape((N, 1))
    first_term = U @ grad_S @ V.T
    out = first_term + out
    return out


def svd_grad_K(S):
    N = S.shape[0]
    s1 = S.view((1, N))
    s2 = S.view((N, 1))
    diff = s2 - s1
    plus = s2 + s1

    # TODO Look into it
    eps = torch.ones((N, N)) * 10 ** (-6)
    eps = eps.cuda(S.get_device())
    max_diff = torch.max(torch.abs(diff), eps)
    sign_diff = torch.sign(diff)

    K_neg = sign_diff * max_diff

    # gaurd the matrix inversion
    K_neg[torch.arange(N), torch.arange(N)] = 10 ** (-6)
    K_neg = 1 / K_neg
    K_pos = 1 / plus

    ones = torch.ones((N, N)).cuda(S.get_device())
    rm_diag = ones - torch.eye(N).cuda(S.get_device())
    K = K_neg * K_pos * rm_diag
    return K


class CustomSVD(Function):
    """
    Costum SVD to deal with the situations when the
    singular values are equal. In this case, if dealt
    normally the gradient w.r.t to the input goes to inf.
    To deal with this situation, we replace the entries of
    a K matrix from eq: 13 in https://arxiv.org/pdf/1509.07838.pdf
    to high value.
    Note: only applicable for the tall and square matrix and doesn't
    give correct gradients for fat matrix. Maybe transpose of the
    original matrix is requires to deal with this situation. Left for
    Note: It is also assumed that gradient w.r.t U is zero.
    """

    @staticmethod
    def forward(ctx, input):
        # Note: input is matrix of size m x n with m >= n.
        # Note: if above assumption is violated, the gradients
        # will be wrong.

        U, S, V = torch.svd(input, some=True)
        ctx.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_V):
        U, S, V = ctx.saved_tensors
        grad_input = compute_grad_V(U, S, V, grad_V, grad_S)
        return grad_input


customsvd = CustomSVD.apply


def standardize_points(points):
    Points = []
    stds = []
    Rs = []
    means = []
    batch_size = points.shape[0]

    for i in range(batch_size):
        point, std, mean, R = standardize_point(points[i])
        Points.append(point)
        stds.append(std)
        means.append(mean)
        Rs.append(R)

    Points = np.stack(Points, 0)
    return Points, stds, means, Rs


def standardize_point(point):
    mean = torch.mean(point, 0)[0]
    point = point - mean

    S, U = pca_numpy(point)
    smallest_ev = U[:, np.argmin(S)]
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    # axis aligns with x axis.
    point = R @ point.T
    point = point.T

    std = np.abs(np.max(point, 0) - np.min(point, 0))
    std = std.reshape((1, 3))
    point = point / (std + EPS)
    return point, std, mean, R


def rotation_matrix_a_to_b(A, B):
    """
    Finds rotation matrix from vector A in 3d to vector B
    in 3d.
    B = R @ A
    """
    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v) + EPS)
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w) + EPS)
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0],
                  [sin, cos, 0],
                  [0, 0, 1]])
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R


def pca_numpy(X):
    S, U = np.linalg.eig(X.T @ X)
    return S, U


def pca_torch(X):
    # TODO 2Change this to do SVD, because it is stable and computationally
    # less intensive.
    covariance = torch.transpose(X, 1, 0) @ X
    S, U = torch.eig(covariance, eigenvectors=True)
    return S, U


def reverse_all_transformations(points, means, stds, Rs):
    new_points = []
    for i in range(len(Rs)):
        new_points.append(reverse_all_transformation(points[i], means[i], stds[i], Rs[i]))
    new_points = np.stack(new_points, 0)
    return new_points


def reverse_all_transformation(point, mean, std, R):
    std = std.reshape((1, 3))
    new_points_scaled = point * std
    new_points_inv_rotation = np.linalg.inv(R) @ new_points_scaled.T
    new_points_final = new_points_inv_rotation.T + mean
    return new_points_final


def project_to_plane(points, a, d):
    a = a.reshape((3, 1))
    a = a / torch.norm(a, 2)
    # Project on the same plane but passing through origin
    projections = points - ((points @ a).permute(1, 0) * a).permute(1, 0)

    # shift the points on the plane back to the original d distance
    # from origin
    projections = projections + a.transpose(1, 0) * d
    return projections


def project_to_point_cloud(points, surface):
    """
    project points on to the surface defined by points
    """
    diff = (np.expand_dims(points, 1) - np.expand_dims(surface, 0)) ** 2
    diff = np.sum(diff, 2)
    return surface[np.argmin(diff, 1)]
