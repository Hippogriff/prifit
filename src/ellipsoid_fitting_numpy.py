import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_ellipsoid(pc):
    pc_centered = pc - pc.mean(axis=0)
    covariance_matrix = np.cov(pc_centered, y=None, rowvar=0, bias=True)
    _, eigen_vectors = np.linalg.eigh(covariance_matrix)
    def try_to_normalize(v):
        n = np.linalg.norm(v)
        if n < np.finfo(float).resolution:
            raise ZeroDivisionError
        return v / n
    r = try_to_normalize(eigen_vectors[:, 0])
    u = try_to_normalize(eigen_vectors[:, 1])
    f = try_to_normalize(eigen_vectors[:, 2])
    
    #Matrix might be a reflection, not rotation, let's fix that!
    if np.linalg.det(eigen_vectors) < 0:
        f = -f
    
    orientation = R.from_dcm(np.array((r, u, f)).T)
    
    #Make rotations consistent
    q = orientation.as_quat()
    if q[0] < 0:
        q *= -1
    orientation = R.from_quat(q)
    p_primes = pc_centered.dot(orientation.as_dcm())
    obb_min = np.min(p_primes, axis=0)
    obb_max = np.max(p_primes, axis=0)
    center = np.dot((obb_min + obb_max)/2.0, orientation.as_dcm()) + pc.mean(axis=0)
    size = np.abs(obb_max - obb_min)
    return size, center, orientation
ellipsoid = trimesh.creation.icosphere(subdivisions=5)
a=5
b=2
c=1
ellipsoid.vertices[:, 0] *= a
ellipsoid.vertices[:, 1] *= b
ellipsoid.vertices[:, 2] *= c
pts, _ = trimesh.sample.sample_surface_even(ellipsoid, count=5000)
s,_,_ = compute_ellipsoid(pts)
print(s/2.0)
