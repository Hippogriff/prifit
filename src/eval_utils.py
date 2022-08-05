import numpy as np

def mean_IOU_one_sample(pred, gt, C):
    IoU_part = 0.0
    for label_idx in range(C):
        locations_gt = (gt == label_idx)
        locations_pred = (pred == label_idx)
        I_locations = np.logical_and(locations_gt, locations_pred)
        U_locations = np.logical_or(locations_gt, locations_pred)
        I = np.sum(I_locations) + np.finfo(np.float32).eps
        U = np.sum(U_locations) + np.finfo(np.float32).eps
        IoU_part = IoU_part + I / U
    return IoU_part / C
