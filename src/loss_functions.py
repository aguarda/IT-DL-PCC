import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


def focal_loss(x_true, x_pred, gamma=2, alpha=0.95, total=True):
    pt_1 = torch.where(torch.eq(x_true, 1), x_pred, torch.ones_like(x_pred))
    pt_0 = torch.where(torch.eq(x_true, 0), x_pred, torch.zeros_like(x_pred))

    pt_1 = torch.clamp(pt_1, 1e-3, .999)
    pt_0 = torch.clamp(pt_0, 1e-3, .999)

    if total:
        return -torch.sum(alpha * torch.pow(1. - pt_1, gamma) * torch.log(pt_1)) - torch.sum((1-alpha) * torch.pow(pt_0, gamma) * torch.log(1. - pt_0))
    else:
        return -torch.mean(alpha * torch.pow(1. - pt_1, gamma) * torch.log(pt_1)) - torch.mean((1-alpha) * torch.pow(pt_0, gamma) * torch.log(1. - pt_0))


def mse_rgb(x_true, x_pred, num_pts):
    pt_true = torch.mul(x_true[:, 0:1, :, :, :], x_true[:, 1:, :, :, :])
    pt_pred = torch.mul(x_true[:, 0:1, :, :, :], x_pred[:, 1:, :, :, :])

    return torch.div(torch.sum(torch.square(pt_true - pt_pred)), num_pts)


def rmse_rgb(x_true, x_pred, num_pts):
    pt_true = torch.mul(x_true[:, 0:1, :, :, :], x_true[:, 1:, :, :, :])
    pt_pred = torch.mul(x_true[:, 0:1, :, :, :], x_pred[:, 1:, :, :, :])

    return -torch.log(torch.div(torch.sqrt(torch.sum(torch.square(pt_true - pt_pred))), num_pts))


def mse_yuv(x_true, x_pred, num_pts):
    # RGB to YUV conversion BT.709
    true_y = torch.mul(x_true[:, 0, :, :, :], (0.2126 * x_true[:, 1, :, :, :] + 0.7152 * x_true[:, 2, :, :, :] + 0.0722 * x_true[:, 3, :, :, :]))
    true_u = torch.mul(x_true[:, 0, :, :, :], (-0.1146 * x_true[:, 1, :, :, :] - 0.3854 * x_true[:, 2, :, :, :] + 0.5000 * x_true[:, 3, :, :, :]) + 0.5000)
    true_v = torch.mul(x_true[:, 0, :, :, :], (0.5000 * x_true[:, 1, :, :, :] - 0.4542 * x_true[:, 2, :, :, :] - 0.0458 * x_true[:, 3, :, :, :]) + 0.5000)

    pred_y = torch.mul(x_true[:, 0, :, :, :], (0.2126 * x_pred[:, 1, :, :, :] + 0.7152 * x_pred[:, 2, :, :, :] + 0.0722 * x_pred[:, 3, :, :, :]))
    pred_u = torch.mul(x_true[:, 0, :, :, :], (-0.1146 * x_pred[:, 1, :, :, :] - 0.3854 * x_pred[:, 2, :, :, :] + 0.5000 * x_pred[:, 3, :, :, :]) + 0.5000)
    pred_v = torch.mul(x_true[:, 0, :, :, :], (0.5000 * x_pred[:, 1, :, :, :] - 0.4542 * x_pred[:, 2, :, :, :] - 0.0458 * x_pred[:, 3, :, :, :]) + 0.5000)

    mse_y = torch.div(torch.sum(torch.square(true_y - pred_y)), num_pts)
    mse_u = torch.div(torch.sum(torch.square(true_u - pred_u)), num_pts)
    mse_v = torch.div(torch.sum(torch.square(true_v - pred_v)), num_pts)

    return torch.pow(mse_y, 6/8) * torch.pow(mse_u, 1/8) * torch.pow(mse_v, 1/8)


def weighted_bce(x_true, x_pred, alpha=0.95, total=True):
    pt_1 = torch.where(torch.eq(x_true, 1), x_pred, torch.ones_like(x_pred))
    pt_0 = torch.where(torch.eq(x_true, 0), x_pred, torch.zeros_like(x_pred))

    pt_1 = torch.clamp(pt_1, 1e-3, .999)
    pt_0 = torch.clamp(pt_0, 1e-3, .999)

    if total:
        return -torch.sum(alpha * torch.log(pt_1)) - torch.sum((1-alpha) * torch.log(1. - pt_0))
    else:
        return -torch.mean(alpha * torch.log(pt_1)) - torch.mean((1-alpha) * torch.log(1. - pt_0))


# For evaluation only
def point2point(x_ori, x_rec, n_ori=None):
    # Check if x_rec is empty
    if x_rec.size == 0:
        return np.inf

    # Set x_ori as the reference. Loop over x_rec and find nearest neighbor in x_ori
    nbrsA = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(x_ori)
    distBA, _ = nbrsA.kneighbors(x_rec)
    mseBA = np.square(distBA).mean()
    # Set x_rec as the reference. Loop over x_ori and find nearest neighbor in x_rec
    nbrsB = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(x_rec)
    distAB, _ = nbrsB.kneighbors(x_ori)
    mseAB = np.square(distAB).mean()
    # Symmetric total mse
    mse_sym = np.maximum(mseBA, mseAB)

    return mse_sym


# For evaluation only
def point2plane(x_ori, x_rec, n_ori):
    # Check if x_rec is empty
    if x_rec.size == 0:
        return np.inf

    # Set x_ori as the reference. Loop over x_rec and find nearest neighbor in x_ori
    nbrsA = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(x_ori)
    idxBA = np.squeeze(nbrsA.kneighbors(x_rec, return_distance=False))
    distBA = np.sum((x_ori[idxBA] - x_rec) * n_ori[idxBA], axis=1)
    mseBA = np.square(distBA).mean()
    # Set x_rec as the reference. Loop over x_ori and find nearest neighbor in x_rec
    nbrsB = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(x_rec)
    idxAB = np.squeeze(nbrsB.kneighbors(x_ori, return_distance=False))
    distAB = np.sum((x_rec[idxAB] - x_ori) * n_ori, axis=1)
    mseAB = np.square(distAB).mean()
    # Symmetric total mse
    mse_sym = np.maximum(mseBA, mseAB)

    return mse_sym


# For evaluation only
def point2point_yuv(x_ori, x_rec, x_ori_colors, x_rec_colors):
    # Check if x_rec is empty
    if x_rec.size == 0:
        return np.inf

    # RGB to YUV conversion BT.709
    ori_y = (0.2126 * x_ori_colors[:, 0] + 0.7152 * x_ori_colors[:, 1] + 0.0722 * x_ori_colors[:, 2])
    ori_u = (-0.1146 * x_ori_colors[:, 0] - 0.3854 * x_ori_colors[:, 1] + 0.5000 * x_ori_colors[:, 2]) + 0.5000
    ori_v = (0.5000 * x_ori_colors[:, 0] - 0.4542 * x_ori_colors[:, 1] - 0.0458 * x_ori_colors[:, 2])  + 0.5000

    rec_y = (0.2126 * x_rec_colors[:, 0] + 0.7152 * x_rec_colors[:, 1] + 0.0722 * x_rec_colors[:, 2])
    rec_u = (-0.1146 * x_rec_colors[:, 0] - 0.3854 * x_rec_colors[:, 1] + 0.5000 * x_rec_colors[:, 2]) + 0.5000
    rec_v = (0.5000 * x_rec_colors[:, 0] - 0.4542 * x_rec_colors[:, 1] - 0.0458 * x_rec_colors[:, 2]) + 0.5000

    # Set x_ori as the reference. Loop over x_rec and find nearest neighbor in x_ori
    nbrsA = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(x_ori)
    idxBA = np.squeeze(nbrsA.kneighbors(x_rec, return_distance=False))
    mseBA_y = np.mean(np.square(ori_y[idxBA] - rec_y))
    mseBA_u = np.mean(np.square(ori_u[idxBA] - rec_u))
    mseBA_v = np.mean(np.square(ori_v[idxBA] - rec_v))
    # Set x_rec as the reference. Loop over x_ori and find nearest neighbor in x_rec
    nbrsB = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(x_rec)
    idxAB = np.squeeze(nbrsB.kneighbors(x_ori, return_distance=False))
    mseAB_y = np.mean(np.square(ori_y - rec_y[idxAB]))
    mseAB_u = np.mean(np.square(ori_u - rec_u[idxAB]))
    mseAB_v = np.mean(np.square(ori_v - rec_v[idxAB]))
    # Symmetric mse
    mse_sym_y = np.maximum(mseBA_y, mseAB_y)
    mse_sym_u = np.maximum(mseBA_u, mseAB_u)
    mse_sym_v = np.maximum(mseBA_v, mseAB_v)
    # Final YUV mse
    mse_sym_yuv = np.power(mse_sym_y, 6/8) * np.power(mse_sym_u, 1/8) * np.power(mse_sym_v, 1/8)

    return mse_sym_yuv


# For evaluation only
def point2point_rgb(x_ori, x_rec, x_ori_colors, x_rec_colors):
    # Check if x_rec is empty
    if x_rec.size == 0:
        return np.inf

    # Set x_ori as the reference. Loop over x_rec and find nearest neighbor in x_ori
    nbrsA = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(x_ori)
    idxBA = np.squeeze(nbrsA.kneighbors(x_rec, return_distance=False))
    mseBA = np.mean(np.square(x_ori_colors[idxBA, :] - x_rec_colors))
    # Set x_rec as the reference. Loop over x_ori and find nearest neighbor in x_rec
    nbrsB = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(x_rec)
    idxAB = np.squeeze(nbrsB.kneighbors(x_ori, return_distance=False))
    mseAB = np.mean(np.square(x_ori_colors - x_rec_colors[idxAB, :]))
    # Symmetric mse
    mse_sym = np.maximum(mseBA, mseAB)

    return mse_sym


def psnr(mse, peak_value, geo=True):
    if mse == 0:
        return float("inf")
    elif np.isinf(mse):
        return 0

    if geo:
        return 10 * np.log10(3 * np.square(peak_value) / mse)
    else:
        return 10 * np.log10(np.square(peak_value) / mse)


def get_metrics(topk_metrics='d1yuv', with_color=False):

    if 'd2' in topk_metrics.lower():
        geo_p2p = point2plane
    else:
        geo_p2p = point2point

    color_p2p = None

    if with_color:        
        if 'rgb' in topk_metrics.lower():
            color_p2p = point2point_rgb
        else:
            color_p2p = point2point_yuv
        
    return geo_p2p, color_p2p

