import numpy as np
import loss_functions

# Return the top-k indices of a block
def largest_indices(blk, k):
    """Returns the n largest indices from a numpy array."""
    k = int(k)

    if k < 1:
        k = 1

    flat = blk.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(-flat[indices])]
    indices3d = np.unravel_index(indices, blk.shape)

    return np.array(indices3d).transpose()


def full_topk_optimization(ori_geo, ori_color, ori_norm, rec_geo, rec_color, num_points, geo_metric, color_metric, blk_size, color_weight=0.5, max_rho=10, patience=5):
    
    best_psnr = 0
    counter = 0
    best_rho = 0

    for k in np.arange(0.1, max_rho + 0.1, 0.1):

        tmp_idx = rec_geo[:round(num_points*k)]

        tmp_psnr_geo = loss_functions.psnr(geo_metric(ori_geo, tmp_idx, ori_norm), blk_size - 1)
        tmp_psnr_color = 0

        if color_metric is not None:
            tmp_psnr_color = loss_functions.psnr(color_metric(ori_geo, tmp_idx, ori_color, rec_color[:, tmp_idx[:, 0], tmp_idx[:, 1], tmp_idx[:, 2]].T), 1, geo=False)

        tmp_psnr = (1-color_weight) * tmp_psnr_geo + color_weight * tmp_psnr_color

        if tmp_psnr > best_psnr:
            best_psnr = tmp_psnr
            best_rho = k
            counter = 0
        else:
            counter = counter + 1
            if counter == patience:
                break

    return best_rho


def fast_topk_optimization(ori_geo, ori_color, ori_norm, rec_geo, rec_color, num_points, geo_metric, color_metric, blk_size, color_weight=0.5, max_rho=10, patience=5):
    
    search_range = np.arange(1,max_rho+1,1)

    for search_index in [0,1]:

        best_psnr = 0
        counter = 0
        best_rho = 0

        for k in search_range:

            tmp_idx = rec_geo[:round(num_points*k)]

            tmp_psnr_geo = loss_functions.psnr(geo_metric(ori_geo, tmp_idx, ori_norm), blk_size - 1)
            tmp_psnr_color = 0

            if color_metric is not None:
                tmp_psnr_color = loss_functions.psnr(color_metric(ori_geo, tmp_idx, ori_color, rec_color[:, tmp_idx[:, 0], tmp_idx[:, 1], tmp_idx[:, 2]].T), 1, geo=False)

            tmp_psnr = (1-color_weight) * tmp_psnr_geo + color_weight * tmp_psnr_color

            if tmp_psnr > best_psnr:
                best_psnr = tmp_psnr
                best_rho = k
                counter = 0
            else:
                counter = counter + 1
                if counter == patience:
                    break
        
        search_range = np.arange(best_rho-0.5,best_rho+0.5,0.1)

    return best_rho
