from utils.geo_utils import decompose_camera_matrix, align_cameras
from datasets.Euclidean import get_raw_data
import torch
from utils.plot_utils import plot_cameras
import numpy as np

if __name__ == '__main__':
    scene = 'AlcatrazCourtyard'
    results_path = f'outputs/{scene}.npz'
    results = np.load(results_path)
    pts3d = results['Xs']
    Rs = results['Rs']
    ts = results['ts']

    M, Ns, Ps_gt = get_raw_data(scene)
    Rs_gt, ts_gt = decompose_camera_matrix(Ps_gt, torch.linalg.inv(Ns))
    Rs_fixed, ts_fixed, similarity_mat = align_cameras(Rs, Rs_gt, ts, ts_gt, True)
    pts3d_fixed = similarity_mat @ pts3d
    
    plot_cameras(Rs_fixed.cpu().numpy(), ts_fixed.cpu().numpy(), pts3d_fixed.cpu().numpy(), Rs_gt.cpu().numpy(), ts_gt.cpu().numpy(), save_path = f'outputs/point_cloud_{scene}.html')




