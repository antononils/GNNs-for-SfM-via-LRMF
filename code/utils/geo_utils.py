from utils import dataset_utils
import torch
import numpy as np
import dask.array as da
from utils.general_utils import nonzero_safe

def xs_valid_points(xs):
    """

    :param xs: [m,n,2]
    :return: A boolean matrix of the visible 2d points
    """
    return dataset_utils.get_M_valid_points(xs)
def get_camera_matrix(R, t, K):
    """
    Get the camera matrix as described in paper
    :param R: Orientation Matrix
    :param t: Camera Position   
    :param K: Intrinsic parameters
    :return: Camera matrix
    """
    if isinstance(R, np.ndarray):
        return K @ R.T @ np.concatenate((np.eye(3), -t.reshape(3, 1)), axis=1)
    else:
        return K @ R.T @ torch.cat((torch.eye(3), -t.view(3, 1)), dim=1)
    
def batch_get_camera_matrix_from_rtk(Rs, ts, Ks):
    n = len(Rs)
    if isinstance(Rs, np.ndarray):
        Ps = np.zeros([n,3,4])
    else:
        Ps = torch.zeros([n, 3, 4])
    for i,r,t,k in zip(np.arange(n),Rs,ts,Ks):
        Ps[i] = get_camera_matrix(r,t,k)
    return Ps

def normalize_points_cams(Ps, xs, Ns):
    """
    Normalize the points and the cameras using the matrices in N.
    if :
    xs[i,j] ~ P[i] @ X[j]
    than so is:
     N[i] @ xs[i,j] ~ N[i] @ P[i] @ X[j]
    :param Ps:  [m,3,4]
    :param xs:  [m,n,2] or [m,n,3]
    :param Ns:  [m,3,3]
    :return:  norm_P, norm_x
    """
    assert isinstance(xs, np.ndarray)
    m, n, d = xs.shape
    xs_3 = np.concatenate([xs, np.ones([m, n, 1])], axis=2) if d == 2 else xs
    norm_P = np.zeros_like(Ps)
    norm_x = np.zeros_like(xs)
    for i in range(m):
        norm_P[i] = Ns[i] @ Ps[i]  # [3,3] @ [3,4]
        norm_points = (Ns[i] @ xs_3[i].T).T  # ([3,3] @ [3,n]) -> [n,3]
        norm_points[norm_points[:,-1]==0,-1] = 1
        norm_points = norm_points / norm_points[:, -1].reshape([-1,1])
        if d == 2:
            norm_x[i] = norm_points[:,:2]
    return norm_P, norm_x

def pflat(x):
    return x / x[-1, :]

def dlt_triangulation(Ps, xs, visible_points, simplified_dlt=False, return_V_H=False):
    """
    Use  linear triangulation to find the points X[j] such that  xs[i,j] ~ P[i] @ X[j]
    :param Ps:  [m,3,4]
    :param xs: [m,n,2] or [m,n,3]
    :param visible_points: [m,n] a boolean matrix of which cameras see which points
    :return: Xs [n,4] normalized such the X[j,-1] == 1
    """
    m, n, _ = xs.shape
    X = np.zeros([n,4])
    if return_V_H:
        V_H_ret = []
    for i in range(n):
        cameras_showing_ind = np.where(visible_points[:, i])[0]  # The cameras that show this point
        num_cam_show = len(cameras_showing_ind)
        if num_cam_show < 2:
            X[i] = np.nan
            continue
        if simplified_dlt:
            A = np.zeros([2 * num_cam_show, 4])
        else:
            A = np.zeros([3 * num_cam_show, num_cam_show + 4])
        for j, cam_index in enumerate(cameras_showing_ind):
            xij = xs[cam_index, i, :2]
            Pj = Ps[cam_index]
            if simplified_dlt:
                A[2*j     : 2*j + 1, :] = xij[0]*Pj[2, :] - Pj[0, :]
                A[2*j + 1 : 2*j + 2, :] = xij[1]*Pj[2, :] - Pj[1, :]
            else:
                A[3 * j:3 * (j + 1), :4] = Pj
                A[3 * j:3 * j + 2, 4 + j] = -xij
                A[3 * j + 2, 4 + j] = -1

        if num_cam_show > 40:
            [U, S, V_H] = da.linalg.svd(da.from_array(A))  # in python svd returns V conjugate! so we need the last row and not column
            X[i] = pflat(V_H[-1, :4].compute().reshape([-1, 1])).squeeze()
            if return_V_H:
                V_H_ret.append((np.diag(S.compute()), V_H.compute()))
        else:
            [U, S, V_H] = np.linalg.svd(A)  # in python svd returns V conjugate! so we need the last row and not column
            X[i] = pflat(V_H[-1, :4].reshape([-1, 1])).squeeze()
            if return_V_H:
                V_H_ret.append((np.diag(S), V_H))

    if return_V_H:
        return X, V_H_ret
    return X

def reprojection_error_with_points(Ps, Xs, xs, visible_points=None):
    """
    :param Ps: [m,3,4]
    :param Xs: [n,3] or [n,4]
    :param xs: [m,n,2]
    :return: errors [m,n]
    """
    m,n,d = xs.shape
    _, D = Xs.shape
    X4 = np.concatenate([Xs, np.ones([n,1])], axis=1) if D == 3 else Xs

    if visible_points is None:
        visible_points = xs_valid_points(xs)

    projected_points = Ps @ X4.T  # [m,3,4] @ [4,n] -> [m,3,n]
    projected_points = projected_points.swapaxes(1, 2)
    visible_points_idx = nonzero_safe(visible_points)
    projected_points[visible_points_idx[0], visible_points_idx[1], :] = projected_points[visible_points_idx[0], visible_points_idx[1], :] / projected_points[visible_points_idx[0], visible_points_idx[1], -1][:, None]
    errors = np.linalg.norm(xs[:,:,:2] - projected_points[:,:,:2], axis=2)
    errors[~visible_points] = np.nan
    return errors

def get_normalization_matrix(pts):
    if isinstance(pts, np.ndarray):
        norm_mat = np.eye(3)
        m = np.mean(pts[:2, :], axis=1)
        s = 1. / np.std(pts[:2, :], axis=1)
        norm_mat[0, 0] = s[0]
        norm_mat[1, 1] = s[1]
        norm_mat[:2, 2] = -s * m
    else:
        pts = pts.unique(dim=1)
        norm_mat = torch.eye(3)
        m = torch.mean(pts[0:2, :], dim=1)
        s = 1. / torch.std(pts[0:2, :], dim=1)
        norm_mat[0, 0] = s[0]
        norm_mat[1, 1] = s[1]
        norm_mat[0:2, 2] = -s * m
    return norm_mat


def decompose_camera_matrix(Ps, Ks=None, inverse_direction_camera2global=True):
    """
    Given camera matrices Ps, first normalize with the inverse of a given calibration matrix (if not provided, assumes camera calibrated already).
    Next extract R & t components from the P_norm = [R t]
    Finally returns Rs=R.T as well as camera centers, determined by ts = -R.T @ t.
    """
    if isinstance(Ps, np.ndarray):
        Rt = np.linalg.inv(Ks) @ Ps if Ks is not None else Ps
        Rs = Rt[:, 0:3, 0:3]
        ts = Rt[:, 0:3, 3]
    else:
        n_cams = Ps.shape[0]
        if Ks is None:
            Ks = torch.eye(3, device=Ps.device).expand((n_cams, 3, 3))

        Rt = torch.bmm(Ks.inverse(), Ps)
        Rs = Rt[:, 0:3, 0:3]
        ts = Rt[:, 0:3, 3]

    if inverse_direction_camera2global:
        Rs, ts = invert_euclidean_trafo(Rs, ts)

    return Rs, ts


def invert_euclidean_trafo(Rs, ts):
    """
    Given a batch of Euclidean transformations, Rs and ts, such that X -> Rs[i]*X + ts[i],
    return the corresponding Rs & ts for the inverse transformation, i.e. such that the above
    formula with the new Rs and ts now maps in the opposite direction.
    """
    assert len(Rs.shape) == 3
    assert Rs.shape[1] == 3
    assert Rs.shape[2] == 3
    assert len(ts.shape) == 2
    assert ts.shape[1] == 3
    if isinstance(Rs, np.ndarray):
        Rs_inv = np.transpose(Rs, [0,2,1]) # Rs_inv = Rs.T
        ts_inv = (-Rs_inv @ ts.reshape([-1, 3, 1])).squeeze() # ts_inv = -Rs.T @ ts = -Rs_inv @ ts
    else:
        Rs_inv = Rs.transpose(1, 2) # Rs_inv = Rs.T
        ts_inv = torch.bmm(-Rs_inv, ts.unsqueeze(-1)).squeeze() # ts_inv = -Rs.T @ ts = -Rs_inv @ ts
    return Rs_inv, ts_inv
