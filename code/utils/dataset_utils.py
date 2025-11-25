import torch
from utils.constants import *
import numpy as np

def reshape_M_to_3D_array(M):
    """
    Reshape a (2*m, n) measurement matrix to a (m, n, 2) 3D array.
    """
    assert len(M.shape) == 2
    assert M.shape[0] % 2 == 0
    n_pts = M.shape[-1]
    M = M.reshape((-1, 2, n_pts)) # (m, 2, n)
    if type(M) is torch.Tensor:
        M = M.transpose(1, 2) # (m, n, 2)
    else:
        M = M.swapaxes(1, 2) # (m, n, 2)
    return M


def get_M_valid_points(M):
    n_pts = M.shape[-1]

    if len(M.shape) == 2:
        # (2*m, n) measurement matrix -> (m, n, 2) 3D array:
        M = reshape_M_to_3D_array(M)
    assert len(M.shape) == 3
    assert M.shape[2] == 2

    if type(M) is torch.Tensor:
        # (m, n, 2) measurement array -> (m, n) validity mask, determined by which points are == (0, 0)
        M_valid_pts = torch.abs(M).sum(dim=2) != 0
        # Due to that we sometimes subsample the set of views, update the (m, n) validity mask, with the further requirement that points are only valid if they are visible in >= 2 views.
        # As a consequence, there may be views without enough visible points to estimate camera motion.
        # While we do not annotate projections of invalid views (as we are projections of invalid points), at the moment, we seem to simple discard subsets of scenes with this issue during training, so it is not an issue in practice.
        # If we were to subsample the set of points as well, we should probably pay even more attention to this, as it may happen more frequently.
        if M_valid_pts.is_cuda:
            M_valid_pts[:, M_valid_pts.sum(dim=0) < MIN_N_VIEWS_PER_POINT] = False
        else:
            # NOTE: Workaround for bug in nonzero_out_cpu(), internally called by pytorch during advanced indexing operation.
            idx = np.nonzero((M_valid_pts.sum(dim=0) < MIN_N_VIEWS_PER_POINT).numpy())[0]
            assert len(idx.shape) == 1, 'Expected 1D-array, but encountered idx.shape == {}'.format(idx.shape)
            M_valid_pts[:, torch.from_numpy(idx)] = False
    else:
        M_valid_pts = np.abs(M).sum(axis=2) != 0
        M_valid_pts[:, M_valid_pts.sum(axis=0) < MIN_N_VIEWS_PER_POINT] = False

    return M_valid_pts

def normalize_M(M, Ns, valid_points=None):
    if valid_points is None:
        valid_points = get_M_valid_points(M)
    norm_M = M.clone()
    n_images = norm_M.shape[0]//2
    norm_M = norm_M.reshape([n_images, 2, -1]) # [m,2,n]
    norm_M = torch.cat((norm_M, torch.ones(n_images, 1, norm_M.shape[-1], device=M.device)), dim=1)  # [m,3,n]

    norm_M = (Ns @ norm_M).permute(0, 2, 1)[:,:,:2]  # [m,3,3]@[m,3,n] -> [m,3,n]->[m,n,3]
    if norm_M.is_cuda:
        norm_M[~valid_points, :] = 0
    else:
        invalid_points_idx = np.nonzero((~valid_points).detach().numpy())
        norm_M[invalid_points_idx[0], invalid_points_idx[1], :] = 0
    return norm_M

def denormalize_M(norm_M, Ns, valid_points=None):
    """
    Reverse of normalize_M:
    norm_M: (m, n, 2)
    Ns: (m, 3, 3) normalization matrices
    returns: original M with shape (2m, n)
    """
    device = norm_M.device
    m, n, _ = norm_M.shape

    if valid_points is None:
        valid_points = torch.ones((m, n), dtype=torch.bool, device=device)

    # Put norm_M back into (m, 2, n)
    M = norm_M.permute(0, 2, 1)  # (m,n,2) -> (m,2,n)

    # Reintroduce homogeneous 1s → (m, 3, n)
    ones = torch.ones((m, 1, n), device=device)
    M_h = torch.cat([M, ones], dim=1)

    # Invert normalization matrices
    Ns_inv = torch.inverse(Ns)

    # Apply inverse transform
    M_orig_h = (Ns_inv @ M_h)  # (m,3,3)@(m,3,n) -> (m,3,n)

    # Drop homogeneous coordinate → (m,2,n)
    M_orig = M_orig_h[:, :2, :]

    # Zero-out invalid points (matching normalize_M behavior)
    if M_orig.is_cuda:
        M_orig[~valid_points] = 0
    else:
        invalid_idx = (~valid_points).nonzero(as_tuple=True)
        M_orig[invalid_idx[0], :, invalid_idx[1]] = 0

    # Reshape back to original (2m, n)
    return M_orig.permute(0, 2, 1)