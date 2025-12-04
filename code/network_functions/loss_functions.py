import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.geo_utils import get_positive_projected_pts_mask

# Stable reconstruction loss
class ReconLossStable(nn.Module):
    def __init__(self, gamma=0.6, eps=1e-1, depth_penalty_w=1.0, huber_delta=1.0):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.depth_penalty_w = depth_penalty_w
        self.huber_delta = huber_delta
        self.smooth_l1 = nn.SmoothL1Loss(reduction='sum', beta=huber_delta)

    # Project 3D points X (n,4) into 2D using cameras P (m,3,4)
    def project(self, P, X):
        z = torch.einsum('mij,nj->mni', P, X)
        z0 = z[..., 0]
        z1 = z[..., 1]
        z2 = z[..., 2]
        z2_safe = z2.clamp(min=self.eps)
        u = z0 / z2_safe
        v = z1 / z2_safe
        proj = torch.stack([u, v], dim=-1)
        return proj, z2


    def forward(self, P_seq, X_seq, M, obs_matrix):
        # M: (m,n,2), obs_matrix: (m,n) boolean
        total_loss = 0.0
        total_visible = obs_matrix.float().sum().clamp(min=1.0)
        depth_penalty = 0.0

        for t, (P_t, X_t) in enumerate(zip(P_seq, X_seq)):
            proj, z2 = self.project(P_t, X_t)

            # Masked reprojection loss
            mask = obs_matrix.bool().unsqueeze(-1)
            sel = proj[mask.expand_as(proj)] - M[mask.expand_as(M)]
            if sel.numel() > 0:
                sel = sel.view(-1, 2)
                loss_reproj = self.smooth_l1(sel, torch.zeros_like(sel))
            else:
                loss_reproj = torch.tensor(0.0, device=proj.device)

            # Depth penalty: negative and very small depths
            z2_masked = z2 * obs_matrix.float()
            neg_depths = F.relu(-z2_masked)
            small_depths = F.relu(0.1 - z2_masked)
            depth_penalty += (neg_depths.sum() * 10.0 + small_depths.sum() * 1.0)

            total_loss = total_loss + (self.gamma ** t) * loss_reproj

        # Normalize by number of visible points
        total_loss = total_loss / total_visible
        total_loss = total_loss + self.depth_penalty_w * depth_penalty / total_visible

        return total_loss
    
class ESFMLoss(nn.Module):
    def __init__(self, infinity_pts_margin = 0.0001, hinge_loss_weight = 1):
        super().__init__()
        self.infinity_pts_margin = infinity_pts_margin
        self.hinge_loss_weight = hinge_loss_weight

    def forward(self, P_seq, X_seq, M, obs_matrix):
        # The predicted cameras "Ps_norm" and the GT 2D points "data.norm_M" are normalized with the N matrices given in the dataset.
        # Consequently, the reprojection error itself is calculated in normalized image space.
        # In the Euclidean setting, N=inv(K).
        Ps = P_seq[-1]  # Use the last predicted cameras
        pts3D = X_seq[-1]  # Use the last predicted 3D
        
        pts_2d = torch.einsum('mij,nj->mni', Ps, pts3D)

        positive_projected_pts_mask = get_positive_projected_pts_mask(pts_2d, self.infinity_pts_margin)
        if pts_2d.requires_grad:
            pts_2d.register_hook(lambda grad: torch.where(
                positive_projected_pts_mask[..., None].expand_as(grad),  # (m,n,1)->(m,n,3)
                F.normalize(grad, dim=-1)
                / max(1, torch.sum(obs_matrix & positive_projected_pts_mask).item()),
                grad
            ))

        # Calculate hinge Loss
        # NOTE: While at this point the "hinge loss" is just a linear loss applied on all depths irrespective of sign, in the end it will be used to replace the reprojection error for only the points with invalid depth, and hence effectively be a hinge loss after all.
        hinge_loss = (self.infinity_pts_margin - pts_2d[:, :, 2]) * self.hinge_loss_weight

        depth = torch.where(
        positive_projected_pts_mask,
        pts_2d[:, :, 2],
        torch.ones_like(pts_2d[:, :, 2], dtype=torch.float32)
    )  # (m,n)
        pts_2d_normed = pts_2d / depth[..., None]  # (m,n,3)

        reproj_err = (pts_2d_normed[:, :, 0:2] - M).norm(dim=-1)  # (m,n)

        #####################################
        # FINAL LOSS
        #####################################
        final = torch.where(
            positive_projected_pts_mask,
            reproj_err,
            hinge_loss
        )[obs_matrix]

        return final.mean()