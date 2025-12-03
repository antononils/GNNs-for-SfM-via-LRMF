import torch
import torch.nn as nn
import torch.nn.functional as F

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