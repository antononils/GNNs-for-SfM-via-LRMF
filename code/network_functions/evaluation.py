import torch
from utils.dataset_utils import denormalize_M


def init_cam_V(n_views, device):
    # quaternion = (w,x,y,z) with w ~ 1, small xyz noise
    eps = 1e-3
    q = torch.randn(n_views, 4, device=device) * 1e-3
    q[:, 0] += 1.0    # bias real part toward 1.0 (identity)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    t = torch.randn(n_views, 3, device=device) * 0.1  # small translation scale tuned to your scene
    V0 = torch.cat([q, t], dim=-1)
    return V0

def compute_pixel_error(P_final, X_final, M, Ns, obs_matrix):
    # Project points and denormalize
    P_final = torch.linalg.inv(Ns) @ P_final
    z = torch.einsum('mij,nj->mni', P_final, X_final)
    z0, z1, z2 = z[..., 0], z[..., 1], z[..., 2].clamp(min=1e-1)
    pred = torch.stack([z0 / z2, z1 / z2], dim=-1)
    #denorm_pred = denormalize_M(pred,Ns, obs_matrix)
    denorm_pred = pred
    denorm_M = denormalize_M(M,Ns,obs_matrix)
    # diff: (m, n, 2)
    diff = (denorm_pred - denorm_M)  # shape (m, n, 2)
    # Compute per-point Euclidean distance
    dist = torch.sqrt((diff ** 2).sum(dim=2))  # shape (m, n)
    # Create mask where obs_matrix != 0
    mask = obs_matrix != 0

    # Select only observed entries
    valid_dists = dist[mask]

    # Compute mean pixel error
    px_error = valid_dists.mean()
    return px_error

def evaluate_model(dataloader,Ns_list,Ms_gt,solver_type,model_path,model,scene_type='Projective',device='cpu'):

    # Load model and optimizer states
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Lists to store results
    px_errors = []
    P_finals = []
    X_finals = []
    obs_matrices = []
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # Extract data
            m, n = data.m, data.n
            edge_index, edge_attr = data.edge_index.to(device), data.edge_attr.to(device)
            M = data.x.to(device)
            obs_matrix = data.obs_matrix.to(device)
            Ns = Ns_list[i]
            # Initialize random inputs

            if scene_type == 'Projective':
                V0, S0 = torch.empty(m, 12).uniform_(0,1).to(device), torch.empty(n, 3).uniform_(0,1).to(device)
            elif scene_type == 'Euclidean':
                V0, S0 = init_cam_V(m,device), torch.empty(n, 3).uniform_(0,1).to(device)
            else:
                raise ValueError(f"Unknown scene type: {scene_type}")

            # Forward pass
            P_seq, X_seq = model(V0, S0, edge_index, edge_attr, M, obs_matrix, solver_type, 0)
            P_final = P_seq[-1]
            X_final = X_seq[-1]

            P_finals.append(P_final)
            X_finals.append(X_final)
            obs_matrices.append(obs_matrix)

    return P_finals, X_finals, obs_matrices
