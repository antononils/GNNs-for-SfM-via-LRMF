from models.layers import FactormerLayer, _init_weights_xavier
from network_functions.ba_solver import admm_ba, ceres_ba
import torch
import torch.nn as nn
import torch.nn.functional as F

def project_to_rot(m):
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    return torch.matmul(u, vt)

def extract_view_outputs(x):
    # Get calibrated predictions
    m = x[:, :9].reshape(-1, 3, 3)
    RTs = project_to_rot(m)

    # Get translation
    minRTts = x[:, -3:]

    # Get camera matrix
    Ps = torch.cat((RTs, minRTts.unsqueeze(dim=-1)), dim=-1)
    return Ps

# InitModel
class InitModel(nn.Module):
    """
    InitModel wraps a pair of Factormer modules operating between two embedding spaces:
      - V (camera / P-like) space of dim dV
      - S (structure / X-like) space of dim dS

    The alternating application requires projecting targets into the Factormer's expected dimension,
    and mapping outputs back.
    """

    def __init__(self, dV, dS, n_factormers, solver_iters=5, device='cpu',
                 edge_dim=2, factormer_kwargs=None):
        super().__init__()

        # Parameters
        self.dV = dV
        self.dS = dS
        self.n_factormers = n_factormers
        self.solver_iters = solver_iters
        self.device = device

        # Input embedding/extraction layers
        self.embed_V = nn.Linear(12, dV)
        self.embed_S = nn.Linear(3, dS)
        self.extract_V = nn.Linear(dV, 12)
        self.extract_S = nn.Linear(dS, 3)

        # Projections between spaces
        self.dVtodS = nn.Linear(dV, dS)
        self.dStodV = nn.Linear(dS, dV)

        # Factormers: VS uses dV internal dimension
        fk_kwargs = {} if factormer_kwargs is None else factormer_kwargs
        self.factormer_VS = FactormerLayer(d=dV, edge_dim=edge_dim, **fk_kwargs, use_edge_update=False)
        self.factormer_SV = FactormerLayer(d=dS, edge_dim=edge_dim, **fk_kwargs, use_edge_update=False)

        # Weight initialization
        self.apply(_init_weights_xavier)

    def forward(self, V0, S0, edge_index, edge_attr, M, obs_matrix, solver_type = 'ceres', solver_iters_override = None):
        """
        V0: (nV, 12) raw V inputs
        S0: (nS, 3) raw S inputs
        edge_index: [2, E] indexing between V and S nodes (assumed V->S ordering)
        edge_attr: (E, edge_dim)
        Returns sequences from admm bundle adjust (keeps your admm call)
        """
        # initial embeddings
        V = self.embed_V(V0)
        S = self.embed_S(S0)

        # Prepare edge indexes for both directions
        edge_index_VS = edge_index
        edge_index_SV = torch.stack([edge_index[1], edge_index[0]], dim=0)

        # Alternate factormer blocks
        for _ in range(self.n_factormers):
            # Apply Factormer in dV internal space:
            S_proj = self.dStodV(S) 
            S_out_proj, edge_attr = self.factormer_VS(V, S_proj, edge_index_VS, edge_attr)
            # Map S back to dS
            S = self.dVtodS(S_out_proj)

            # Now apply Factormer in dS internal space:
            V_proj = self.dVtodS(V)
            V_out_proj, edge_attr = self.factormer_SV(S, V_proj, edge_index_SV, edge_attr)
            # Map V back to dV
            V = self.dStodV(V_out_proj)

        # Extract to numeric P (m,3,4) and X (n,4)
        P = self.extract_V(V)
        P_out = P.reshape(-1, 3, 4)
        X = self.extract_S(S)
        ones = torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)
        X = torch.cat([X, ones], dim=1)

        solver_iters = self.solver_iters if solver_iters_override is None else solver_iters_override

        # Optional: run ADMM bundle adjustment
        if solver_type == 'ceres':
            P_seq, X_seq = ceres_ba(P_out,X,M, obs_matrix, solver_iters, True)
        elif solver_type == 'admm':
            P_seq, X_seq = admm_ba(P_out,X, M, obs_matrix, 1, solver_iters, 1, 1, 1e-2, True)
        else:
            raise ValueError(f'Unknown solver type {solver_type}')
        return P_seq, X_seq