import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_sum


# Drop Path module for stochastic depth
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_prob
        return x * keep.div(1.0 - self.drop_prob)

# Weight initialization
def _init_weights_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# Factormer class
class FactormerLayer(nn.Module):
    """
    Multi-head attention graph module
    - d: embedding dim
    - edge_dim: dimension of provided edge_attr (if edge_attr is concatenated with pairwise node product)
    - use_edge_update: whether to produce updated edge_attr
    - drop_path: stochastic depth probability to apply to the residual branch
    """

    def __init__(self, d, num_heads=4, attn_dropout=0.0, dropout=0.0, edge_dim=None,
                 use_edge_update=True, drop_path=0.0, ff_mult=4, use_gated_ffn=True):
        super().__init__()
        assert d % num_heads == 0, "d must be divisible by num_heads"
        if edge_dim is None:
            raise ValueError("You must pass edge_dim=<edge_attr_dim> when constructing ImprovedFactormer.")

        # Parameters
        self.d = d
        self.h = num_heads
        self.dh = d // num_heads
        self.edge_dim = edge_dim
        self.edge_feat_dim = d + edge_dim
        self.use_edge_update = use_edge_update
        self.use_gated_ffn = use_gated_ffn

        # Node projections
        self.q = nn.Linear(d, d, bias=False)
        self.kN = nn.Linear(d, d, bias=False)
        self.vN = nn.Linear(d, d, bias=False)

        # Edge -> key/value and attention bias
        self.kE = nn.Linear(self.edge_feat_dim, d)
        self.vE = nn.Linear(self.edge_feat_dim, d)

        # Edge bias per head (small MLP)
        self.edge_bias_mlp = nn.Sequential(
            nn.Linear(self.edge_feat_dim, max(32, self.edge_feat_dim//2)),
            nn.ReLU(),
            nn.Linear(max(32, self.edge_feat_dim//2), self.h)
        )

        # Normalization for edge features
        self.edge_norm = nn.LayerNorm(self.edge_feat_dim)

        # Final projection
        self.out = nn.Linear(d, d)

        # Feed-forward network, gated or plain
        if use_gated_ffn:
            self.ffn_gate = nn.Linear(d, ff_mult * d)
            self.ffn_up = nn.Linear(d, ff_mult * d)
            self.ffn_down = nn.Linear(ff_mult * d, d)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d, ff_mult * d),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_mult * d, d),
                nn.Dropout(dropout)
            )

        # LayerNorms
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

        # Dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.msg_dropout = nn.Dropout(dropout)

        # Residual gating / layer scale
        self.res_scale = nn.Parameter(torch.ones(1) * 1e-1)
        self.ffn_scale = nn.Parameter(torch.ones(1) * 1e-1)

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Small MLP to update edge features
        if use_edge_update:
            self.edge_updater = nn.Sequential(
                nn.Linear(self.edge_feat_dim + d, max(self.edge_feat_dim, d)),
                nn.ReLU(),
                nn.Linear(max(self.edge_feat_dim, d), edge_dim)
            )

        # Per-head learned temperature
        self.inv_temp = nn.Parameter(torch.ones(self.h) * (1.0 / math.sqrt(self.dh)))

        # Initialize weights
        self.apply(_init_weights_xavier)

    def _reshape_heads(self, x):
        # x: (N, d) -> (N, h, dh)
        return x.view(x.size(0), self.h, self.dh)

    def forward(self, source, target, edge_index, edge_attr=None):
        """
        source: (N_src, d)
        target: (N_tgt, d)
        edge_index: [2, E] (src_idx, tgt_idx)
        edge_attr: (E, edge_dim)
        returns:
            y: (N_tgt, d)
            updated_edge_attr: (E, edge_dim) or same as input
        """

        # Unpack edge_index
        src_idx, tgt_idx = edge_index

        # Pre-norm
        target_norm = self.norm_q(target)
        source_norm = self.norm_kv(source)
        
        # Projections
        Q = self.q(target_norm)
        K_node = self.kN(source_norm)
        V_node = self.vN(source_norm)

        # Pairwise node product for implicit edge features
        pairwise = source[src_idx] * target[tgt_idx]

        # Edge features
        edge_feat = torch.cat([pairwise, edge_attr], dim=-1)
        edge_feat = self.edge_norm(edge_feat)

        # Edge-derived keys, values and bias
        K_edge = self.kE(edge_feat)
        V_edge = self.vE(edge_feat)
        att_bias = self.edge_bias_mlp(edge_feat)

        # Sum keys/values per-edge (node + edge)
        sum_keys = K_node[src_idx] + K_edge
        sum_vals = V_node[src_idx] + V_edge

        # Reshape into heads
        Qh = self._reshape_heads(Q[tgt_idx])
        Kh = self._reshape_heads(sum_keys)
        Vh = self._reshape_heads(sum_vals)

        # Dot product per head
        att_logits = (Qh * Kh).sum(-1)

        # Scale by learned per-head inverse temperature
        att_logits = att_logits * self.inv_temp.view(1, -1)

        # Add edge-derived per-head bias
        att_logits = att_logits + att_bias

        # Softmax over incoming edges grouped by target node
        att = scatter_softmax(att_logits, tgt_idx, dim=0)
        att = self.attn_dropout(att)

        # Message attention
        msg = att.unsqueeze(-1) * Vh
        msg = self.msg_dropout(msg)

        # Aggregate messages to target nodes
        out = scatter_sum(msg, tgt_idx, dim=0, dim_size=target.size(0))
        out = out.view(target.size(0), self.d)
        out = self.out(out)

        # Scaled residual
        gated = self.res_scale * out
        y = target + self.drop_path(gated)
        y = self.ln1(y)

        # Feed-forward (gated or plain)
        if self.use_gated_ffn:
            gate = self.ffn_gate(y)
            up = self.ffn_up(y)
            hidden = gate * F.gelu(up)
            y_ff = self.ffn_down(hidden)
        else:
            y_ff = self.ff(y)

        y = y + self.drop_path(self.ffn_scale * y_ff)
        y = self.ln2(y)

        # Update edge features with pooled message info
        if self.use_edge_update and edge_attr is not None:
            edge_update_input = torch.cat([edge_feat, sum_vals], dim=-1)
            updated_edge_attr = self.edge_updater(edge_update_input)
        else:
            updated_edge_attr = edge_attr

        return y, updated_edge_attr
