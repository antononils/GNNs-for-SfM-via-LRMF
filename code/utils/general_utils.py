import torch
import numpy as np
import pandas as pd
import os
import torch.distributed as dist

def nonzero_safe(A):
    """
    Given a pytorch tensor / numpy array A (typically a boolean mask), return a tuple of indices for each of the dimensions, pointing to the non-zero elements of the input tensor / array.
    That is, the functionality is the same as np.nonzero(A) or torch.nonzero(A, as_tuple=True).
    The only difference is that if A is a torch CPU tensor, the torch.nonzero() function is avoided, as the CPU-implementation appears to contain bugs related to memory management.
    """
    if isinstance(A, np.ndarray):
        return np.nonzero(A)
    else:
        if A.is_cuda:
            return torch.nonzero(A, as_tuple=True)
        else:
            return tuple(torch.from_numpy(x) for x in np.nonzero(A.detach().numpy()))
        
def mark_occurences_of_tensor_in_other(A, B):
    """
    Given 1D-tensors A & B, return a boolean tensor of the same shape as A, where each element indicates whether the corresponding element in A also exists anywhere in B.
    """
    assert A.ndim == 1
    assert B.ndim == 1

    A_df = pd.DataFrame({'idx': A.cpu()})
    A_df['val'] = False
    assert A_df.shape[0] == A.shape[0]
    B_df = pd.DataFrame({'idx': B.cpu()})
    B_df['val'] = True
    tmp = pd.concat([A_df, B_df])
    tmp = tmp.drop_duplicates(subset=['idx'], keep='last').set_index('idx')
    A_notin_B = tmp.loc[A_df['idx'], 'val'].values
    assert A_notin_B.shape[0] == A.shape[0]
    A_notin_B = torch.from_numpy(A_notin_B).to(A.device)
    assert A_notin_B.shape == A.shape

    # # Simple (memory-inefficient) implementation:
    # result_trivial = torch.any(A[:, None] == B[None, :], dim=1)
    # assert result_trivial.shape == A.shape
    # assert torch.all(A_notin_B == result_trivial)
    # # A_notin_B = result_trivial

    return A_notin_B

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()