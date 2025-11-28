import torch  # DO NOT REMOVE
from utils import path_utils
import numpy as np
import os.path


def get_raw_data(scene):
    """
    # :param conf:
    :return:
    M - Points Matrix (2mxn)
    Ns - Normalization matrices (mx3x3)
    Ps_gt - Olsson's estimated camera matrices (mx3x4)
    NBs - Normzlize Bifocal Tensor (Normalized Fn) (3mx3m)
    triplets
    """
    # Init
    dataset_path_format = os.path.join(path_utils.path_to_datasets(), 'Projective', '{}.npz')

    # Get raw data
    dataset = np.load(dataset_path_format.format(scene))

    # Get bifocal tensors and 2D points
    M = dataset['M']
    Ps_gt = dataset['Ps_gt']
    Ns = dataset['Ns']
    N33 = Ns[:, 2, 2][:, None, None]
    Ns /= N33 # Divide by N33 to ensure last row [0, 0, 1] (although generally the case, a small deviation in scale has been observed for e.g. the PantheonParis scene)

    M = torch.from_numpy(M).float()
    Ps_gt = torch.from_numpy(Ps_gt).float()
    Ns = torch.from_numpy(Ns).float()

    return M, Ns, Ps_gt

