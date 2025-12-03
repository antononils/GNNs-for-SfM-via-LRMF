from datasets.SceneData import *
from torch_geometric.loader import DataLoader
from utils.dataset_utils import OutlierInjector, inject_outliers
import torch

def create_dataloader(scene_names, scene_type, max_points = None, batch_size=1, shuffle=True, outlier_threshold = None, device='cpu'):
    torch.manual_seed(42)
    M_gt_list = []
    Ns_list = []
    graphs = []
    for scene_name in scene_names:
        scene_data = create_scene_data(scene_name, scene_type)
        M_gt = scene_data.x.to(device)
        if max_points is not None and M_gt.shape[1] > max_points:
            idx = torch.randperm(M_gt.shape[1])[:max_points]
            M_gt = M_gt[:, idx,:]
        M_gt_list.append(M_gt)
        if outlier_threshold is not None:
            scene_data = inject_outliers(scene_data, outlier_threshold)
        Ns = scene_data.Ns.to(device)
        Ns_list.append(Ns)

        M = scene_data.x.to(device)
        if max_points is not None:
            if M.shape[1] > max_points:
                M = M[:, idx,:]
        graph = scene_data.matrix_to_graph(M)
        graphs.append(graph)

    dataloader = DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)

    return dataloader, Ns_list, M_gt_list