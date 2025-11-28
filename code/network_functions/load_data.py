from datasets.SceneData import *
from torch_geometric.loader import DataLoader
from utils.dataset_utils import OutlierInjector, inject_outliers

def create_dataloader(scene_names, scene_type, batch_size=1, shuffle=True, outlier_threshold = None, device='cpu'):
    M_gt_list = []
    Ns_list = []
    graphs = []
    for scene_name in scene_names:
        scene_data = create_scene_data(scene_name, scene_type)
        M_gt_list.append(scene_data.x.to(device))
        if outlier_threshold is not None:
            scene_data = inject_outliers(scene_data, outlier_threshold)
        Ns = scene_data.Ns.to(device)
        Ns_list.append(Ns)
        graph = scene_data.matrix_to_graph(scene_data.x.to(device))
        graphs.append(graph)

    dataloader = DataLoader(graphs, batch_size=batch_size, shuffle=shuffle)

    return dataloader, Ns_list, M_gt_list