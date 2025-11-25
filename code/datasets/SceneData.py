from torch_geometric.data import Data
import torch
from utils import dataset_utils
from datasets import Projective

class SceneData:
    def __init__(
        self,
        M,
        Ns,
        Ps_gt,
        scene_name
    ):

        # Determine the device
        self.device = M.device

        # Set attribute
        self.scene_name = scene_name
        self.y = Ps_gt
        self._M = M
        self.Ns = Ns

        # M to sparse matrix
        self.x = dataset_utils.normalize_M(self.M, self.Ns)

        # Get valid points
        self.valid_pts = dataset_utils.get_M_valid_points(self.M)

        # Define the graph connectivity for the aggregations from projection features to view features and scenepoint features, respectively.
        self.graph_wrappers = self.matrix_to_graph(
            self.x,
        )

    @property
    def M(self):
        if not self._M.device == self.device:
            self._M = self._M.to(self.device)
        return self._M

    def matrix_to_graph(self, x):
        m, n, _ = x.shape
        obs_matrix = (x.abs().sum(-1) > 0)

        source_idx, target_idx = obs_matrix.nonzero(as_tuple=True)
        edge_index = torch.stack([source_idx, target_idx], dim=0)
        edge_attr = x[source_idx, target_idx]  # (num_edges, 2)

        data = Data(edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=m+n,
                    x=x,
                    m=m,
                    n=n,
                    obs_matrix=obs_matrix)
        return data


def create_scene_data(
    conf,
    scene = None,
):

    # Optionally override some configuration options:
    scene = scene if scene is not None else conf.get_string('dataset.scene')

    M, Ns, Ps_gt = Projective.get_raw_data(scene)

    scene_data = SceneData(
        M,
        Ns,
        Ps_gt,
        scene,
    )
    return scene_data


def create_scene_data_from_list(scene_names_list, conf):
    data_list = []
    for scene_name in scene_names_list:
        data = create_scene_data(conf, scene=scene_name)
        data_list.append(data)

    return data_list
