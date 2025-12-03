from torch_geometric.data import Data
import torch
from utils import dataset_utils
from datasets import Projective, Euclidean

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
        self.Ps_gt = Ps_gt
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


def create_scene_data(scene = None,scene_type='Projective'):
    if scene is None:
        scene = "Dino319"

    if scene_type=='Projective':
        M, Ns, Ps_gt = Projective.get_raw_data(scene)
    elif scene_type=='Euclidean':
        M, Ns, Ps_gt = Euclidean.get_raw_data(scene)
    else:
        raise ValueError(f"Unknown scene type: {scene_type}")
    
    scene_data = SceneData(
        M,
        Ns,
        Ps_gt,
        scene,
    )
    return scene_data
