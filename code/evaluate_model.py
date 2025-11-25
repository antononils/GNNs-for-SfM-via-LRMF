import torch
from network_functions.evaluation import evaluate_model, compute_pixel_error
from datasets.SceneData import *
from models.init_model import *
from torch_geometric.loader import DataLoader
from utils.plot_utils import *
from network_functions.ba_solver import ceres_ba, admm_ba
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    scene = create_scene_data(None, "House")
    Ns = scene.Ns.to(device)
    graph = scene.matrix_to_graph(scene.x)


    model_admm_default = 5
    graph_M = [graph]
    dataloader = DataLoader(graph_M, batch_size=1, shuffle=True)
    model = InitModel(dV=1024, dS=64, n_factormers=2, admm_iters=model_admm_default, device=device).to(device)
    px_errors, P_finals, X_finals, Ms, obs_matrices= evaluate_model(dataloader,Ns, 'outputs/best_model.pth', model, device=device)
    print("Pixel Errors:", px_errors)



    for i, X_final in enumerate(X_finals):
        P_final, X_final = ceres_ba(P_finals[i], X_finals[i], Ms[i], obs_matrices[i], max_iters=100, lm_lambda_init=1e-1, lm_lambda_factor=1)
        px_error = compute_pixel_error(P_final, X_final, Ms[i], Ns, obs_matrices[i])
        print(px_error)
        save_path = f'outputs/point_cloud_{i}.html'
        plotly_3d_points(X_final, save_path=save_path)