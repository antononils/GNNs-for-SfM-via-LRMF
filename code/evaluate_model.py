import torch
from network_functions.evaluation import evaluate_model, compute_pixel_error
from datasets.SceneData import *
from models.init_model import *
from torch_geometric.loader import DataLoader
from utils.plot_utils import *
from network_functions.ba_solver import ceres_ba, admm_ba
from network_functions.load_data import create_dataloader
from utils.dataset_utils import denormalize_M
from datasets.Euclidean import get_raw_data

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model_solver_default = 5       # default internal value (unused because we use override scheduling)
    scene_names = ["DrinkingFountainSomewhereInZurich"]
    dataloader, Ns_list, Ms_gt = create_dataloader(scene_names,scene_type='Euclidean',max_points=None, batch_size=1, shuffle=False, outlier_threshold=None, device=device)
    model = InitModel(dV=1024, dS=64, n_factormers=2, solver_iters=model_solver_default, device=device).to(device)
    px_errors, P_finals, X_finals, obs_matrices= evaluate_model(dataloader,Ns_list,Ms_gt,'ceres', 'outputs/best_model.pth', model, device=device)
    print("Pixel Errors:", px_errors)



    for i, X_final in enumerate(X_finals):
        #P_seq, X_seq = ceres_ba(P_finals[i], X_finals[i], Ms_gt[i], obs_matrices[i], max_iters=20, lm_lambda_init=1e-3, lm_lambda_factor=10)
        P_final = P_finals[i]
        X_final = X_finals[i]
        px_error = compute_pixel_error(P_final, X_final, Ms_gt[i], Ns_list[i], obs_matrices[i])
        lower = X_final.quantile(0.01)
        upper = X_final.quantile(0.99)

        mask = ((X_final >= lower) & (X_final <= upper)).all(axis=1)
        X_final = X_final[mask]
        
        save_path = f'outputs/point_cloud_{i}.html'
        plotly_3d_points(X_final, save_path=save_path)
        