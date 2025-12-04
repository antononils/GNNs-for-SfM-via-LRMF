import torch
from network_functions.evaluation import evaluate_model
from datasets.SceneData import *
from models.init_model import *
from utils.plot_utils import *
from utils.dataset_utils import *
from utils.ba_functions import *
from utils.geo_utils import *
from network_functions.load_data import create_dataloader


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    scene_names = ["AlcatrazCourtyard", "AlcatrazWaterTower", "DrinkingFountainSomewhereInZurich",
                   "NijoCastleGate", "PortaSanDonatoBologna", "RoundChurchCambridge", "SmolnyCathedralStPetersburg",
                   "SomeCathedralInBarcelona", "SriVeeramakaliammanSingapore", "YuehHaiChingTempleSingapore"]
    scene_type = 'Euclidean'

    dataloader, Ns, Ms = create_dataloader(scene_names,scene_type=scene_type,max_points=None, batch_size=1, shuffle=False, outlier_threshold=None, device=device)

    model = InitModel(dV=1024, dS=64, n_factormers=2, scene_type=scene_type, solver_iters=0, device=device).to(device)
    px_errors, Ps, Xs, Os = evaluate_model(dataloader, Ns, Ms, 'ceres', '../../pretrained_models/euc_model.pth', model, scene_type=scene_type, device=device)

    i = 0
    for P, X, M, N, O in zip(Ps, Xs, Ms, Ns, Os):
        if scene_type == 'Projective':
            P = torch.linalg.inv(N) @ P
            M = denormalize_M(M, N, O)
            P, X, M, N, O = P.cpu().numpy(), X.cpu().numpy(), M.cpu().numpy(), N.cpu().numpy(), O.cpu().numpy()
            results = proj_ba(P, M, X, N)
        elif scene_type == 'Euclidean':
            M = denormalize_M(M, N, O)
            P, X, M, N, O = P.cpu().numpy(), X.cpu().numpy(), M.cpu().numpy(), N.cpu().numpy(), O.cpu().numpy()
            R, t = decompose_camera_matrix(P, inverse_direction_camera2global=True)
            results = euc_ba(M, R, t, np.linalg.inv(N), X, None, N)
        else:
            raise ValueError(f"Unknown scene type: {scene_type}")

        print(f"Results before: {results['repro_before']}")
        print(f"Results after: {results['repro_after']}")

        P = results['Ps']
        X = results['Xs']
        P, X, M, N, O = torch.as_tensor(P, dtype=torch.double), torch.as_tensor(X, dtype=torch.double), torch.as_tensor(M, dtype=torch.double), torch.as_tensor(N, dtype=torch.double), torch.as_tensor(O, dtype=torch.bool)

        lower, upper = X.quantile(0.01), X.quantile(0.99)
        mask = ((X >= lower) & (X <= upper)).all(axis=1)
        X = X[mask]

        save_path = f'outputs/point_cloud_{i}.html'
        plotly_3d_points(X, save_path=save_path)
        i += 1
