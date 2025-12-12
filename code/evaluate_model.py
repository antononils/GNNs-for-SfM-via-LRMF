import torch, os, time, numpy as np, pandas as pd
from utils.dataset_utils import *
from utils.geo_utils import *
from utils.ba_functions import *
from utils.plot_utils import *
from datasets.SceneData import *
from models.init_model import *
from network_functions.load_data import create_dataloader
from network_functions.evaluation import evaluate_model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    scene_type = 'Euclidean'
    
    #scene_names = ["EcoleSuperiorDeGuerre", "GustavIIAdolf", "JonasAhlstromer", "KingsCollegeUniversityOfToronto", "ParkGateClermontFerrand"]

    scene_names = ["AlcatrazCourtyard", "AlcatrazWaterTower", "DrinkingFountainSomewhereInZurich",
                   "NijoCastleGate", "PortaSanDonatoBologna", "RoundChurchCambridge", "SmolnyCathedralStPetersburg",
                   "SomeCathedralInBarcelona", "SriVeeramakaliammanSingapore", "YuehHaiChingTempleSingapore"]

    #scene_names = ["AlcatrazWaterTower", "Dino319", "Dino4983", "DrinkingFountain", "Dome", "GustavVasa", "Nijo", 
    #               "SkansenKronan", "SomeCathedralInBarcelona", "SriVeeramakaliammanSingapore"]

    dataloader, Ns, Ms = create_dataloader(scene_names, scene_type=scene_type, max_points=None, batch_size=1, shuffle=False, outlier_threshold=None, device=device)
    model = InitModel(dV=1024, dS=64, n_factormers=2, scene_type=scene_type, solver_iters=0, device=device).to(device)
    Ps, Xs, Os = evaluate_model(dataloader, Ns, Ms, 'ceres', '../../pretrained_models/best_euc_model.pth', model, scene_type=scene_type, device=device)

    out_dir = "outputs_time"
    os.makedirs(out_dir, exist_ok=True)
    repro_stats = []

    for idx, (P, X, M, N, O) in enumerate(zip(Ps, Xs, Ms, Ns, Os)):
        scene_name = scene_names[idx]

        if scene_type == 'Projective':
            P = torch.linalg.inv(N) @ P
            M = denormalize_M(M, N, O)
            P, X, M, N, O = P.cpu().numpy(), X.cpu().numpy(), M.cpu().numpy(), N.cpu().numpy(), O.cpu().numpy()
            results = proj_ba(P, M, X, N)

        elif scene_type == 'Euclidean':
            M = denormalize_M(M, N, O)
            P, X, M, N, O = P.cpu().numpy(), X.cpu().numpy(), M.cpu().numpy(), N.cpu().numpy(), O.cpu().numpy()
            R, t = decompose_camera_matrix(P, inverse_direction_camera2global=True)
            start = time.time()
            results = euc_ba(M, R, t, np.linalg.inv(N), X, None, N)
            print(f'BA: {time.time()-start}')

        else:
            raise ValueError(f"Unknown scene type: {scene_type}")

        print(f"[{scene_name}] Results before: {results['repro_before']}")
        print(f"[{scene_name}] Results after:  {results['repro_after']}")

        repro_stats.append({
            "scene": scene_name,
            "repro_before": float(results.get("repro_before", np.nan)),
            "repro_after": float(results.get("repro_after", np.nan))
        })

        # Extract BA results
        Rs_ba = results.get("Rs", None)
        ts_ba = results.get("ts", None)
        Ps_ba = results.get("Ps", None)
        Xs_ba = results.get("Xs", None)

        # Save NPZ
        npz_path = os.path.join(out_dir, f"{scene_name}.npz")
        np.savez(npz_path, Rs=Rs_ba, ts=ts_ba, Ps=Ps_ba, Xs=Xs_ba)

        # Plot point cloud
        #X_ref = results["Xs"]
        #X_ref = torch.as_tensor(X_ref, dtype=torch.double)

        #lower, upper = X_ref.quantile(0.01), X_ref.quantile(0.99)
        #mask = ((X_ref >= lower) & (X_ref <= upper)).all(axis=1)
        #X_ref = X_ref[mask]

        #plot_path = os.path.join(out_dir, f"{scene_name}.html")
        #plotly_3d_points(X_ref, save_path=plot_path)
    
    df = pd.DataFrame(repro_stats)
    excel_path = os.path.join(out_dir, "reprojection_stats.xlsx")
    df.to_excel(excel_path, index=False)
