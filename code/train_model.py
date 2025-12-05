from datasets.SceneData import *
from models.init_model import *
from datasets.Projective import *
from network_functions.loss_functions import *
from network_functions.train import *
from network_functions.load_data import create_dataloader
from utils.general_utils import *

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Hyperparameters you can tweak
    epochs = 1000
    warmup_epochs = 1000          # train GNN alone for these many epochs
    solver_iters_schedule = [1,2,3]  # after warmup, ramp to these many ALS iters (index 0 = first epoch after warmup)
    model_solver_default = 3       # default internal value (unused because we use override scheduling)
    lr = 3e-5                   # small lr stabilizes training
    max_grad_norm = 1.0

    # dataset path (same as before)
    training_scenes = ["EcoleSuperiorDeGuerre", "DoorLund", "ParkGateClermontFerrand", "ThianHookKengTempleSingapore", "StatueOfLiberty", "KingsCollegeUniversityOfToronto", "SriThendayuthapaniSingapore", "SkansenKronanGothenburg", "BuddahToothRelicTempleSingapore", "Eglisedudome", "FortChanningGateSingapore", "GustavVasa"]
    validation_scenes = ["GoldenStatueSomewhereInHongKong", "EastIndiamanGoteborg", "PantheonParis"]
    scene_type = 'Euclidean'
    train_dataloader, train_Ns_list, train_M_gt_list = create_dataloader(training_scenes, scene_type, max_points=None,batch_size=1, shuffle=False, outlier_threshold=None, device=device)
    val_dataloader, val_Ns_list, val_M_gt_list = create_dataloader(validation_scenes, scene_type, max_points=None, batch_size=1, shuffle=False, outlier_threshold=None, device=device)

    model = InitModel(dV=1024, dS=64, n_factormers=2, scene_type=scene_type, solver_iters=model_solver_default, device=device).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    loss_fn = ReconLossStable(gamma=0.8, eps=1e-1, depth_penalty_w=1.0, huber_delta=0.5)
    loss_ESFM = ESFMLoss(0.1)

    train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn,
                epochs,train_Ns_list, train_M_gt_list, val_Ns_list, val_M_gt_list, scene_type=scene_type, device=device, warmup_epochs=warmup_epochs,
                max_grad_norm=max_grad_norm, solver_type = 'ceres', solver_iters_schedule=solver_iters_schedule)
    
