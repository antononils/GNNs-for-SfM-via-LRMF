from datasets.SceneData import *
from models.init_model import *
from datasets.Projective import *
from network_functions.loss_functions import *
from network_functions.train import *
from torch_geometric.loader import DataLoader

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Hyperparameters you can tweak
    epochs = 50
    warmup_epochs = 50          # train GNN alone for these many epochs
    admm_iters_schedule = [5, 5, 5, 5, 5, 5]  # after warmup, ramp to these many ALS iters (index 0 = first epoch after warmup)
    model_admm_default = 5       # default internal value (unused because we use override scheduling)
    lr = 3e-5                   # small lr stabilizes training
    max_grad_norm = 0.5

    # dataset path (same as before)

    scene = create_scene_data(None, "House")
    Ns = scene.Ns.to(device)
    X = scene.x.to(device)
    no_batches = X.shape[0] // 128 + 1
    no_batches = 1
    graphs = []

    batch = X[:,:,:]
    graph = scene.matrix_to_graph(batch)
    graphs.append(graph)


    Ns_list = [Ns]*no_batches

    dataloader = DataLoader(graphs, batch_size=1, shuffle=True)
    model = InitModel(dV=1024, dS=64, n_factormers=2, admm_iters=model_admm_default, device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    loss_fn = ReconLossStable(gamma=0.8, eps=1e-1, depth_penalty_w=1.0, huber_delta=0.5)

    train_model(model, dataloader, optimizer, scheduler, loss_fn,
                epochs,Ns_list, device=device, warmup_epochs=warmup_epochs,
                max_grad_norm=max_grad_norm, admm_iters_schedule=admm_iters_schedule)