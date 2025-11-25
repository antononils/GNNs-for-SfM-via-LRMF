import torch
from utils.dataset_utils import denormalize_M

# -------------------------------
# Training loop with warm-up and gradient clipping
# -------------------------------
def train_model(model, dataloader, optimizer, scheduler, loss_fn,
                epochs, Ns_list, device='cpu', warmup_epochs=5, max_grad_norm=1.0,
                admm_iters_schedule=(0,1,2), save_path='outputs/best_model.pth'):

    best_px_error = float('inf')  # initialize best validation metric
    model.train()
    for epoch in range(epochs):
        running = 0.0

        # Schedule ADMM iterations:
        if epoch < warmup_epochs:
            admm_iters = 0
        else:
            idx = min(epoch - warmup_epochs, len(admm_iters_schedule)-1)
            admm_iters = admm_iters_schedule[idx]

        # Training loop
        for k, data in enumerate(dataloader):
            # Extract data
            m, n = data.m, data.n
            edge_index, edge_attr = data.edge_index.to(device), data.edge_attr.to(device)
            num_nodes = data.num_nodes
            V0, S0 = torch.empty(m, 12).uniform_(0,1).to(device), torch.empty(n, 3).uniform_(0,1).to(device)
            M = data.x.to(device)
            obs_matrix = data.obs_matrix.to(device)
            
            # Forward pass
            print(admm_iters)
            P_seq, X_seq = model(V0, S0, edge_index, edge_attr, M, obs_matrix, admm_iters_override = admm_iters)

            # Compute loss
            loss = loss_fn(P_seq, X_seq, M, obs_matrix)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss)

            running += loss.item()
            P_final = P_seq[-1]
            X_final = X_seq[-1]

            # Project points
            z = torch.einsum('mij,nj->mni', P_final, X_final)
            z0, z1, z2 = z[..., 0], z[..., 1], z[..., 2].clamp(min=1e-1)
            pred = torch.stack([z0 / z2, z1 / z2], dim=-1)
            denorm_pred = denormalize_M(pred, Ns_list[k], obs_matrix)
            denorm_M = denormalize_M(M,Ns_list[k],obs_matrix)
            diff = (denorm_pred - denorm_M)

            # Compute per-point Euclidean distance
            dist = torch.sqrt((diff ** 2).sum(dim=2))
            # Create mask where obs_matrix != 0
            mask = obs_matrix != 0

            # Select only observed entries
            valid_dists = dist[mask]

            # Compute mean pixel error
            px_error = valid_dists.mean()
        avg = running / max(1, len(dataloader))
        print(f"Epoch {epoch+1}/{epochs}  loss={avg:.6f}  px_error={px_error}  (admm_iters={admm_iters})")

        # Save best model
        if px_error < best_px_error:
            best_px_error = px_error
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'px_error': px_error
            }, save_path)
            print(f"Best model saved with px_error={px_error:.6f} at epoch {epoch+1}")