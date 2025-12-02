import torch
from utils.dataset_utils import denormalize_M

# -------------------------------
# Training loop with warm-up and gradient clipping
# -------------------------------
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn,
                epochs, train_Ns_list, train_M_gt_list, val_Ns_list, val_M_gt_list, device='cpu', warmup_epochs=5,
                max_grad_norm=1.0, solver_type = 'ceres', solver_iters_schedule=(0,1,2),
                save_path='outputs/best_model.pth'):

    best_px_error = float('inf')  # initialize best validation metric
    model.train()
    for epoch in range(epochs):
        train_running = 0.0

        # Schedule ADMM iterations:
        if epoch < warmup_epochs:
            solver_iters = 0
        else:
            idx = min(epoch - warmup_epochs, len(solver_iters_schedule)-1)
            solver_iters = solver_iters_schedule[idx]

        # Training loop
        for k, data in enumerate(train_dataloader):
            # Extract data
            m, n = data.m, data.n
            edge_index, edge_attr = data.edge_index.to(device), data.edge_attr.to(device)
            V0, S0 = torch.empty(m, 12).uniform_(0,1).to(device), torch.empty(n, 3).uniform_(0,1).to(device)
            M = data.x.to(device)
            obs_matrix = data.obs_matrix.to(device)
            
            # Forward pass
            P_seq, X_seq = model(V0, S0, edge_index, edge_attr, M, obs_matrix, solver_type = solver_type, solver_iters_override = solver_iters)
            M_gt = train_M_gt_list[k]
            # Compute loss
            loss = loss_fn(P_seq, X_seq, M_gt, obs_matrix)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            train_running += loss.item()
        

        # Validation step
        model.eval()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.no_grad():
            val_running = 0.0
            for k, data in enumerate(val_dataloader):
                # Extract data
                m, n = data.m, data.n
                edge_index, edge_attr = data.edge_index.to(device), data.edge_attr.to(device)
                V0, S0 = torch.empty(m, 12).uniform_(0,1).to(device), torch.empty(n, 3).uniform_(0,1).to(device)
                M = data.x.to(device)
                obs_matrix = data.obs_matrix.to(device)

                # Forward pass
                P_seq, X_seq = model(V0, S0, edge_index, edge_attr, M, obs_matrix, solver_type = solver_type, solver_iters_override = solver_iters)
                M_gt = val_M_gt_list[k]
                # Compute loss
                loss = loss_fn(P_seq, X_seq, M_gt, obs_matrix)
                val_running += loss.item()
                
                P_final = P_seq[-1]
                X_final = X_seq[-1]

                # Project points
                z = torch.einsum('mij,nj->mni', P_final, X_final)
                z0, z1, z2 = z[..., 0], z[..., 1], z[..., 2].clamp(min=1e-1)
                pred = torch.stack([z0 / z2, z1 / z2], dim=-1)
                denorm_pred = denormalize_M(pred, val_Ns_list[k], obs_matrix)
                denorm_M = denormalize_M(M_gt,val_Ns_list[k],obs_matrix)
                diff = (denorm_pred - denorm_M)

                # Compute per-point Euclidean distance
                dist = torch.sqrt((diff ** 2).sum(dim=2))
                # Create mask where obs_matrix != 0
                mask = obs_matrix != 0

                # Select only observed entries
                valid_dists = dist[mask]

                # Compute mean pixel error
                px_error = valid_dists.mean()
        
        avg_train_loss = train_running / max(1, len(train_dataloader))
        avg_val_loss = val_running / max(1, len(val_dataloader))

        if scheduler is not None:
                scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs}  train loss={avg_train_loss:.6f}  val loss={avg_val_loss:.6f}  px_error={px_error}  (admm_iters={solver_iters})")

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