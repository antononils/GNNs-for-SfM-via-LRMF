import torch


def admm_ba(P_init, X_init, M, obs_matrix, 
            rho=1.0, 
            num_iters=20, 
            gn_iters=1, 
            ZP_iters=1, 
            lambda_lm=1e-2, 
            verbose=True
):
    """ADMM-style bundle adjustment."""
    device, dtype = P_init.device, P_init.dtype
    m, n = P_init.shape[0], X_init.shape[0]

    # Project 3D point to 2D with safe depth
    def project_perspective(z, eps=1e-8):
        x, y, t = z
        t_safe = torch.where(torch.abs(t) < eps, eps * torch.sign(t) + (t == 0).to(dtype) * eps, t)
        return torch.stack([x/t_safe, y/t_safe])
    
    # Convert mask to boolean
    obs_matrix = obs_matrix.bool()

    # Initialize variables from inputs
    P, X = P_init, X_init

    # Extract indices of valid observations
    obs_pairs = obs_matrix.nonzero(as_tuple=False)   # (K,2)

    # Initialize auxiliary projected points
    Z = torch.zeros((m, n, 3), device=device)
    # Initialize scaled dual variables
    U = torch.zeros((m, n, 3), device=device)

    # Build per-camera lists of observed points
    cams_obs = [obs_matrix[i].nonzero(as_tuple=True)[0] for i in range(m)]
    # Build per-point lists of observing cameras
    pts_obs  = [obs_matrix[:, j].nonzero(as_tuple=True)[0] for j in range(n)]

    # Compute ADMM augmented objective for monitoring
    def objective(P, X, Z, U):
        val = torch.zeros((), device=device)
        for idx in obs_pairs:
            i, j = int(idx[0]), int(idx[1])
            Mij = M[i, j]
            proj = project_perspective(Z[i, j])
            val = val + torch.sum((Mij - proj) ** 2)
            val = val + 0.5 * rho * torch.sum((Z[i, j] - P[i] @ X[j] + U[i, j]) ** 2)
        return val
    
    # Track objective values over iterations
    history = []
    # Store camera iterates
    P_seq = [P_init]
    # Store point iterates
    X_seq = [X_init]

    # Main ADMM loop
    for it in range(num_iters):
        
        # Optionally alternate Z and P updates multiple times
        for _ in range(ZP_iters):
          
            # Update Z for each observed pair
            for idx in obs_pairs:
                i, j = int(idx[0]), int(idx[1])
                Mij = M[i, j]

                # Current consensus target for Z
                Z_hat = P[i] @ X[j] - U[i, j]
                # Initialize Z with its target
                z = Z_hat.clone()

                # Run a few GN/LM steps on the local Z subproblem
                for _ in range(gn_iters):
                    # Project current z
                    proj = project_perspective(z)
                    # Reprojection residual
                    r = Mij - proj

                    # Split z into components
                    x, y, t = z
                    eps = 1e-8
                    # Stabilize depth for derivatives
                    t_safe = torch.where(torch.abs(t) < eps, eps * torch.sign(t) + (t == 0).to(t.dtype) * eps, t).to(device)

                    # Precompute normalized coordinates
                    f1 = x / t_safe
                    f2 = y / t_safe
                    
                    # Jacobian of projection wrt z
                    J_pi = torch.stack([
                            torch.stack([1.0 / t_safe, torch.tensor(0.0, device=device), -f1 / t_safe]),
                            torch.stack([torch.tensor(0.0, device=device), 1.0 / t_safe, -f2 / t_safe]),
                    ], dim=0)

                    # Jacobian of residual wrt z
                    J_res = -J_pi
                    # Local GN Hessian approximation
                    JTJ = J_res.T @ J_res
                    # Local GN gradient
                    grad_lsq = J_res.T @ r

                    # Local LM system matrix
                    H = JTJ + (rho + lambda_lm) * torch.eye(3).to(device)
                    # Local right-hand side
                    rhs = rho * (Z_hat - z) - grad_lsq

                    # Solve for Z step
                    try:
                        dz = torch.linalg.solve(H, rhs)
                    except RuntimeError:
                        dz = torch.linalg.lstsq(H, rhs.unsqueeze(-1)).solution.squeeze(-1)
                    
                    # Apply Z step
                    z = z + dz
                
                # Write back updated Z
                Z[i, j] = z
          
            # Update P camera-by-camera
            P_new = P.clone()
            for i in range(m):
                # Get points seen by camera i
                js = cams_obs[i]
                # Skip cameras with no observations
                if js.numel() == 0:
                    continue

                # Stack points observed by camera i
                X_stack = X[js]
                # Precompute normal matrix for this camera
                H = X_stack.T @ X_stack

                # Solve row-wise least squares for P_i
                for r in range(3):
                    # Target for row r from Z and U
                    t_vec = Z[i, js, r] + U[i, js, r]
                    # Compute RHS for row r
                    b = X_stack.T @ t_vec

                    # Solve for row r weights
                    try:
                        w = torch.linalg.solve(H, b)
                    except RuntimeError:
                        w = torch.linalg.lstsq(H, b.unsqueeze(-1)).solution.squeeze(-1)
                    
                    # Update row r of camera matrix
                    P_new[i, r, :] = w
            
            # Commit updated cameras
            P = P_new
        
        # Update X point-by-point
        X_new = X.clone()
        for j in range(n):
            # Get cameras that see point j
            is_ = pts_obs[j]
            # Skip points with no observations
            if is_.numel() == 0:
                continue

            # Collect stacked linear systems for this point
            H_rows, c_rows = [], []
            for i in is_:
                Pi = P[i]
                # Split camera into A and d so Pi*[xyz;1] = A*xyz + d
                Ai = Pi[:, :3]
                di = Pi[:, 3]
                H_rows.append(Ai)
                c_rows.append((Z[i, j] + U[i, j]) - di)
            
            # Build full least squares matrix
            H_j = torch.cat(H_rows, dim=0)
            # Build full RHS vector
            c_j = torch.cat(c_rows, dim=0)

            # Form normal equations for xyz
            HtH = H_j.T @ H_j
            Htc = H_j.T @ c_j

            # Solve for xyz
            try:
                xyz = torch.linalg.solve(HtH, Htc)
            except RuntimeError:
                xyz = torch.linalg.lstsq(HtH, Htc.unsqueeze(-1)).solution.squeeze(-1)
            
            # Update Euclidean part of X_j
            X_new[j, :3] = xyz

            # Reset homogeneous coordinate to 1
            X_new[j, 3] = 1.0
        
        # Commit updated points
        X = X_new

        # Save iterates for output
        P_seq.append(P)
        X_seq.append(X)
        
        # Update dual variables for observed pairs
        for idx in obs_pairs:
            i, j = int(idx[0]), int(idx[1])
            U[i, j] = U[i, j] + Z[i, j] - (P[i] @ X[j])
        
        # Evaluate objective for logging
        f = objective(P, X, Z, U)
        history.append(f.detach().item())

        # Print progress for this iteration
        if verbose:
            print(f"Iter {it:03d}, f_rho={history[-1]:.6e}")
    
    return P_seq, X_seq


def ceres_ba(P_init, X_init, M, obs_matrix, 
             max_iters=30, 
             lm_lambda_init=1e-3, 
             lm_lambda_factor=10.0, 
             max_lm_tries=5, 
             huber_delta=2.0, 
             min_depth=1e-3, 
             verbose=True
):
    """Ceres-style BA with robust loss, LM damping, and Schur complement."""
    device, dtype = P_init.device, P_init.dtype
    m, n = P_init.shape[0], X_init.shape[0]

    # Flatten cameras and points into one vector
    def pack_params(P, X):
        return torch.cat([P.reshape(-1), X.reshape(-1)], dim=0)

    # Unflatten one vector back into cameras and points
    def unpack_params(theta):
        P_vec = theta[:m*12]
        X_vec = theta[m*12:]
        return P_vec.view(m,3,4), X_vec.view(n,4)

    # Normalize each camera matrix by its Frobenius norm
    def normalize_cameras(P):
        norms = torch.linalg.norm(P.reshape(m,-1), dim=1, keepdim=True).clamp_min(1e-12)
        return P / norms.view(m,1,1)

    # Project 3D homogeneous point to 2D with safe depth
    def project_perspective(z, eps=1e-8):
        x, y, t = z
        t_safe = torch.where(torch.abs(t) < eps, eps * torch.sign(t) + (t == 0).to(dtype) * eps, t)
        return torch.stack([x/t_safe, y/t_safe])

    # Compute sqrt Huber weight from squared residual norm
    def huber_sqrt_weight(r2, delta=1.0, eps=1e-12):
        r = torch.sqrt(r2 + eps)
        w = torch.where(r <= delta, torch.ones_like(r), delta / r)
        return torch.sqrt(w)

    # List of observed (camera, point) pairs
    obs = obs_matrix.nonzero(as_tuple=False)  # (K,2)
    K = obs.shape[0]

    # Initial parameter vector
    theta = pack_params(normalize_cameras(P_init.clone()), X_init.clone())
    lm_lambda = lm_lambda_init

    # Evaluate robust cost for acceptance checks
    def compute_cost(P, X):
        cost = torch.zeros((), device=device, dtype=dtype)
        for k in range(K):
            i, j = int(obs[k,0]), int(obs[k,1])
            z = P[i] @ X[j]
            if z[2] < min_depth:
                cost = cost + (min_depth - z[2])**2 * 1e3
                continue
            r = M[i,j] - project_perspective(z)
            r2 = (r*r).sum()
            w = huber_sqrt_weight(r2, huber_delta)**2
            cost = cost + w * r2
        return cost

    # Main optimization loop
    for it in range(max_iters):

        # Unpack current parameters
        P, X = unpack_params(theta)
        # Re-normalize cameras to keep scale stable
        P = normalize_cameras(P)
        # Re-pack after normalization
        theta = pack_params(P, X)

        # Allocate per-camera Hessian blocks
        B = [torch.zeros((12,12), device=device, dtype=dtype) for _ in range(m)]
        # Allocate per-camera gradient blocks
        gc = [torch.zeros((12,), device=device, dtype=dtype) for _ in range(m)]
        # Allocate per-point Hessian blocks
        C = [torch.zeros((4,4), device=device, dtype=dtype) for _ in range(n)]
        # Allocate per-point gradient blocks
        gp = [torch.zeros((4,), device=device, dtype=dtype) for _ in range(n)]
        # Allocate sparse cross blocks for observed pairs
        E = {}  # key (i,j) -> (12,4)

        # Reset iteration cost accumulator
        cost = torch.zeros((), device=device, dtype=dtype)

        # Loop over observations to build normal equations
        for k in range(K):
            # Read camera/point indices
            i, j = int(obs[k,0]), int(obs[k,1])
            # Grab current camera, point, and measurement
            Pi, Xj, Mij = P[i], X[j], M[i,j]

            # Project point into camera coordinates
            z = Pi @ Xj
            # Split projected coordinates
            x, y, t = z

            # Handle invalid or tiny depth with penalty
            if t < min_depth:
                r_depth = (min_depth - t)
                cost = cost + r_depth**2 * 1e3

                # Derivative of depth w.r.t camera parameters
                dtdP = torch.zeros((1,12), device=device, dtype=dtype)
                dtdP[0, 8:12] = Xj
                # Derivative of depth w.r.t point parameters
                dtdX = Pi[2:3,:4].clone()
                dtdX[:,3] = 0.0

                # Use depth-only Jacobians
                JP = dtdP
                JX = dtdX

                # Accumulate depth penalty into camera block
                B[i] += (JP.T @ JP) * 1e3
                # Accumulate depth penalty into point block
                C[j] += (JX.T @ JX) * 1e3
                # Accumulate depth penalty into cross block
                E[(i,j)] = E.get((i,j), torch.zeros((12,4),device=device,dtype=dtype)) + (JP.T @ JX) * 1e3

                # Accumulate depth penalty gradient for camera
                gc[i] += (JP.T.squeeze(1) * (-r_depth)) * 1e3
                # Accumulate depth penalty gradient for point
                gp[j] += (JX.T.squeeze(1) * (-r_depth)) * 1e3
                continue

            # Project to image plane
            proj = project_perspective(z)
            # Compute reprojection residual
            r = Mij - proj
            # Compute squared residual norm
            r2 = (r*r).sum()

            # Compute robust downweighting
            sqrt_w = huber_sqrt_weight(r2, huber_delta)
            # Apply robust weight to residual
            r_w = sqrt_w * r
            # Accumulate robust cost
            cost = cost + (r_w*r_w).sum()

            # Build safe depth for Jacobian
            eps = 1e-8
            t_safe = torch.where(torch.abs(t) < eps, eps * torch.sign(t) + (t == 0)*eps, t)
            # Reuse projected ratios
            f1, f2 = x/t_safe, y/t_safe

            # Jacobian of projection wrt z
            J_pi = z.new_tensor([
                [1.0/t_safe, 0.0,        -f1/t_safe],
                [0.0,        1.0/t_safe, -f2/t_safe]
            ])
            # Jacobian of residual wrt z
            Jz = -J_pi

            # Derivative of z wrt camera entries
            dPdP = torch.zeros((3,12), device=device, dtype=dtype)
            dPdP[0,0:4] = Xj
            dPdP[1,4:8] = Xj
            dPdP[2,8:12] = Xj

            # Derivative of z wrt point entries
            dPdX = Pi[:,:4].clone()
            dPdX[:,3] = 0.0

            # Jacobian wrt camera parameters
            JP = sqrt_w * (Jz @ dPdP)
            # Jacobian wrt point parameters
            JX = sqrt_w * (Jz @ dPdX)

            # Accumulate into camera block
            B[i] += JP.T @ JP
            # Accumulate into point block
            C[j] += JX.T @ JX
            # Accumulate into cross block
            E[(i,j)] = E.get((i,j), torch.zeros((12,4),device=device,dtype=dtype)) + JP.T @ JX

            # Accumulate robust gradient for camera
            gc[i] += JP.T @ r_w
            # Accumulate robust gradient for point
            gp[j] += JX.T @ r_w

        # Print progress for this iteration
        if verbose:
            print(f"[Iter {it}] cost={cost.item():.6e}, lambda={lm_lambda:.2e}")

        # Store best-so-far values for acceptance
        best_theta = theta
        # Store best cost for acceptance
        best_cost = cost
        # Track whether LM step is accepted
        success = False

        # Allocate reduced Schur system blocks
        S = [[torch.zeros((12,12), device=device, dtype=dtype) for _ in range(m)] for _ in range(m)]
        # Initialize reduced RHS from camera gradients
        b = [gc[i].clone() for i in range(m)]

        # Apply LM damping to each camera block
        for i in range(m):
            diagBi = torch.diag(B[i]).clamp_min(1e-12)
            B[i] = B[i] + lm_lambda * torch.diag(diagBi)

        # Loop over points to eliminate them into S and b
        for j in range(n):
            # Skip points with no observations
            if torch.allclose(C[j], torch.zeros_like(C[j])):
                continue

            # Apply LM damping to point block
            diagCj = torch.diag(C[j]).clamp_min(1e-12)
            C_damped = C[j] + lm_lambda * torch.diag(diagCj)

            # Invert the damped point block
            try:
                Cinv = torch.linalg.inv(C_damped)
            except RuntimeError:
                Cinv = torch.linalg.lstsq(C_damped, torch.eye(4,device=device,dtype=dtype)).solution

            # Collect cameras that see this point
            cams = [i for (i,jj) in E.keys() if jj == j]
            # Skip if no camera links
            if len(cams) == 0:
                continue

            # Precompute point-side contribution to RHS
            Cinv_gp = Cinv @ gp[j]

            # Update reduced RHS and blocks for all linked cameras
            for a in cams:
                Eaj = E[(a,j)]
                b[a] = b[a] - Eaj @ Cinv_gp
                for bcam in cams:
                    Ebj = E[(bcam,j)]
                    S[a][bcam] = S[a][bcam] - Eaj @ Cinv @ Ebj.T

        # Add damped camera blocks to reduced diagonal
        for i in range(m):
            S[i][i] = S[i][i] + B[i]

        # Allocate dense reduced matrix
        Smat = torch.zeros((m*12, m*12), device=device, dtype=dtype)
        # Allocate dense reduced RHS
        bvec = torch.zeros((m*12,), device=device, dtype=dtype)
        # Scatter reduced blocks into dense form
        for i in range(m):
            bvec[i*12:(i+1)*12] = b[i]
            for jcam in range(m):
                if S[i][jcam].abs().sum() > 0:
                    Smat[i*12:(i+1)*12, jcam*12:(jcam+1)*12] = S[i][jcam]

        # Try LM steps with current lambda
        for lm_try in range(max_lm_tries):
            # Solve reduced system for camera update
            try:
                delta_c = torch.linalg.solve(Smat, -bvec)
            except RuntimeError:
                delta_c = torch.linalg.lstsq(Smat, (-bvec).unsqueeze(-1)).solution.squeeze(-1)

            # Allocate point updates
            delta_p = [torch.zeros((4,), device=device, dtype=dtype) for _ in range(n)]
            # Back-substitute each point update
            for j in range(n):
                if torch.allclose(C[j], torch.zeros_like(C[j])):
                    continue

                diagCj = torch.diag(C[j]).clamp_min(1e-12)
                C_damped = C[j] + lm_lambda * torch.diag(diagCj)

                try:
                    Cinv = torch.linalg.inv(C_damped)
                except RuntimeError:
                    Cinv = torch.linalg.lstsq(C_damped, torch.eye(4,device=device,dtype=dtype)).solution

                cams = [i for (i,jj) in E.keys() if jj == j]
                if len(cams)==0:
                    continue

                # Accumulate camera influence on this point
                Et_dc = torch.zeros((4,), device=device, dtype=dtype)
                for i in cams:
                    Eij = E[(i,j)]
                    dc_i = delta_c[i*12:(i+1)*12]
                    Et_dc += Eij.T @ dc_i

                # Compute final point step
                delta_p[j] = -Cinv @ (gp[j] + Et_dc)

            # Allocate full parameter update vector
            delta_theta = torch.zeros_like(theta)
            # Insert camera updates into delta_theta
            for i in range(m):
                delta_theta[i*12:(i+1)*12] = delta_c[i*12:(i+1)*12]
            # Insert point updates into delta_theta
            for j in range(n):
                base = m*12 + j*4
                delta_theta[base:base+4] = delta_p[j]

            # Initialize line search scale
            step_scale = 1.0
            # Track accepted candidate
            theta_new = None
            # Track accepted cost
            cost_new = None
            # Backtracking line search
            for _ in range(6):
                cand = theta + step_scale * delta_theta
                P_c, X_c = unpack_params(cand)
                P_c = normalize_cameras(P_c)
                cand = pack_params(P_c, X_c)
                c_cost = compute_cost(P_c, X_c)
                if c_cost < best_cost:
                    theta_new, cost_new = cand, c_cost
                    break
                step_scale *= 0.5

            # Accept step if it decreased cost
            if theta_new is not None:
                best_theta, best_cost = theta_new, cost_new
                success = True
                lm_lambda = lm_lambda / lm_lambda_factor
                theta = best_theta
                break
            # Increase lambda if step failed
            else:
                lm_lambda = lm_lambda * lm_lambda_factor

        # Stop if no LM step was accepted
        if not success:
            if verbose:
                print("  Schur-LM failed to decrease cost; stopping.")
            break

        # Stop if improvement is tiny
        if torch.abs(cost - best_cost) < 1e-9:
            if verbose:
                print("  Converged: small cost change.")
            break

    # Unpack final parameters
    P_final, X_final = unpack_params(theta)
    # Normalize cameras one last time
    P_final = normalize_cameras(P_final)

    return P_final, X_final
