import torch
import time
torch.set_default_dtype(torch.float64)

import gpytorch as gp
from gpytorch.priors import GammaPrior
import numpy as np

from src.helpers.gp_simple import RBF_example
from src.core.simulation import Simulation
from src.core.manipulator import Manipulator
from config import RunConfig
from src.core.logger import TRUE_GRAD_H     # true ∇h = [−1, 0] for h = θ_max − θ₁

def main(cfg: RunConfig = None, logger=None):
    """logger: optional RunLogger. With a logger the run is headless (no plots) and returns the summary dict."""
    cfg = cfg or RunConfig()    # defaults reproduce the pre-config behavior
    start_time = time.perf_counter()

    # -- Select Algorithm / Convex Opt. Problem --
    algo       = {"oracle": "0", "gibo": "1", "random": "2", "lambda": "3"}[cfg.algo]   # legacy branch keys; replaced in run(cfg) refactor
    optim_type = cfg.optim_type                                        # "QP" | "SOCP"

    # i. Initialize Sim
    sim = Simulation(dt=cfg.sim.dt, end_time=cfg.sim.end_time)
    # NOTE: dt must be sufficiently small to not explode

    # -- GIBO Line 1: Setup GIBO Hyperparameters -- 
    # a. stepsize η (Not applicable, step forward using numerical integration with dt)
    # -- N/A --

    # b. number of iterations -> N = 201 (20 seconds, 0.1s timestep)
    N = sim.dt
    
    # c. number of samples for gradient estimate -> M =
    M_samples = cfg.gibo.M
    Arm = Manipulator(alpha_cbf=cfg.cbf.alpha_cbf, M_number_of_queries=M_samples, dt=N, class_k=cfg.cbf.class_k, cfg=cfg)
    n_sigma = cfg.gp.n_sigma

    # e. Fix GP Hyper-parameters based on Manipulator noise + rough estimate of lengthscale.
    RBF_gaussian_likelihood = gp.likelihoods.GaussianLikelihood()
    GP_model = RBF_example(train_x=[], train_y=[],               # Empty Training set to start (no warm-start)
                           likelihood=RBF_gaussian_likelihood,   # Gaussian Probability Distribution
                           l_prior=None, output_prior=None)      
    l_0     = cfg.gp.l_0                                         # Length-scale (rough derivation in config.GPConfig)
    S0      = cfg.gp.S0                                          # Output-scale (rough derivation in config.GPConfig)
    NOISE_0 = Arm.Noise.noise_std_dev**2                         # Feeds the true noise distribution to our GP

    GP_model.covar_module.base_kernel.lengthscale = l_0          # Setup GP with initial estimates of lengthscale / outputscale
    GP_model.covar_module.outputscale             = S0           # Setup likelihood with NOISE_0 estimate
    RBF_gaussian_likelihood.noise                 = NOISE_0      # Likelihood [sigma_n^2 * I] in the posterior covaraince calculation

    GP_model.eval()                                              # Evaluation mode causes GP_model(x) to return posterior
    RBF_gaussian_likelihood.eval()

    # -- GIBO Line 2: Set theta1 and theta2 (next_query_point) and empty dataset D = {} --
    Arm.D = {"thetas": [], "h": []}

    next_query_point = torch.tensor([Arm.theta.tolist()])  # initial query at arm's joint state

    # -- Safety metrics --
    cbf_violations = 0
    min_h = float("inf")

    # -- GIBO Line 3: for t = 0, ..., N do --
    while sim.is_running():
        step_t0  = time.perf_counter()
        t_now    = sim.curr_time
        theta_t  = Arm.theta.copy()                 # θ_t — where the GP posterior and queries are centred
        dtheta_t = Arm.dtheta.copy()

        if algo != "0":   # -- Learned h: GIBO steps 4–12 --
            # -- GIBO Line 4: Sample Noisy Objective Function | J(θ_t) + ε_t --
            h_noisy = Arm.observe_J(theta=Arm.theta)  # noisy J — added to GP dataset

            # -- GIBO Line 5: Extend Dataset D --
            Arm.D["thetas"].append(Arm.theta.tolist())  # .tolist() avoids reference sharing with rk4_step
            Arm.D["h"].append(h_noisy)
            train_x = torch.tensor(Arm.D["thetas"])
            train_y = torch.tensor(Arm.D["h"])

            # -- GIBO Line 6 (Retraining Skipped for Safety) : Construct Training Data for GIBO / GP --
            GP_model.set_train_data(inputs=train_x, 
                                    targets=train_y,
                                    strict=False)
            sigma2 = GP_model.covar_module.outputscale.detach()
            l = GP_model.covar_module.base_kernel.lengthscale.detach().squeeze(0)   # Length-scale (defines local search space for acq function to select over)
            obs_noise = RBF_gaussian_likelihood.noise.detach() 

            # -- GIBO Step 7: For m = 1,2, ... M:
            query_space = Arm.return_gridspace(wander_angle_deg=cfg.gibo.box_factor*np.rad2deg(float(l.max())))    # Defines the LOCAL Search Space for acquisition function (creates space to learn gradient)
            # -- Step 8: Get query point = argmax(acq_function) --
            queries, acq_vals, acq_time = [], [], 0.0
            for m in range(Arm.M_number_of_queries): # Creates
                acq_t0 = time.perf_counter()
                # -- GIBO (Original) --
                if algo == "1":
                    # -- Vanilla GIBO --
                    next_query_point, query_covariance_gain = Arm.acquisition_function(train_x=train_x, train_y=train_y, 
                                                            GP_model=GP_model, GP_likelihood=RBF_gaussian_likelihood, 
                                                            sigma2=sigma2, l=l, obs_noise=obs_noise, 
                                                            next_query_point=next_query_point, query_space=query_space, m=m)
                    print(f"QP Covariance Gain: {query_covariance_gain}")

                elif algo == "2":
                    # -- Randomly selected acquisition function --
                    next_query_point, query_covariance_gain = Arm.random_acq_function(query_space=query_space)
                    print(f"QP Covariance Gain: {query_covariance_gain}")

                elif algo == "3":
                    # -- Experimental Lambda-weighted GIBO --
                    next_query_point, query_covariance_gain = Arm.composite_acq_function(
                        train_x, GP_model, RBF_gaussian_likelihood, l, query_space)
                    print(f"Lambda Value: {query_covariance_gain}")

                else:
                    raise Exception("Invalid Index for acquisition function.")
                acq_time += time.perf_counter() - acq_t0
                queries.append(next_query_point.squeeze().tolist())
                acq_vals.append(float(query_covariance_gain))

                # -- Step 9: Sample Noisy Objective Function J(query_point) + noise
                angular_dist_qp = Arm.observe_J(theta=next_query_point.squeeze().numpy())

                # -- Step 10: Extend Data Set to include next query point --
                Arm.D["thetas"].append(next_query_point.squeeze().tolist())
                Arm.D["h"].append(angular_dist_qp)
                train_x = torch.tensor(Arm.D["thetas"], dtype=torch.float64)
                train_y = torch.tensor(Arm.D["h"], dtype=torch.float64)

                # -- Step 11: Update the posterior of ∇θJ <-> (∇ Signed Distance)
                weights = Arm.compute_gradient_update_weights(train_x, train_y, GP_model, GP_model.likelihood).unsqueeze(-1) # (3,1)
                grad_K = Arm.compute_grad_K(train_x, GP_model, lengthscale=l)
                expected_grad_ang_dist = torch.sum(weights * grad_K, dim=0).detach()

            # -- Step 12: End Inner For Loop  (Also add safety margin to reading) --
            GP_model.set_train_data(inputs=train_x, targets=train_y, strict=False)
            x_curr = torch.tensor([[Arm.theta[0], Arm.theta[1]]])           # Evaluate
            with torch.no_grad(), gp.settings.max_cholesky_size(cfg.gp.max_cholesky_size):     
                pred   = GP_model(x_curr)
                h_mean = pred.mean.item()
                h_std  = pred.variance.sqrt().item()
            h_safe = h_mean - n_sigma * h_std
            grad_std = Arm.compute_grad_h_uncertainty(train_x, GP_model, RBF_gaussian_likelihood, l)   # logged every step
        else:            # -- Oracle: perfect knowledge of h and ∇h; no readings, GP, or queries --
            h_mean = h_safe = Arm.THETA_MAX - Arm.theta[0]
            h_std  = 0.0
            expected_grad_ang_dist = torch.tensor([-1.0, 0.0])
            grad_std = 0.0               # gradient known exactly → SOCP robustness cone vanishes
            queries, acq_vals, acq_time, h_noisy = [], [], 0.0, np.nan

        # -- Step 13:  Solves CBF-QP X_{t+1} = X_t + η·E[∇J | X=X_t] --
        torque_nom = Arm.u_nom(sim.curr_time)

        if optim_type == "QP":
            Arm.u_final = Arm.solve_hocbf_qp(torque_nom=torque_nom, h=h_safe, grad_h=expected_grad_ang_dist)
        elif optim_type == "SOCP":
            e_grad_h = grad_std                     # same value as before — the function is deterministic
            Arm.u_final = Arm.solve_hocbf_SOCP(torque_nom=torque_nom, h=h_mean, e_h=h_safe, grad_h=expected_grad_ang_dist, e_grad_h=e_grad_h)
        else:
            raise Exception("Invalid optimization type.")
        Arm.rk4_step(Arm.u_final)  # updates self.theta, self.dtheta
        h_true = Arm.THETA_MAX - Arm.theta[0]
        if h_true < 0:
            cbf_violations += 1
        if h_true < min_h:
            min_h = h_true

        # ----------------------------------------------------------
        # -- End of Algorithm. Repeat and plot. --
        # ----------------------------------------------------------
        if logger is not None:
            q = np.array(queries).reshape(-1, 2)                       # (M, 2); (0, 2) for the oracle
            logger.log_step(
                t=t_now, theta=theta_t, dtheta=dtheta_t, ref=Arm.reference(t_now)[0],
                u=Arm.u_final, u_nom=torque_nom,
                h_true=h_true, h_true_t=Arm.THETA_MAX - theta_t[0], h_noisy=h_noisy,
                h_mean=h_mean, h_std=h_std, h_safe=h_safe,
                grad_mean=np.asarray(expected_grad_ang_dist), grad_var=[np.nan, np.nan], grad_std=grad_std,
                grad_err=np.linalg.norm(np.asarray(expected_grad_ang_dist) - TRUE_GRAD_H),   # ‖μ_∇ − ∇h‖ per step
                queries=q, query_offset=np.linalg.norm(q - theta_t, axis=1) / cfg.gp.l_0,
                query_h_true=Arm.THETA_MAX - q[:, 0], acq_value=acq_vals,
                n_data=len(Arm.D["h"]),
                cbf_active=float(np.linalg.norm(Arm.u_final - torque_nom) > 1e-6),
                socp_slack=np.nan,
                step_time=time.perf_counter() - step_t0, acq_time=acq_time)

        sim.log_data_manipulator(
            theta=Arm.theta, tau=Arm.u_final, h=h_noisy,
            t=sim.curr_time, query_points=next_query_point,
            grad=expected_grad_ang_dist.tolist(),
            h_mean=h_mean, h_std=h_std,
            lambda_val=query_covariance_gain if algo == "3" else None,
        )
        sim.step(print_t=True)
    end_time = time.perf_counter()
    print(f"Total Runtime (s): {(end_time - start_time):.4f}")
    print(f"CBF Violations:    {cbf_violations}")
    print(f"Min h value (deg): {np.degrees(min_h):.3f}")
    if logger is not None:
        logger.status = "ok"
        return logger.save()        # sweep mode: save + return summary, skip plots
    if algo == "1":
        sim.plot_acq_function(Arm, n_sigma=n_sigma)
    elif algo == "2":
        sim.plot_random(Arm, n_sigma=n_sigma)
    elif algo == "3":
        sim.plot_composite(Arm, class_k_str=Arm.class_K, n_sigma=n_sigma,)
    else:
        raise Exception("Invalid Index for acquisition function.")

if __name__ == "__main__":
    main()
    print("End Function")
