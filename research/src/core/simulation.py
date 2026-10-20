import matplotlib.pyplot as plt
import numpy as np

class Simulation():
    '''
    Simulation Class
     - times
     - obstacles
     - logging
    
    TODO: add config.py as an input to initialize
    '''
    def __init__(self, dt, end_time):
        self.curr_time = 0.0
        self.dt = dt
        self.end_time = end_time

        self.logs = {
            "state": [],
            "u": [],
            "dist": [],
            "times": [],
            "query_points": [],
            "grad": [],
            "h_mean": [],
            "h_std": [],
            "lambda": [],
        }
    def is_running(self):
        return self.curr_time < self.end_time

    def arrived_at_goal(self, goal_radius, GIBO):
        if GIBO.dist_to_goal <= goal_radius:
            GIBO.goal_reached = True

    def step(self, print_t):
        self.curr_time += self.dt
        if print_t:
            print(f"Current Time: {self.curr_time}")

    def log_data(self, X, u, dist, time, query_points=None):
        self.logs["state"].append(X)
        self.logs["u"].append(u)
        self.logs["dist"].append(dist)
        self.logs["times"].append(time)
        if query_points is not None:
                self.logs["query_points"].append(query_points.flatten().tolist())

    def plot_trajectory(self, GIBO):
        X = []
        X_p = []
        Y = []
        Y_p = []
        for State in self.logs["state"]:
            X.append(State.x)
            X_p.append(State.xp)
            Y.append(State.y)
            Y_p.append(State.yp)
        fig, ax = plt.subplots(figsize=(8, 8))

        # Robot center and lookahead point trajectories
        ax.plot(X, Y, color='blue', linewidth=1.5, label='Robot center')
        ax.plot(X_p, Y_p, color='green', label='Lookahead point')

        # Start marker
        ax.scatter(X[0], Y[0], color='blue', marker='o', s=80, zorder=5)

        # Obstacle (close the loop by appending first point)
        obs = GIBO.Obstacle.circle_points
        ax.plot(np.append(obs[:, 0], obs[0, 0]),
                np.append(obs[:, 1], obs[0, 1]),
                color='red', linewidth=2, label='Obstacle')
        ax.fill(obs[:, 0], obs[:, 1], color='red', alpha=0.15)

        # Waypoint
        ax.scatter(GIBO.waypoint[0], GIBO.waypoint[1],
                   color='green', marker='*', s=250, zorder=5, label='Waypoint')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title(f"GIBO Trajectory | M samples: {GIBO.M_number_of_queries}")
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()



    def log_data_manipulator(self, theta, tau, h, t, query_points, grad,
                             h_mean=None, h_std=None, lambda_val=None):
        self.logs["state"].append(theta.tolist())
        self.logs["u"].append(tau.tolist())
        self.logs["dist"].append(float(h))
        self.logs["times"].append(t)
        self.logs["query_points"].append(query_points.flatten().tolist())
        self.logs["grad"].append(grad)
        if h_mean is not None:
            self.logs["h_mean"].append(float(h_mean))
        if h_std is not None:
            self.logs["h_std"].append(float(h_std))
        if lambda_val is not None:
            self.logs["lambda"].append(float(lambda_val))

    def save_figure(self, fig, save_path=None):
        """Lay out, optionally save to save_path (PNG, parent folders created), then show."""
        plt.tight_layout()
        if save_path is not None:
            from pathlib import Path
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot → {save_path}")
        plt.show()

    def plot_acq_function(self, arm, n_sigma=2.0, save_path=None):
        t_arr     = np.array(self.logs["times"])
        th_arr    = np.array(self.logs["state"])
        qp_arr    = np.array(self.logs["query_points"])
        h_std_arr = np.array(self.logs["h_std"]) if self.logs["h_std"] else None

        ref = np.array([arm.reference(t)[0] for t in t_arr])  # (N, 2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"GIBO + HOCBF — 2-DOF Planar Arm   ($\\theta_1 \\leq 150°$,  M={arm.M_number_of_queries},  $\\lambda = 1.0$)",
            fontweight="bold",
        )
        BLUE, RED, ORANGE = "#2563EB", "#DC2626", "#F59E0B"

        # ── (a) θ₁ tracking ──────────────────────────────────────────────────
        ax = axes[0]
        th1_deg = np.degrees(th_arr[:, 0])
        ax.plot(t_arr, np.degrees(ref[:, 0]), "--", color="gray", lw=1.2, label="reference")
        ax.plot(t_arr, th1_deg, color=BLUE, lw=1.5, label="GIBO+HOCBF")
        if h_std_arr is not None:
            half_band = np.degrees(h_std_arr) * n_sigma
            ax.fill_between(t_arr, th1_deg - half_band, th1_deg + half_band,
                            color=BLUE, alpha=0.15, label=f"±{n_sigma}σ GP")
        ax.scatter(t_arr, np.degrees(qp_arr[:, 0]),
                   s=10, color=ORANGE, alpha=0.6, zorder=4, label="query pts")
        ax.axhline(150, color=RED, ls="--", lw=1.2, label="theta_max")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("θ₁ (deg)")
        ax.set_title("Joint 1 angle")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ── (b) Posterior covariance over time ───────────────────────────────
        ax = axes[1]
        if h_std_arr is not None:
            ax.plot(t_arr, h_std_arr, color=BLUE, lw=1.5, label="posterior std (√variance)")
            ax.fill_between(t_arr, 0, h_std_arr, color=BLUE, alpha=0.15)
            print(f"Sum of Posterior Covariance: {np.sum(h_std_arr)}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Posterior Covariance")
        ax.set_title("Posterior Covariance over time")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_random(self, arm, n_sigma=2.0):
        t_arr     = np.array(self.logs["times"])
        th_arr    = np.array(self.logs["state"])
        qp_arr    = np.array(self.logs["query_points"])
        h_std_arr = np.array(self.logs["h_std"]) if self.logs["h_std"] else None

        ref = np.array([arm.reference(t)[0] for t in t_arr])  # (N, 2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Random Querying — 2-DOF Planar Arm   ($\\theta_1 \\leq 150°$,  M={arm.M_number_of_queries},  $\\lambda = 0.0$)",
            fontweight="bold",
        )
        BLUE, RED, ORANGE = "#2563EB", "#DC2626", "#F59E0B"

        # ── (a) θ₁ tracking ──────────────────────────────────────────────────
        ax = axes[0]
        th1_deg = np.degrees(th_arr[:, 0])
        ax.plot(t_arr, np.degrees(ref[:, 0]), "--", color="gray", lw=1.2, label="reference")
        ax.plot(t_arr, th1_deg, color=BLUE, lw=1.5, label="GIBO+HOCBF")
        if h_std_arr is not None:
            half_band = np.degrees(h_std_arr) * n_sigma
            ax.fill_between(t_arr, th1_deg - half_band, th1_deg + half_band,
                            color=BLUE, alpha=0.15, label=f"±{n_sigma}σ GP")
        ax.scatter(t_arr, np.degrees(qp_arr[:, 0]),
                   s=10, color=ORANGE, alpha=0.6, zorder=4, label="query pts")
        ax.axhline(150, color=RED, ls="--", lw=1.2, label="theta_max")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("θ₁ (deg)")
        ax.set_title("Joint 1 angle")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ── (b) Posterior covariance over time ───────────────────────────────
        ax = axes[1]
        if h_std_arr is not None:
            ax.plot(t_arr, h_std_arr, color=BLUE, lw=1.5, label="posterior std (√variance)")
            ax.fill_between(t_arr, 0, h_std_arr, color=BLUE, alpha=0.15)
            print(f"Sum of Posterior Covariance: {np.sum(h_std_arr)}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Posterior Covariance")
        ax.set_title("Posterior Covariance over time")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_composite(self, arm, class_k_str, n_sigma=2.0):
        t_arr     = np.array(self.logs["times"])
        th_arr    = np.array(self.logs["state"])
        qp_arr    = np.array(self.logs["query_points"])
        h_std_arr = np.array(self.logs["h_std"]) if self.logs["h_std"] else None

        ref     = np.array([arm.reference(t)[0] for t in t_arr])  # (N, 2)
        lam_arr = np.array(self.logs["lambda"]) if self.logs["lambda"] else None

        lam_str = (f"$\\lambda$ adaptive: mean={lam_arr.mean():.2f}, "
                   f"range=[{lam_arr.min():.2f}, {lam_arr.max():.2f}] | Class K (CBF): {class_k_str}"
                   if lam_arr is not None else "$\\lambda$ = N/A")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"Adaptive $\\lambda$ Acq. Function — 2-DOF Planar Arm   "
            f"($\\theta_1 \\leq 150°$,  M={arm.M_number_of_queries},  {lam_str})",
        )
        BLUE, RED, ORANGE = "#2563EB", "#DC2626", "#F59E0B"

        # ── (a) θ₁ tracking ──────────────────────────────────────────────────
        ax = axes[0]
        th1_deg = np.degrees(th_arr[:, 0])
        ax.plot(t_arr, np.degrees(ref[:, 0]), "--", color="gray", lw=1.2, label="reference")
        ax.plot(t_arr, th1_deg, color=BLUE, lw=1.5, label="GIBO+HOCBF")
        if h_std_arr is not None:
            half_band = np.degrees(h_std_arr) * n_sigma
            ax.fill_between(t_arr, th1_deg - half_band, th1_deg + half_band,
                            color=BLUE, alpha=0.15, label=f"±{n_sigma}σ GP")
        ax.scatter(t_arr, np.degrees(qp_arr[:, 0]),
                   s=10, color=ORANGE, alpha=0.6, zorder=4, label="query pts")
        ax.axhline(150, color=RED, ls="--", lw=1.2, label="theta_max")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("θ₁ (deg)")
        ax.set_title("Joint 1 angle")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ── (b) Posterior covariance + λ(x) over time ────────────────────────
        ax = axes[1]
        if h_std_arr is not None:
            ax.plot(t_arr, h_std_arr, color=BLUE, lw=1.5, label="posterior std (√variance)")
            ax.fill_between(t_arr, 0, h_std_arr, color=BLUE, alpha=0.15)
            print(f"Sum of Posterior Covariance: {np.sum(h_std_arr)}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Posterior Covariance", color=BLUE)
        ax.set_title("Posterior Covariance & λ(x) over time")
        ax.tick_params(axis='y', labelcolor=BLUE)
        ax.grid(alpha=0.3)

        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        plt.show()


    def plot_manipulator(self, arm, n_sigma=2.0):
        t_arr     = np.array(self.logs["times"])
        th_arr    = np.array(self.logs["state"])
        qp_arr    = np.array(self.logs["query_points"])
        h_std_arr = np.array(self.logs["h_std"]) if self.logs["h_std"] else None

        ref = np.array([arm.reference(t)[0] for t in t_arr])  # (N, 2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"GIBO + HOCBF — 2-DOF Planar Arm   ($\\theta_1 \\leq 150°$,  M={arm.M_number_of_queries})",
            fontweight="bold",
        )
        BLUE, RED, ORANGE = "#2563EB", "#DC2626", "#F59E0B"

        # ── (a) θ₁ tracking ──────────────────────────────────────────────────
        ax = axes[0]
        th1_deg = np.degrees(th_arr[:, 0])
        ax.plot(t_arr, np.degrees(ref[:, 0]), "--", color="gray", lw=1.2, label="reference")
        ax.plot(t_arr, th1_deg, color=BLUE, lw=1.5, label="GIBO+HOCBF")
        if h_std_arr is not None:
            half_band = np.degrees(h_std_arr) * n_sigma
            ax.fill_between(t_arr, th1_deg - half_band, th1_deg + half_band,
                            color=BLUE, alpha=0.15, label=f"±{n_sigma}σ GP")
        ax.scatter(t_arr, np.degrees(qp_arr[:, 0]),
                   s=10, color=ORANGE, alpha=0.6, zorder=4, label="query pts")
        ax.axhline(150, color=RED, ls="--", lw=1.2, label="theta_max")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("θ₁ (deg)")
        ax.set_title("Joint 1 angle")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # ── (b) Posterior covariance over time ───────────────────────────────
        ax = axes[1]
        if h_std_arr is not None:
            ax.plot(t_arr, h_std_arr, color=BLUE, lw=1.5, label="posterior std (√variance)")
            ax.fill_between(t_arr, 0, h_std_arr, color=BLUE, alpha=0.15)
            print(f"Sum of Posterior Covariance: {np.sum(h_std_arr)}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Posterior Covariance")
        ax.set_title("Posterior Covariance over time")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def list_to_np(self):
        processed_logs = {}
        for key, value in self.logs.items():
            if key == "state":
                # Extract x and y attributes from each State object
                # This turns a list of Objects into a [N, 2] float matrix
                processed_logs["state"] = np.array([[s.x, s.y] for s in value])
            elif key == "query_points":
                # Ensure query points are also a 2D array [N, 2]
                processed_logs["query_points"] = np.array(value)
            else:
                processed_logs[key] = np.array(value)
        return processed_logs