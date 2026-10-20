"""
Central configuration for GIBO_Robotic_Manipulator runs and sweeps.

Every hard-coded constant from GIBO_Robotic_Manipulator.py, src/core/manipulator.py and
src/core/simulation.py lives here. Defaults reproduce the pre-config behavior.
Dump a run's config with `json.dumps(asdict(cfg))`.
"""
from dataclasses import dataclass, field, asdict, replace
import itertools
import numpy as np


@dataclass
class SimConfig:
    dt: float       = 0.04      # [s] integration + GIBO step (must be small or RK4 explodes)
    end_time: float = 15.0      # [s]


@dataclass
class ArmConfig:
    # -- Physical parameters (manipulator.py:24–26) --
    m1: float  = 1.0
    m2: float  = 1.0
    l1: float  = 1.0
    l2: float  = 1.0
    lc1: float = 0.5
    lc2: float = 0.5
    # -- Tracking controller gains (manipulator.py:41–42) --
    KP: float = 100.0
    KD: float = 20.0
    # -- Reference r(t) = [A1·sin(w1 t), A2·sin(w2 t)] (manipulator.py:98) --
    ref_amp: tuple  = (np.pi, np.pi / 2)
    ref_freq: tuple = (1.0, 4.0)


@dataclass
class CBFConfig:
    THETA_MAX_deg: float = 150.0    # safety limit θ₁ ≤ θ_max
    CBF_GAMMA: float     = 5.0      # ψ = ḣ + γ·α₁(h)
    CBF_ALPHA: float     = 5.0      # ψ̇ + α·ψ ≥ 0
    class_k: str         = "linear" # linear | sqrt | cubic | quadratic | tanh
    alpha_cbf: float     = 0.5      # V_h in compute_lambda (λ-acquisition only)
    qp_grad_norm_eps: float = 1e-6  # QP falls back to g = [-1, 0] below this ‖μ_∇‖
    socp_slack_weight: float = 1e6  # penalty on SOCP slack


@dataclass
class GPConfig:
    # Fixed hyperparameters (no retraining → chosen, not learned)
    l_0: float = 1.0        # l [rad]: (max step travel 0.28)/0.46 ≈ 0.6 ≤ l ≤ (θ₁ span 5.76)/3 ≈ 1.9
    S0: float  = 11.8       # σ² [rad²] ≈ E[h²] = Var(h) + mean(h)² = 4.57 + 2.68² over the reference sweep
    GP_JITTER: float = 1e-6             # added on top of σₙ² in K_XX_noisy (not a floor)
    max_cholesky_size: int = 10**6      # force exact solves (GPyTorch default 800 → CG/Lanczos, wrong variances)
    n_sigma: float = 2.0                # h_safe = μ − n_sigma·σ


@dataclass
class GIBOConfig:
    M: int = 4                  # acquisition queries per step
    box_factor: float = 2.0     # local search box half-width = box_factor · ℓ
    grid_n: int = 8             # candidates per axis → grid_n² candidates


@dataclass
class RunConfig:
    algo: str        = "gibo"   # oracle | gibo | random | lambda
    optim_type: str  = "SOCP"   # QP | SOCP
    seed: int        = 2        # sensor-noise RNG (acquisition RNG uses seed + 1000)
    noise_std: float = 0.1      # sensor noise std [rad]; GP noise variance = noise_std²
    theta0_offset: tuple = (0.3, -0.2)   # θ₀ = r(0) + offset; θ̇₀ = ṙ(0)
    theta0_id: int   = -1       # index into SweepConfig.theta0_offsets (-1 = custom)

    sim: SimConfig   = field(default_factory=SimConfig)
    arm: ArmConfig   = field(default_factory=ArmConfig)
    cbf: CBFConfig   = field(default_factory=CBFConfig)
    gp: GPConfig     = field(default_factory=GPConfig)
    gibo: GIBOConfig = field(default_factory=GIBOConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def run_name(self) -> str:
        return (f"{self.algo}_{self.optim_type}_M{self.gibo.M}_n{self.noise_std:g}"
                f"_th{self.theta0_id}_s{self.seed}")


@dataclass
class SweepConfig:
    algos: tuple       = ("gibo", "random", "lambda")
    optim_type: str    = "SOCP"
    noise_stds: tuple  = (0.05, 0.2, 0.5)
    Ms: tuple          = (1, 2, 3, 4)
    seeds: tuple       = (0, 1, 2)
    include_oracle: bool = True                # oracle runs once
    theta0_offsets: tuple = ((-0.025,  0.093), # 5 sets of manipulator states from scrambled Sobol
                             ( 0.079, -0.463),
                             ( 0.451,  0.362),
                             (-0.380, -0.239),
                             (-0.285,  0.474))

    def runs(self, base: RunConfig = None):
        """Yield one RunConfig per sweep cell (oracle cells first, then algo × M)."""
        base = base or RunConfig(optim_type=self.optim_type)
        for noise, (k, off), seed in itertools.product(self.noise_stds, enumerate(self.theta0_offsets), self.seeds):
            common = dict(noise_std=noise, theta0_offset=off, theta0_id=k, seed=seed, optim_type=self.optim_type)
            if self.include_oracle:
                yield replace(base, algo="oracle", gibo=replace(base.gibo, M=0), **common)
            for algo, M in itertools.product(self.algos, self.Ms):
                yield replace(base, algo=algo, gibo=replace(base.gibo, M=M), **common)
