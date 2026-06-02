from dataclasses import dataclass


@dataclass
class DPConfig:
    # Array dimensions
    n_tx: int = 36
    n_rx: int = 36

    # DP state space
    max_age: int = 20           # maximum channel_age tracked as a state
    n_sinr_bins: int = 36       # number of discrete SINR bins
    sinr_min_db: float = -2.0   # lower edge of SINR grid (dB)
    sinr_max_db: float = 80.0  # upper edge of SINR grid (dB)

    # DP solver
    probe_cost: float = 0.0     # reward penalty applied when probing
    discount: float = 0.95      # discount factor gamma
    vi_max_iters: int = 500     # maximum value iteration sweeps
    vi_tol: float = 1e-6        # convergence tolerance for value iteration

    # Channel drift (random walk model, no known reflector velocity)
    drift_scale: float = 0.2    # std of complex Gaussian noise added per timestep

    # Monte Carlo precomputation
    n_mc_samples: int = 300     # trajectories per (age, sinr_bin) cell

    # Simulation
    n_timesteps: int = 300      # length of evaluation episode
