from dataclasses import dataclass


@dataclass
class MCTSConfig:
    # UCT exploration constant
    exploration_c: float = 1.414

    # Number of MCTS simulations to run before committing to an action
    n_simulations: int = 200

    # Maximum rollout depth (timesteps simulated per leaf evaluation)
    rollout_depth: int = 10

    # Discount factor applied to future rewards during rollout
    discount_gamma: float = 0.95

    # Number of beams in the probing codebook (align with partner's codebook size)
    n_probe_beams: int = 8

    # Per-timestep variance added to reflector position uncertainty when not probing
    # Corresponds to the unknown variation v in the dynamics model
    process_noise_variance: float = 0.01

    # Factor by which probing reduces position uncertainty (0 < value <= 1)
    # 1.0 means probing fully resolves uncertainty; lower values = noisier measurements
    probe_uncertainty_reduction: float = 0.1

    # SINR threshold below which the rollout policy prefers probing over serving
    sinr_threshold_db: float = 5.0

    # Maximum channel age (timesteps) before the rollout policy always probes
    max_channel_age: int = 5
