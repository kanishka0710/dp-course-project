from __future__ import annotations

import numpy as np

from belief_state import BeliefState
from config import MCTSConfig
from environment import BeamformingEnvironment
from monte_carlo_tree import MonteCarloTreeSearch
from rollout_policy import ThresholdRolloutPolicy
from state import BeamformingState
import matplotlib.pyplot as plt


def make_initial_state(config: MCTSConfig) -> BeamformingState:
    """
    Build a placeholder initial state for testing the MCTS loop.
    Replace with real initialisation once partner's channel code is ready.
    """
    n_reflectors = 3
    n_tx = 4   # transmit array size (adjust to match partner's UPA dimensions)
    n_rx = 4   # receive array size

    return BeamformingState(
        sinr_ul=10.0,
        sinr_dl=10.0,
        H_SI_estimate=np.zeros((n_rx, n_tx), dtype=complex),
        beam_f=np.ones(n_tx, dtype=complex) / np.sqrt(n_tx),
        beam_w=np.ones(n_rx, dtype=complex) / np.sqrt(n_rx),
        reflector_positions=np.zeros((n_reflectors, 3)),
        position_uncertainty=np.ones((n_reflectors, 3)) * 0.1,
        channel_age=0,
        timestep=0,
    )


def main() -> None:
    config = MCTSConfig()
    belief = BeliefState(config)

    # Known velocity V for each reflector (N x 3): [V_R, V_phi, V_theta]
    # Replace with real values from multi-modal sensing data.
    n_reflectors = 3
    known_velocity = np.zeros((n_reflectors, 3))

    env = BeamformingEnvironment(
        config=config,
        belief=belief,
        known_velocity=known_velocity,
    )

    rollout_policy = ThresholdRolloutPolicy(config)
    planner = MonteCarloTreeSearch(env, rollout_policy, config)

    state = make_initial_state(config)

    # Placeholder true channel — replace with partner's channel simulator.
    true_H_SI = state.H_SI_estimate.copy()

    print("Running MCTS simulation loop...")
    n_timesteps = 2000
    total_reward = 0.0

    plt.figure()

    for t in range(n_timesteps):
        action = planner.search(state, true_H_SI)
        next_state, reward = env.step(state, action, true_H_SI)
        total_reward += reward
        print(f"t={t:3d}  action={action}  reward={reward:.4f}  cumulative={total_reward:.4f}")
        state = next_state
        plt.scatter(t, reward)
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Reward over Time")
    plt.show()

    print(f"\nTotal reward over {n_timesteps} timesteps: {total_reward:.4f}")


if __name__ == "__main__":
    main()
