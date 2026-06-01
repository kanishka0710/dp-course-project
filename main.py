from __future__ import annotations

import numpy as np

from belief_state import BeliefState
from channel import draw_static_si_channel_ula
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

    # Realistic true SI channel: mix of reflectors + spherical wave.
    # kappa controls reflector vs direct-path balance (0=pure direct, 1=pure reflectors).
    n_tx = state.H_SI_estimate.shape[1]
    n_rx = state.H_SI_estimate.shape[0]
    true_H_SI = draw_static_si_channel_ula(
        N_t=n_tx, N_r=n_rx, sep=10.0, n_reflectors=3, kappa=0.7
    )

    print("Running MCTS simulation loop...")
    n_timesteps = 100
    total_reward = 0.0
    rewards: list[float] = []
    channel_norms: list[float] = []

    for t in range(n_timesteps):
        action = planner.search(state, true_H_SI)
        next_state, reward = env.step(state, action, true_H_SI)
        total_reward += reward
        rewards.append(reward)
        channel_norms.append(float(np.linalg.norm(true_H_SI, "fro")))
        print(f"t={t:3d}  action={action}  reward={reward:.4f}  cumulative={total_reward:.4f}")
        state = next_state
        # Slowly drift the true channel each timestep to simulate reflector motion.
        true_H_SI = env.advance_true_channel(true_H_SI)

    fig, (ax_reward, ax_channel) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    ax_reward.plot(rewards, "b-o", markersize=3)
    ax_reward.set_ylabel("Reward")
    ax_reward.set_title("Reward over Time")
    ax_reward.grid(True)

    ax_channel.plot(channel_norms, "r-o", markersize=3)
    ax_channel.set_xlabel("Timestep")
    ax_channel.set_ylabel(r"$\|H_\mathrm{SI}\|_F$")
    ax_channel.set_title("True SI Channel Magnitude")
    ax_channel.grid(True)

    fig.tight_layout()
    plt.show()

    print(f"\nTotal reward over {n_timesteps} timesteps: {total_reward:.4f}")


if __name__ == "__main__":
    main()
