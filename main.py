"""
Main entry point: solve the serve/probe DP and evaluate the resulting policy.

Steps
-----
1. Precompute transition matrix P and reward vector R via Monte Carlo.
2. Run value iteration to find the optimal policy (a 2D grid over age × SINR bin).
3. Print the policy as a human-readable table.
4. Simulate the DP policy and a fixed-period baseline over n_timesteps.
5. Plot cumulative reward vs. timestep for both policies.
"""

from __future__ import annotations
import pickle
import numpy as np
import matplotlib.pyplot as plt


from action import Action
from channel import draw_static_si_channel_ula
from config import DPConfig
from dp_solver import (
    bin_to_sinr_db,
    estimate_fresh_sinr_bin,
    precompute_transitions,
    sinr_db_to_bin,
    sinr_to_db,
    value_iteration,
)
from environment import compute_sinr, design_beams, step
from state import State


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_initial_state(config: DPConfig, true_H: np.ndarray) -> State:
    """Build a starting state by probing the initial true channel."""
    beam_f, beam_w = design_beams(true_H)
    sinr_ul, _, _ = compute_sinr(beam_f, beam_w, true_H)
    return State(
        channel_age=0,
        sinr_ul=sinr_ul,
        H_SI_estimate=true_H.copy(),
        beam_f=beam_f,
        beam_w=beam_w,
    )


def print_policy(policy: np.ndarray, config: DPConfig) -> None:
    """Print the 2D policy as a table: rows = age, columns = SINR bin."""
    print("\nOptimal policy  (S = SERVE,  P = PROBE)")
    print(f"{'age':>4}", end="")
    for s in range(config.n_sinr_bins):
        sinr_label = f"{bin_to_sinr_db(s, config):.0f}"
        print(f"  {sinr_label:>5}", end="")
    print()
    for a in range(config.max_age + 1):
        print(f"{a:>4}", end="")
        for s in range(config.n_sinr_bins):
            symbol = "P" if policy[a, s] else "S"
            print(f"  {'':>4}{symbol}", end="")
        print()
    print("(columns = SINR bin centre, dB)\n")


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

SimResult = dict  # keys: rewards, si_power, sinr_ul_db, sinr_dl_db


def simulate(
    policy: np.ndarray | None,
    probe_period: int | None,
    config: DPConfig,
    true_H_init: np.ndarray,
    label: str,
    channels: list[np.ndarray] | None = None,
) -> SimResult:
    """
    Run one episode and return a dict of per-timestep metric lists.

    Keys
    ----
    rewards    : per-step reward (bits/s/Hz)
    si_power   : self-interference power after beam suppression (linear)
    sinr_ul_db : uplink SINR in dB
    sinr_dl_db : downlink SNR in dB (constant — beam-independent)

    If policy is provided, use the DP policy (2D lookup).
    If probe_period is provided, probe every probe_period steps (fixed baseline).
    If channels is provided, the channel evolves by replaying that sequence instead
    of using the random-walk drift model.
    """
    true_H = true_H_init.copy()
    state = make_initial_state(config, true_H)
    step_index = 0
    rewards: list[float] = []
    si_power_list: list[float] = []
    sinr_ul_db_list: list[float] = []
    sinr_dl_db_list: list[float] = []

    for t in range(config.n_timesteps):
        if policy is not None:
            age = min(state.channel_age, config.max_age)
            sinr_bin = sinr_db_to_bin(sinr_to_db(state.sinr_ul), config)
            action = Action.PROBE if policy[age, sinr_bin] else Action.SERVE
        else:
            action = Action.PROBE if (t % probe_period == 0) else Action.SERVE

        state, reward, true_H, step_index = step(
            state, action, true_H, config, channels=channels, step_index=step_index
        )
        rewards.append(reward)

        sinr_ul, sinr_dl, si_power = compute_sinr(state.beam_f, state.beam_w, true_H)
        si_power_list.append(si_power)
        sinr_ul_db_list.append(10.0 * np.log10(sinr_ul) if sinr_ul > 0 else -np.inf)
        sinr_dl_db_list.append(10.0 * np.log10(sinr_dl) if sinr_dl > 0 else -np.inf)

    return {
        "rewards": rewards,
        "si_power": si_power_list,
        "sinr_ul_db": sinr_ul_db_list,
        "sinr_dl_db": sinr_dl_db_list,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config = DPConfig()

    # --- Load replay channels (used for both MC precomputation and simulation) ---
    with open("H_one_reflection.pkl", "rb") as fh:
        replay_channels = pickle.load(fh)

    # --- Step 1: precompute transition model ---
    print("Precomputing transition matrix (Monte Carlo)...")
    P, R = precompute_transitions(config, channels=replay_channels)
    print(f"  Reward range: {R.min():.2f} – {R.max():.2f} bits/s/Hz")

    fresh_bin = estimate_fresh_sinr_bin(config)
    print(f"  Fresh SINR bin after PROBE: {fresh_bin} "
          f"({bin_to_sinr_db(fresh_bin, config):.1f} dB)")

    # --- Step 2: value iteration ---
    print("\nRunning value iteration...")
    V, policy = value_iteration(P, R, config, fresh_sinr_bin=fresh_bin)

    # --- Step 3: print policy table ---
    print_policy(policy, config)

    # --- Step 4: simulate ---
    true_H_init = replay_channels[0]

    res_dp    = simulate(policy,   None,  config, true_H_init, "DP",         channels=replay_channels)
    res_fix3  = simulate(None,    3,     config, true_H_init, "Fixed-3",    channels=replay_channels)
    res_fix5  = simulate(None,    5,     config, true_H_init, "Fixed-5",    channels=replay_channels)
    res_never = simulate(None,    10**9, config, true_H_init, "Never probe", channels=replay_channels)

    rewards_dp    = res_dp["rewards"]
    rewards_fix3  = res_fix3["rewards"]
    rewards_fix5  = res_fix5["rewards"]
    rewards_never = res_never["rewards"]

    cum_dp    = np.cumsum(rewards_dp)
    cum_fix3  = np.cumsum(rewards_fix3)
    cum_fix5  = np.cumsum(rewards_fix5)
    cum_never = np.cumsum(rewards_never)

    print(f"Total reward  DP        : {cum_dp[-1]:.2f}")
    print(f"Total reward  Fixed-3   : {cum_fix3[-1]:.2f}")
    print(f"Total reward  Fixed-5   : {cum_fix5[-1]:.2f}")
    print(f"Total reward  Never     : {cum_never[-1]:.2f}")

    # --- Step 5: plot ---

    t = np.arange(config.n_timesteps)

    # Figure 1: reward curves (unchanged)
    fig1, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(t, cum_dp,    label="DP policy",     linewidth=2)
    ax.plot(t, cum_fix3,  label="Fixed period=3", linestyle="--")
    ax.plot(t, cum_fix5,  label="Fixed period=5", linestyle="--")
    ax.plot(t, cum_never, label="Never probe",    linestyle=":")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative reward (bits/s/Hz)")
    ax.set_title("Cumulative reward vs. timestep")
    ax.legend()
    ax.grid(True)

    ax2 = axes[1]
    ax2.plot(t, rewards_dp,   label="DP policy",     linewidth=1.5)
    ax2.plot(t, rewards_fix5, label="Fixed period=5", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Per-step reward (bits/s/Hz)")
    ax2.set_title("Per-step reward vs. timestep")
    ax2.legend()
    ax2.grid(True)

    fig1.tight_layout()

    # Figure 2: SI power and SINR breakdown
    fig2, axes2 = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # --- SI power (log scale so small suppression values are visible) ---
    ax_si = axes2[0]
    ax_si.semilogy(t, res_dp["si_power"],   label="DP policy",     linewidth=1.5)
    ax_si.semilogy(t, res_fix5["si_power"], label="Fixed period=5", linestyle="--", alpha=0.8)
    ax_si.set_ylabel("SI power (linear)")
    ax_si.set_title("Self-interference power after beam suppression")
    ax_si.legend()
    ax_si.grid(True, which="both")

    # --- UL SINR (dB) ---
    ax_ul = axes2[1]
    ax_ul.plot(t, res_dp["sinr_ul_db"],   label="DP policy",     linewidth=1.5)
    ax_ul.plot(t, res_fix5["sinr_ul_db"], label="Fixed period=5", linestyle="--", alpha=0.8)
    ax_ul.set_ylabel("SINR_UL (dB)")
    ax_ul.set_title("Uplink SINR  [= desired power / (SI power + noise floor)]")
    ax_ul.legend()
    ax_ul.grid(True)

    # --- DL SNR (dB) — beam-independent, shown for reference ---
    ax_dl = axes2[2]
    ax_dl.plot(t, res_dp["sinr_dl_db"], color="tab:green", linewidth=1.5,
               label="SNR_DL (beam-independent)")
    ax_dl.set_xlabel("Timestep")
    ax_dl.set_ylabel("SNR_DL (dB)")
    ax_dl.set_title("Downlink SNR  [= desired power / noise floor, constant]")
    ax_dl.legend()
    ax_dl.grid(True)

    fig2.suptitle("SI power and SINR/SNR analysis", fontsize=13, fontweight="bold")
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
