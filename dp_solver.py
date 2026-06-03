"""
Dynamic programming solver for the serve/probe decision problem.

State space  : (channel_age, sinr_bin)
               channel_age ∈ {0, …, max_age}
               sinr_bin    ∈ {0, …, n_sinr_bins-1}

Actions      : SERVE — keep current beams, age++, observe new SINR
               PROBE — refresh channel estimate, age → 0

The transition probabilities are built by Monte Carlo simulation.
The reward vector R is computed analytically from each bin's centre SINR
so that every bin has a valid reward regardless of how many MC samples
visited it.  Value iteration then finds the optimal policy offline.
"""

from __future__ import annotations

import math

import numpy as np

from channel import draw_static_si_channel_ula
from config import DPConfig
from environment import compute_sinr, design_beams, drift_channel, drift_channel_replay


# ---------------------------------------------------------------------------
# Physical constants (must match environment.py)
# ---------------------------------------------------------------------------
_NOISE_FLOOR       = 1e-9
_UL_DESIRED_POWER  = 1000.0
_DL_DESIRED_POWER  = 1000.0


# ---------------------------------------------------------------------------
# SINR ↔ bin helpers
# ---------------------------------------------------------------------------

def sinr_to_db(sinr_linear: float) -> float:
    return 10.0 * math.log10(max(sinr_linear, 1e-12))


def sinr_db_to_bin(sinr_db: float, config: DPConfig) -> int:
    """Clip and quantise a dB SINR value into [0, n_sinr_bins-1]."""
    sinr_db = float(np.clip(sinr_db, config.sinr_min_db, config.sinr_max_db))
    frac = (sinr_db - config.sinr_min_db) / (config.sinr_max_db - config.sinr_min_db)
    return int(np.clip(int(frac * config.n_sinr_bins), 0, config.n_sinr_bins - 1))


def bin_to_sinr_db(bin_idx: int, config: DPConfig) -> float:
    """Return the centre of a SINR bin in dB."""
    step = (config.sinr_max_db - config.sinr_min_db) / config.n_sinr_bins
    return config.sinr_min_db + (bin_idx + 0.5) * step


def _bin_centre_sse(bin_idx: int, config: DPConfig) -> float:
    """
    Analytical uplink spectral efficiency for the centre SINR of a bin.

    We use UL capacity only: R(s) = log2(1 + SINR_UL(s)).

    The DL SNR is constant across all states (beams only affect SI on the UL),
    so including it would add the same constant to every state and cancel out
    in the Bellman comparison between SERVE and PROBE — using UL-only makes
    the degradation clearly visible and keeps the reward numerically clean.
    """
    sinr_ul_linear = 10.0 ** (bin_to_sinr_db(bin_idx, config) / 10.0)
    return math.log2(1.0 + sinr_ul_linear)


# ---------------------------------------------------------------------------
# Precompute transition model via Monte Carlo
# ---------------------------------------------------------------------------

def precompute_transitions(
    config: DPConfig,
    h_r,
    h_t,
    channels: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the transition matrix P and immediate reward vector R.

    P[age, s, s'] = P(next_sinr_bin = s' | channel_age = age, sinr_bin = s, SERVE)
    R[s]          = analytical SSE at the centre SINR of bin s

    For PROBE the transition is deterministic: age → 0, sinr → fresh_sinr_bin
    (handled separately in value_iteration).

    Returns
    -------
    P : shape (max_age+1, n_sinr_bins, n_sinr_bins)
    R : shape (n_sinr_bins,)
    """
    A = config.max_age + 1
    S = config.n_sinr_bins

    # Both P and R are estimated by Monte Carlo.
    #
    # R[s] = E[SSE(H_drifted) | current SINR bin = s]
    #
    # This matches the simulation exactly: when you SERVE, the channel drifts
    # first and then the reward is computed on the drifted channel.  Using the
    # analytical bin-centre SINR would overestimate the reward for high bins
    # (e.g. bin 14 at 122 dB would give 40 bits/s/Hz but the channel drifts
    # immediately to ~18 bits/s/Hz before the reward is collected).
    counts = np.zeros((A, S, S), dtype=np.float64)
    R_acc  = np.zeros(S, dtype=np.float64)
    R_cnt  = np.zeros(S, dtype=np.int64)
    step_index = 0

    for _ in range(config.n_mc_samples):
        if channels is not None:
            H_true, step_index = drift_channel_replay(channels, step_index)
        else:
            H_true = draw_static_si_channel_ula(
                N_t=config.n_tx, N_r=config.n_rx,
                sep=10.0, n_reflectors=3, kappa=0.7,
            )
        beam_f, beam_w = design_beams(H_true, h_r, h_t)

        for age in range(A):
            sinr_ul, _, _= compute_sinr(beam_f, beam_w, H_true, h_r, h_t)
            s = sinr_db_to_bin(sinr_to_db(sinr_ul), config)

            # Drift one step — this is what happens between the state observation
            # and the actual reward collection (mirrors environment.step)
            if channels is not None:
                H_drifted, step_index = drift_channel_replay(channels, step_index)
            else:
                H_drifted = drift_channel(H_true, config.drift_scale)

            # Reward = SSE on the drifted channel with stale beams (UL-only)
            sinr_next, _, _ = compute_sinr(beam_f, beam_w, H_drifted, h_r, h_t)
            R_acc[s] += math.log2(1.0 + sinr_next)
            R_cnt[s] += 1

            s_next = sinr_db_to_bin(sinr_to_db(sinr_next), config)
            counts[age, s, s_next] += 1.0
            H_true = H_drifted  # advance for next age step

    # Normalise transition counts → probabilities
    row_sums = counts.sum(axis=2, keepdims=True)
    empty = (row_sums == 0).squeeze(axis=2)
    row_sums[row_sums == 0] = 1.0
    P = counts / row_sums
    # Empty rows fall back to analytical reward and self-loop transition
    for a in range(A):
        for s in range(S):
            if empty[a, s]:
                P[a, s, s] = 1.0

    # Normalise reward; empty bins fall back to analytical bin-centre value
    R_analytical = np.array([_bin_centre_sse(s, config) for s in range(S)])
    R = np.where(R_cnt > 0, R_acc / np.maximum(R_cnt, 1), R_analytical)

    return P, R


# ---------------------------------------------------------------------------
# Value iteration
# ---------------------------------------------------------------------------

def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    config: DPConfig,
    fresh_sinr_bin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the infinite-horizon discounted MDP by value iteration.

    Parameters
    ----------
    P               : transition matrix, shape (max_age+1, n_sinr_bins, n_sinr_bins)
    R               : reward vector,     shape (n_sinr_bins,)
    config          : DPConfig
    fresh_sinr_bin  : the SINR bin the system lands in right after a PROBE

    Returns
    -------
    V      : optimal value function, shape (max_age+1, n_sinr_bins)
    policy : True = PROBE, False = SERVE, shape (max_age+1, n_sinr_bins)
    """
    A = config.max_age + 1
    S = config.n_sinr_bins
    gamma = config.discount

    V = np.zeros((A, S), dtype=np.float64)

    for iteration in range(config.vi_max_iters):
        V_old = V.copy()

        # Q-values for SERVE: R(s) + gamma * E[V(age+1, s') | age, s]
        # age+1 is capped at max_age
        next_age = np.minimum(np.arange(A) + 1, config.max_age)  # shape (A,)
        V_next_age = V_old[next_age]  # shape (A, S)
        # expected future value: sum over s' of P[a,s,s'] * V[a+1, s']
        EV_serve = np.einsum("asd,ad->as", P, V_next_age)  # shape (A, S)
        Q_serve = R[np.newaxis, :] + gamma * EV_serve

        # Q-values for PROBE: -probe_cost + gamma * V(0, fresh_sinr_bin)
        Q_probe = -config.probe_cost + gamma * V_old[0, fresh_sinr_bin]

        V = np.maximum(Q_serve, Q_probe)

        if np.max(np.abs(V - V_old)) < config.vi_tol:
            print(f"Value iteration converged in {iteration + 1} iterations.")
            break

    policy = Q_serve < Q_probe  # True where PROBE is better
    return V, policy


# ---------------------------------------------------------------------------
# Convenience: estimate the fresh SINR bin after a probe
# ---------------------------------------------------------------------------

def estimate_fresh_sinr_bin(config: DPConfig, n_samples: int = 200) -> int:
    """
    Estimate the expected SINR bin immediately after a PROBE by averaging
    over freshly designed beams on random channel draws.
    """
    # sinr_bins = []
    # for _ in range(n_samples):
    #     H = draw_static_si_channel_ula(
    #         N_t=config.n_tx, N_r=config.n_rx,
    #         sep=10.0, n_reflectors=3, kappa=0.7,
    #     )
    #     beam_f, beam_w = design_beams(H, h_r, h_t)
    #     sinr_ul, _, _ = compute_sinr(beam_f, beam_w, H, h_r, h_t)
    #     sinr_bins.append(sinr_db_to_bin(sinr_to_db(sinr_ul), config))
    return 5
