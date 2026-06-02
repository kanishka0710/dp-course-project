from __future__ import annotations

import math

import numpy as np

from action import Action
from config import DPConfig
from state import State


_NOISE_FLOOR = 1e-9
_UL_DESIRED_POWER = 1.0
_DL_DESIRED_POWER = 1.0


def drift_channel(H: np.ndarray, scale: float) -> np.ndarray:
    """
    Advance the true channel by one timestep using a random-walk model.
    Adds complex Gaussian noise with per-element std = scale * ||H||_F / sqrt(n).
    """
    H = np.asarray(H, dtype=complex)
    noise = scale * (
        np.random.standard_normal(H.shape) + 1j * np.random.standard_normal(H.shape)
    ) / math.sqrt(2)
    return H + noise


def design_beams(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a SI-suppressing beam pair from a channel estimate H.

    Transmit beam f: weakest right singular vector of H (minimises SI power).
    Receive beam w : null-space of H @ f (zeroes residual SI at the combiner).
    """
    H = np.asarray(H, dtype=complex)
    _, _, vh = np.linalg.svd(H, full_matrices=True)
    f = np.conj(vh[-1, :])
    f = f / np.linalg.norm(f)

    # find a unit vector orthogonal to H @ f
    Hf = H @ f
    Hf_norm = np.linalg.norm(Hf)
    if Hf_norm < 1e-15:
        w = np.zeros(H.shape[0], dtype=complex)
        w[0] = 1.0
        return f, w
    Hf = Hf / Hf_norm
    for j in range(H.shape[0]):
        e = np.zeros(H.shape[0], dtype=complex)
        e[j] = 1.0
        v = e - Hf * np.vdot(Hf, e)
        if np.linalg.norm(v) > 1e-9:
            w = v / np.linalg.norm(v)
            return f, w

    w = np.zeros(H.shape[0], dtype=complex)
    w[0] = 1.0
    return f, w


def compute_sinr(
    beam_f: np.ndarray, beam_w: np.ndarray, H: np.ndarray
) -> tuple[float, float]:
    """Return (SINR_UL, SNR_DL) for the given beam pair and true channel."""
    f = np.asarray(beam_f, dtype=complex).reshape(-1)
    w = np.asarray(beam_w, dtype=complex).reshape(-1)
    si_power = abs(np.vdot(w, H @ f)) ** 2
    sinr_ul = _UL_DESIRED_POWER / (si_power + _NOISE_FLOOR)
    sinr_dl = _DL_DESIRED_POWER / _NOISE_FLOOR
    return float(sinr_ul), float(sinr_dl), float(si_power)


def compute_sse(beam_f: np.ndarray, beam_w: np.ndarray, H: np.ndarray) -> float:
    """
    Uplink spectral efficiency (bits/s/Hz): log2(1 + SINR_UL).

    Only the UL SINR is affected by beam quality (SI suppression).  The DL SNR
    is constant for all states, so it cancels in the DP Bellman comparison and
    is omitted here to keep the reward numerically focused on the decision.
    """
    sinr_ul, _, _ = compute_sinr(beam_f, beam_w, H)
    return math.log2(1.0 + sinr_ul)


def step(
    state: State,
    action: Action,
    true_H: np.ndarray,
    config: DPConfig,
) -> tuple[State, float, np.ndarray]:
    """
    Apply one action and return (next_state, reward, new_true_H).

    SERVE: keep current (stale) beams, measure SINR against the drifted channel,
           increment channel_age, collect SSE reward.
    PROBE: measure the current true channel, re-design beams, reset channel_age
           to 0, collect no reward (minus probe_cost).

    The true channel is always drifted by one step regardless of action,
    simulating reflector motion between timeslots.
    """
    # The true channel drifts every timeslot
    new_true_H = drift_channel(true_H, config.drift_scale)

    if action == Action.SERVE:
        sinr_ul, _, _ = compute_sinr(state.beam_f, state.beam_w, new_true_H)
        reward = compute_sse(state.beam_f, state.beam_w, new_true_H)
        next_state = State(
            channel_age=min(state.channel_age + 1, config.max_age),
            sinr_ul=sinr_ul,
            H_SI_estimate=state.H_SI_estimate,
            beam_f=state.beam_f,
            beam_w=state.beam_w,
        )
    else:  # PROBE
        # Treat the current true channel as the new estimate (noiseless probe for simplicity)
        H_new = new_true_H.copy()
        beam_f, beam_w = design_beams(H_new)
        sinr_ul, _, _ = compute_sinr(beam_f, beam_w, new_true_H)
        reward = -config.probe_cost
        next_state = State(
            channel_age=0,
            sinr_ul=sinr_ul,
            H_SI_estimate=H_new,
            beam_f=beam_f,
            beam_w=beam_w,
        )

    return next_state, reward, new_true_H
