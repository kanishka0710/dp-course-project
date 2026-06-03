from __future__ import annotations

import math

import numpy as np

from action import Action
from config import DPConfig
from state import State

from array_utils import *
from channel import *
from scipy.linalg import null_space


SI_GAIN = 1e5
UL_GAIN = 1
DL_GAIN = 1
NOISE_POWER = 1e-1

def drift_channel_replay(
    channels: list[np.ndarray], step_index: int
) -> tuple[np.ndarray, int]:
    """
    Return the next channel from a pre-recorded sequence, wrapping around.

    Returns (H, next_step_index) so the caller can track position.
    """
    idx = step_index % len(channels)
    return np.asarray(channels[idx], dtype=complex), (idx + 1) % len(channels)


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


def design_beams(H, h_t, h_r) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a SI-suppressing beam pair from a channel estimate H.

    Transmit beam f: weakest right singular vector of H (minimises SI power).
    Receive beam w : null-space of H @ f (zeroes residual SI at the combiner).
    """

    f = h_t.conj()
    w = h_r.conj()

    f = f / np.linalg.norm(f)
    w = w / np.linalg.norm(w)

    h_t_eff = w @ H

    B = null_space(h_t_eff.reshape(1,-1))
    P = B @ np.linalg.pinv(B.conj().T @ B) @ B.conj().T

    f_bfc = P @ f
    
    return f_bfc, w


def compute_sinr(
    beam_f: np.ndarray, beam_w: np.ndarray, H: np.ndarray, h_t, h_r
) -> tuple[float, float]:
    """Return (SINR_UL, SNR_DL) for the given beam pair and true channel."""
    f = np.asarray(beam_f, dtype=complex).reshape(-1)
    w = np.asarray(beam_w, dtype=complex).reshape(-1)
    si_power = SI_GAIN * np.abs(w @ H @ f)**2

    ul_power = UL_GAIN * np.abs(h_r @ w)**2
    dl_power = DL_GAIN * np.abs(h_t @ f)**2

    sinr_ul = ul_power / (si_power + NOISE_POWER)
    snr_dl = dl_power / NOISE_POWER
    return float(sinr_ul), float(snr_dl), float(si_power)


def compute_sse(sinr_ul, snr_dl) -> float:
    """
    Uplink spectral efficiency (bits/s/Hz): log2(1 + SINR_UL).

    Only the UL SINR is affected by beam quality (SI suppression).  The DL SNR
    is constant for all states, so it cancels in the DP Bellman comparison and
    is omitted here to keep the reward numerically focused on the decision.
    """
    

    return math.log2(1.0 + sinr_ul) #+ math.log2(1.0 + snr_dl)


def step(
    state: State,
    action: Action,
    true_H: np.ndarray,
    config: DPConfig,
    h_t,
    h_r,
    channels: list[np.ndarray] | None = None,
    step_index: int = 0,
) -> tuple[State, float, np.ndarray, int]:
    """
    Apply one action and return (next_state, reward, new_true_H, next_step_index).

    SERVE: keep current (stale) beams, measure SINR against the drifted channel,
           increment channel_age, collect SSE reward.
    PROBE: measure the current true channel, re-design beams, reset channel_age
           to 0, collect no reward (minus probe_cost).

    If channels is provided the channel is drawn from that sequence (replay mode);
    otherwise a random-walk drift step is applied to true_H.
    next_step_index is always returned (0 when not using replay).
    """
    if channels is not None:
        new_true_H, next_step_index = drift_channel_replay(channels, step_index)
    else:
        new_true_H = drift_channel(true_H, config.drift_scale)
        next_step_index = 0

    if action == Action.SERVE:
        print(f"serve {step_index}")
        sinr_ul, snr_dl, _ = compute_sinr(state.beam_f, state.beam_w, new_true_H, h_t, h_r)
        reward = compute_sse(sinr_ul, snr_dl)
        next_state = State(
            channel_age=min(state.channel_age + 1, config.max_age),
            sinr_ul=sinr_ul,
            H_SI_estimate=state.H_SI_estimate,
            beam_f=state.beam_f,
            beam_w=state.beam_w,
        )
    else:  # PROBE
        # Treat the current true channel as the new estimate (noiseless probe for simplicity)
        print(f"probe {step_index}")

        H_new = new_true_H.copy()
        beam_f, beam_w = design_beams(H_new, h_t, h_r)
        sinr_ul, _, _ = compute_sinr(beam_f, beam_w, new_true_H, h_t, h_r)
        reward = -config.probe_cost
        next_state = State(
            channel_age=0,
            sinr_ul=sinr_ul,
            H_SI_estimate=H_new,
            beam_f=beam_f,
            beam_w=beam_w,
        )

    return next_state, reward, new_true_H, next_step_index
