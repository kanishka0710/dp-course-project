from __future__ import annotations

import math
from typing import Callable

import numpy as np

from action import Action, ActionType
from array_utils import get_dft_codebook_ula
from belief_state import BeliefState
from config import MCTSConfig
from state import BeamformingState


# --- Physical / numerical constants (toy full-duplex model, see main_channel_test.py) ---
_NOISE_FLOOR = 1e-9
_UL_DESIRED_POWER = 1000.0 # Power in linear scale
_DL_DESIRED_POWER = 1000.0 # Power in linear scale
_PROBE_H_FUSION = 0.35  # weight of fresh sounding matrix in H_SI_estimate update


def _unit_vector_orthogonal_to(h: np.ndarray) -> np.ndarray:
    """Return a deterministic unit-norm vector w with h^H w = 0 (column vectors)."""
    h = np.asarray(h, dtype=complex).reshape(-1)
    n = h.size
    h_norm = np.linalg.norm(h)
    if h_norm < 1e-15:
        w = np.zeros(n, dtype=complex)
        w[0] = 1.0
        return w
    h = h / h_norm
    for j in range(n):
        e = np.zeros(n, dtype=complex)
        e[j] = 1.0
        v = e - h * np.vdot(h, e)
        vn = np.linalg.norm(v)
        if vn > 1e-9:
            return v / vn
    w = np.zeros(n, dtype=complex)
    w[0] = 1.0
    return w


def compute_sinr(
    beam_f: np.ndarray,
    beam_w: np.ndarray,
    H_SI: np.ndarray,
) -> tuple[float, float]:
    """
    Compute (SINR_UL, SNR_DL) given beams and the full self-interference channel.

    Uses the same SI power convention as ``main_channel_test.py``:
    P_SI = |w^H H_SI f|^2. Uplink is interference-limited by SI; downlink SNR
    uses a clean desired link (SI not in the DL denominator), matching the
    toy sum-rate plot there.
    """
    f = np.asarray(beam_f, dtype=complex).reshape(-1)
    w = np.asarray(beam_w, dtype=complex).reshape(-1)
    si_power = abs(np.vdot(w, H_SI @ f)) ** 2
    sinr_ul = _UL_DESIRED_POWER / (si_power + _NOISE_FLOOR)
    sinr_dl = _DL_DESIRED_POWER / _NOISE_FLOOR
    return float(sinr_ul), float(sinr_dl)


def design_beams(H_SI_estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Self-interference-aware beam pair: transmit in the weakest singular direction
    of H_SI, receive with a combiner orthogonal to H_SI f (numeric nullspace).
    """
    H = np.asarray(H_SI_estimate, dtype=complex)
    _, _, vh = np.linalg.svd(H, full_matrices=True)
    f = np.conjugate(vh[-1, :])
    fn = np.linalg.norm(f)
    if fn < 1e-15:
        f = np.ones(H.shape[1], dtype=complex)
        f = f / np.linalg.norm(f)
    else:
        f = f / fn
    w = _unit_vector_orthogonal_to(H @ f)
    return f, w


def apply_probe_beam(
    probe_beam_index: int,
    H_SI: np.ndarray,
) -> np.ndarray:
    """
    Sound the channel with a DFT codebook beam and return a noisy full-matrix
    snapshot of H_SI (AWGN), for use in Kalman-style channel tracking.
    """
    H = np.asarray(H_SI, dtype=complex)
    n_rx, n_tx = H.shape
    B = max(n_tx, probe_beam_index + 1)
    codebook = get_dft_codebook_ula(n_tx, B)
    row = int(probe_beam_index) % B
    f_probe = np.conjugate(codebook[row])
    assert f_probe.shape == (n_tx,)
    scale = 0.05 * (np.linalg.norm(H, "fro") / math.sqrt(max(n_rx * n_tx, 1)) + 1e-12)
    noise = scale * (
        np.random.standard_normal(H.shape)
        + 1j * np.random.standard_normal(H.shape)
    )
    return H + noise


class BeamformingEnvironment:
    """
    Simulates one discrete timeslot of the serve/probe decision problem.

    This class owns the transition logic (reflector dynamics, uncertainty
    propagation, channel age tracking) and delegates SINR computation and
    beam design to functions built on the partner ``array_utils`` / channel
    conventions (injectable for testing).
    """

    def __init__(
        self,
        config: MCTSConfig,
        belief: BeliefState,
        known_velocity: np.ndarray,
        *,
        compute_sinr_fn: Callable = compute_sinr,
        design_beams_fn: Callable = design_beams,
        apply_probe_fn: Callable = apply_probe_beam,
    ) -> None:
        self.config = config
        self.belief = belief
        self.known_velocity = known_velocity
        self._compute_sinr = compute_sinr_fn
        self._design_beams = design_beams_fn
        self._apply_probe = apply_probe_fn

    def step(
        self,
        state: BeamformingState,
        action: Action,
        true_H_SI: np.ndarray,
    ) -> tuple[BeamformingState, float]:
        predicted = self.belief.predict(state, self.known_velocity)

        if action.type == ActionType.SERVE:
            sinr_ul, sinr_dl = self._compute_sinr(
                predicted.beam_f, predicted.beam_w, true_H_SI
            )
            next_state = BeamformingState(
                sinr_ul=sinr_ul,
                sinr_dl=sinr_dl,
                H_SI_estimate=predicted.H_SI_estimate,
                beam_f=predicted.beam_f,
                beam_w=predicted.beam_w,
                reflector_positions=predicted.reflector_positions,
                position_uncertainty=predicted.position_uncertainty,
                channel_age=state.channel_age + 1,
                timestep=predicted.timestep,
            )
            reward = self.compute_sse(next_state, true_H_SI)
            return next_state, reward

        assert action.probe_beam_index is not None
        H_meas = self._apply_probe(action.probe_beam_index, true_H_SI)
        H_new = (1.0 - _PROBE_H_FUSION) * predicted.H_SI_estimate + _PROBE_H_FUSION * H_meas
        beam_f, beam_w = self._design_beams(H_new)
        sinr_ul, sinr_dl = self._compute_sinr(beam_f, beam_w, true_H_SI)
        sig_new = predicted.position_uncertainty * self.config.probe_uncertainty_reduction
        next_state = BeamformingState(
            sinr_ul=sinr_ul,
            sinr_dl=sinr_dl,
            H_SI_estimate=H_new,
            beam_f=beam_f,
            beam_w=beam_w,
            reflector_positions=predicted.reflector_positions,
            position_uncertainty=sig_new,
            channel_age=0,
            timestep=predicted.timestep,
        )
        return next_state, 0.0

    def compute_sse(self, state: BeamformingState, true_H_SI: np.ndarray) -> float:
        sinr_ul, sinr_dl = self._compute_sinr(state.beam_f, state.beam_w, true_H_SI)
        return math.log2(1.0 + sinr_ul) + math.log2(1.0 + sinr_dl)

    def propagate_reflectors(self, state: BeamformingState) -> BeamformingState:
        return self.belief.predict(state, self.known_velocity)

    def advance_true_channel(self, true_H_SI: np.ndarray) -> np.ndarray:
        """Slow fading on the simulator copy of H_SI used inside MCTS rollouts."""
        H = np.asarray(true_H_SI, dtype=complex)
        scale = 0.01 * float(np.mean(np.abs(H)) + 1e-12)
        noise = scale * (
            np.random.standard_normal(H.shape)
            + 1j * np.random.standard_normal(H.shape)
        )
        return H + noise

    def get_all_actions(self) -> list[Action]:
        from action import all_actions

        return all_actions(self.config.n_probe_beams)
