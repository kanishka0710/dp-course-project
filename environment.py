from __future__ import annotations

import math
from typing import Callable

import numpy as np

from action import Action, ActionType
from belief_state import BeliefState
from config import MCTSConfig
from state import BeamformingState


# ---------------------------------------------------------------------------
# Partner interface — replace these stubs with imports from your partner's
# module once those are available.
# ---------------------------------------------------------------------------

def compute_sinr(
    beam_f: np.ndarray,
    beam_w: np.ndarray,
    H_SI: np.ndarray,
) -> tuple[float, float]:
    """
    [PARTNER CODE] Compute (SINR_UL, SNR_DL) given beams and the full
    self-interference channel.

    Args:
        beam_f: Transmit beamforming vector.
        beam_w: Receive combining vector.
        H_SI:   Full self-interference channel matrix H_SI.

    Returns:
        (sinr_ul, sinr_dl) in linear scale.
    """
    raise NotImplementedError("Implement with partner's channel code")


def design_beams(H_SI_estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    [PARTNER CODE] Compute optimal beams (f, w) given an H_SI estimate.

    Args:
        H_SI_estimate: Current estimate of the self-interference channel.

    Returns:
        (beam_f, beam_w) as numpy arrays.
    """
    raise NotImplementedError("Implement with partner's beamforming code")


def apply_probe_beam(
    probe_beam_index: int,
    H_SI: np.ndarray,
) -> np.ndarray:
    """
    [PARTNER CODE] Transmit a probing beam and return an H_SI measurement.

    Args:
        probe_beam_index: Index into the estimation codebook.
        H_SI:             True channel used by the simulator.

    Returns:
        Noisy measurement of H_SI.
    """
    raise NotImplementedError("Implement with partner's channel estimation code")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BeamformingEnvironment:
    """
    Simulates one discrete timeslot of the serve/probe decision problem.

    This class owns the transition logic (reflector dynamics, uncertainty
    propagation, channel age tracking) and delegates SINR computation and
    beam design to partner-supplied functions.

    The true channel is maintained internally for simulation purposes and
    is never exposed to the MCTS planner — only the state (which contains
    estimates and uncertainties) is visible to the tree.
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
        """
        Args:
            config:           Hyperparameters.
            belief:           BeliefState instance for uncertainty updates.
            known_velocity:   Shape (N, 3). Known drift V per reflector per
                              coordinate, used in the predict step each timestep.
            compute_sinr_fn:  Partner's SINR function (injectable for testing).
            design_beams_fn:  Partner's beam design function (injectable).
            apply_probe_fn:   Partner's probe measurement function (injectable).
        """
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
        """
        Apply an action and return the next state and immediate reward.

        Transition logic:
            SERVE:
                1. Predict reflector positions (apply known V, inflate variance).
                2. Increment channel_age.
                3. Compute SSE with current beams against the true channel.
                4. Reward = SSE.

            PROBE:
                1. Predict reflector positions.
                2. Apply probe beam to get an H_SI measurement.
                3. Update belief (reduce uncertainty, shift mean).
                4. Re-design beams with the refreshed estimate.
                5. Reset channel_age to 0.
                6. Reward = 0 (data cannot be sent while probing).

        Args:
            state:      Current system state (not mutated).
            action:     Action to execute.
            true_H_SI:  True self-interference channel (simulator-only,
                        not visible to the planner).

        Returns:
            (next_state, reward)
        """
        raise NotImplementedError

    def compute_sse(self, state: BeamformingState, true_H_SI: np.ndarray) -> float:
        """
        Compute the sum spectral efficiency (SSE) for Eq. (1).

            SSE = log2(1 + SINR_UL) + log2(1 + SNR_DL)

        Args:
            state:      Current state (beams taken from state.beam_f / beam_w).
            true_H_SI:  True channel used to evaluate actual SINR.

        Returns:
            SSE in bits/s/Hz.
        """
        raise NotImplementedError

    def propagate_reflectors(self, state: BeamformingState) -> BeamformingState:
        """
        Advance reflector positions by one timestep using the predict step
        of the belief model.

        Applies known velocity V and inflates position_uncertainty by
        config.process_noise_variance. Does not mutate the input state.

        Args:
            state: Current state.

        Returns:
            New state with updated reflector_positions and position_uncertainty.
        """
        raise NotImplementedError

    def get_all_actions(self) -> list[Action]:
        """
        Return the complete action space for the current configuration.

        Returns:
            List of Action objects (1 SERVE + n_probe_beams PROBE actions).
        """
        from action import all_actions
        return all_actions(self.config.n_probe_beams)
