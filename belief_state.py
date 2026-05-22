from __future__ import annotations

import numpy as np

from config import MCTSConfig
from state import BeamformingState


class BeliefState:
    """
    Maintains a Gaussian belief over reflector positions using a
    predict-update (Kalman-filter-style) cycle.

    Predict step (every timestep):
        - Advance mean positions by known velocity V.
        - Inflate variance by process noise (unknown v).

    Update step (on PROBE):
        - Apply Kalman gain to shift mean toward measurement.
        - Reduce variance according to Eqs 6.15-6.17.

    position_uncertainty stores per-element scalar variances, shape (N, 3),
    under the diagonal (independent coordinates) assumption. All operations
    are elementwise — no full covariance matrix is needed.

    This class does not mutate state in place, making it safe inside the
    MCTS tree.
    """

    def __init__(self, config: MCTSConfig) -> None:
        self.config = config

    def predict(
        self,
        state: BeamformingState,
        known_velocity: np.ndarray,
    ) -> BeamformingState:
        """
        Propagate belief forward by one timestep.

        Applies known drift V to each reflector's mean position and
        inflates per-coordinate variance by process_noise_variance.

        Args:
            state: Current state (not mutated).
            known_velocity: Shape (N, 3). Known velocity [V_R, V_phi, V_theta]
                            for each reflector, corresponding to Eqs. (4)-(6).

        Returns:
            New BeamformingState with updated reflector_positions and
            increased position_uncertainty.
        """
        meanNew = state.reflector_positions + known_velocity
        varNew = state.position_uncertainty + self.config.process_noise_variance
        return BeamformingState(
            reflector_positions=meanNew,
            position_uncertainty=varNew,
            channel_age=state.channel_age,
            timestep=state.timestep + 1,
            sinr_ul=state.sinr_ul,
            sinr_dl=state.sinr_dl,
            H_SI_estimate=state.H_SI_estimate,
            beam_f=state.beam_f,
            beam_w=state.beam_w,
        )

    def update(
        self,
        state: BeamformingState,
        probe_measurement: np.ndarray,
    ) -> BeamformingState:
        """
        Collapse belief after a probing measurement (Kalman update step).

        With O_s = I (probe directly measures positions with Gaussian noise),
        Eqs 6.15-6.17 reduce to elementwise operations:

            K       = SigP / (SigP + SigO)             (Kalman gain)
            mu_new  = mu + K * (measurement - mu)      (shift mean)
            sig_new = (1 - K) * SigP                   (reduce variance)

        Args:
            state: Current state (not mutated).
            probe_measurement: Shape (N, 3). Noisy reflector position estimate
                               returned by partner's channel estimation code.

        Returns:
            New BeamformingState with updated mean and reduced uncertainty.
            channel_age is reset to 0.
        """
        SigP = state.position_uncertainty          # (N, 3) per-element variances
        SigO = self.config.measurement_noise_variance  # scalar

        K = SigP / (SigP + SigO)                                        # Eq 6.15
        muNew = state.reflector_positions + K * (probe_measurement - state.reflector_positions)  # Eq 6.16
        sigNew = (1 - K) * SigP                                              # Eq 6.17

        return BeamformingState(
            reflector_positions=muNew,
            position_uncertainty=sigNew,
            channel_age=0,
            timestep=state.timestep,
            sinr_ul=state.sinr_ul,
            sinr_dl=state.sinr_dl,
            H_SI_estimate=state.H_SI_estimate,
            beam_f=state.beam_f,
            beam_w=state.beam_w,
        )

    def sample_channel(self, state: BeamformingState) -> np.ndarray:
        """
        Draw one Monte Carlo sample of H_dynamic from the current belief.

        Samples reflector positions from the Gaussian belief, then passes
        them to partner's array response model to construct H_dynamic.

        Args:
            state: Current state containing reflector position mean and variance.

        Returns:
            H_dynamic sample as a complex numpy array matching the shape of
            state.H_SI_estimate.
        """
        sampled_positions = np.random.normal(
            loc=state.reflector_positions,
            scale=np.sqrt(state.position_uncertainty),
        )  # shape (N, 3)

        from channel import single_reflection_si_ula

        n_rx, n_tx = state.H_SI_estimate.shape
        H_dynamic = np.zeros((n_rx, n_tx), dtype=complex)
        for pos in sampled_positions:
            # pos = [R, phi, theta]; ULA response only needs the azimuth angle theta
            theta = pos[2]
            H_dynamic += single_reflection_si_ula(n_tx, n_rx, theta)

        n = len(sampled_positions)
        if n > 0:
            H_dynamic /= n
            norm = np.linalg.norm(H_dynamic, "fro")
            if norm > 1e-12:
                H_dynamic = np.sqrt(n_rx * n_tx) * H_dynamic / norm

        return H_dynamic
        

    def expected_sinr(
        self,
        state: BeamformingState,
        n_samples: int = 50,
    ) -> tuple[float, float]:
        """
        Estimate expected (SINR_UL, SNR_DL) under the current belief.

        Because the true channel is uncertain, the SINR you will actually
        observe is a random variable. This method averages over n_samples
        channel draws from the belief to get E[SINR] — giving the MCTS
        rollout a more honest value estimate when the channel age is high.

        Args:
            state: Current state.
            n_samples: Number of Monte Carlo samples to average over.

        Returns:
            Tuple (expected_sinr_ul, expected_sinr_dl) in linear scale.
        """
        from environment import compute_sinr  # avoid circular import at module level

        sinr_ul_acc = 0.0
        sinr_dl_acc = 0.0
        for _ in range(n_samples):
            H_sample = self.sample_channel(state)
            sinr_ul, sinr_dl = compute_sinr(state.beam_f, state.beam_w, H_sample)
            sinr_ul_acc += sinr_ul
            sinr_dl_acc += sinr_dl
        return sinr_ul_acc / n_samples, sinr_dl_acc / n_samples
