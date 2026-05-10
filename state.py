from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BeamformingState:
    """
    Snapshot of the system at a single discrete timeslot.

    Fields owned by this module (DP side):
        channel_age, reflector_positions, position_uncertainty

    Fields populated by partner's channel/beamforming code:
        sinr_ul, sinr_dl, beam_f, beam_w, H_SI_estimate
    """

    # --- Channel quality (filled by partner) ---
    sinr_ul: float
    """Uplink SINR estimate at the current timeslot (linear scale)."""

    sinr_dl: float
    """Downlink SNR estimate at the current timeslot (linear scale)."""

    # --- Self-interference channel estimate (filled by partner) ---
    H_SI_estimate: np.ndarray
    """Current estimate of the self-interference channel matrix H_SI."""

    # --- Current beam vectors (filled by partner) ---
    beam_f: np.ndarray
    """Transmit (downlink) beamforming vector f."""

    beam_w: np.ndarray
    """Receive (uplink) combining vector w."""

    # --- Reflector dynamics (owned by DP side) ---
    reflector_positions: np.ndarray
    """
    Shape (N, 3). Each row is [R_n, phi_n, theta_n] — the spherical
    coordinates of reflector n relative to the base station.
    """

    position_uncertainty: np.ndarray
    """
    Shape (N, 3). Per-reflector, per-coordinate variance.
    Grows each timestep due to unknown noise v; collapses when probing.
    """

    # --- Probing history (owned by DP side) ---
    channel_age: int = 0
    """Number of consecutive SERVE timeslots since the last PROBE."""

    # --- Timestep ---
    timestep: int = 0
    """Absolute discrete timestep index."""

    def copy(self) -> BeamformingState:
        """Return a deep copy suitable for tree expansion."""
        return BeamformingState(
            sinr_ul=self.sinr_ul,
            sinr_dl=self.sinr_dl,
            H_SI_estimate=self.H_SI_estimate.copy(),
            beam_f=self.beam_f.copy(),
            beam_w=self.beam_w.copy(),
            reflector_positions=self.reflector_positions.copy(),
            position_uncertainty=self.position_uncertainty.copy(),
            channel_age=self.channel_age,
            timestep=self.timestep,
        )
