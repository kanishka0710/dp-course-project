from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    """
    Snapshot of the system at a single discrete timeslot.

    channel_age : how many consecutive SERVE steps have elapsed since the last PROBE.
                  Resets to 0 after every PROBE.
    sinr_ul     : uplink SINR (linear) observed this timeslot.  Used by the DP
                  as the second state dimension to detect channel degradation.
    H_SI_estimate : the channel matrix estimated at the last PROBE, used to
                    design the current beam pair.  Stays fixed until next PROBE.
    beam_f, beam_w : transmit / receive beam vectors derived from H_SI_estimate.
    """

    channel_age: int
    sinr_ul: float
    H_SI_estimate: np.ndarray
    beam_f: np.ndarray
    beam_w: np.ndarray
