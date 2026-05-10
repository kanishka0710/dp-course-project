from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class ActionType(Enum):
    SERVE = auto()
    """Use current beam estimates to serve both UL and DL users. Reward = SSE."""

    PROBE = auto()
    """Transmit a probing beam to update H_SI estimate. Reward = 0."""


@dataclass(frozen=True)
class Action:
    """
    An action taken by the base station at a single timeslot.

    For SERVE actions, probe_beam_index must be None.
    For PROBE actions, probe_beam_index selects a beam from the
    estimation codebook (0-indexed, range [0, n_probe_beams - 1]).
    """

    type: ActionType
    probe_beam_index: int | None = None

    def __post_init__(self) -> None:
        if self.type == ActionType.SERVE and self.probe_beam_index is not None:
            raise ValueError("SERVE actions must have probe_beam_index=None")
        if self.type == ActionType.PROBE and self.probe_beam_index is None:
            raise ValueError("PROBE actions must specify a probe_beam_index")

    def __repr__(self) -> str:
        if self.type == ActionType.SERVE:
            return "Action(SERVE)"
        return f"Action(PROBE, beam={self.probe_beam_index})"


def all_actions(n_probe_beams: int) -> list[Action]:
    """
    Return the full action space: one SERVE action plus one PROBE action
    per codebook beam.

    Args:
        n_probe_beams: Number of beams in the probing codebook.

    Returns:
        List of all valid Action objects.
    """
    actions: list[Action] = [Action(type=ActionType.SERVE)]
    for i in range(n_probe_beams):
        actions.append(Action(type=ActionType.PROBE, probe_beam_index=i))
    return actions
