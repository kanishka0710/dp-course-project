from enum import Enum, auto


class Action(Enum):
    SERVE = auto()
    """Transmit using current (possibly stale) beams. Collects SSE reward."""

    PROBE = auto()
    """Sound the channel to refresh H_SI estimate. Zero reward this timeslot."""
