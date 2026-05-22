from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod

from action import Action, ActionType, all_actions
from config import MCTSConfig
from state import BeamformingState


class RolloutPolicy(ABC):
    """
    Abstract base class for MCTS rollout (simulation) policies.

    A rollout policy selects actions during the simulation phase of MCTS,
    after the tree expansion step. The policy does not need to be optimal —
    it only needs to produce informative value estimates efficiently.
    """

    @abstractmethod
    def select_action(self, state: BeamformingState) -> Action:
        """
        Choose an action for the current rollout step.

        Args:
            state: Current state during the rollout simulation.

        Returns:
            The action to take at this step.
        """
        raise NotImplementedError


class RandomRolloutPolicy(RolloutPolicy):
    """
    Uniform random policy over the full action space.

    Selects SERVE or a random PROBE beam with equal probability.
    Use as a baseline or when no domain knowledge is available.
    """

    def __init__(self, config: MCTSConfig) -> None:
        self.config = config
        self._actions = all_actions(config.n_probe_beams)

    def select_action(self, state: BeamformingState) -> Action:
        """
        Uniformly sample one action from the full action space.

        Args:
            state: Ignored for this policy.

        Returns:
            A randomly selected Action.
        """
        return random.choice(self._actions)


class ThresholdRolloutPolicy(RolloutPolicy):
    """
    Domain-informed heuristic: probe when the channel estimate is stale
    or SINR is below a threshold; otherwise serve.

    Probe beam is chosen by cycling through codebook indices to ensure
    the full codebook is used across multiple probing decisions.

    This policy captures the core trade-off:
        - Probing too often wastes data transmission opportunities (reward = 0).
        - Serving with a stale channel estimate degrades SINR and hence SSE.
    """

    def __init__(self, config: MCTSConfig) -> None:
        self.config = config
        self._probe_beam_counter: int = 0

    def select_action(self, state: BeamformingState) -> Action:
        """
        Probe if the channel is stale or SINR is low; serve otherwise.

        Staleness condition:  state.channel_age >= config.max_channel_age
        Low-SINR condition:   min(sinr_ul, sinr_dl) in dB < sinr_threshold_db

        Args:
            state: Current state.

        Returns:
            Action(SERVE) or Action(PROBE, beam_index).
        """
        if state.channel_age >= self.config.max_channel_age:
            return Action(ActionType.PROBE, self._next_probe_beam())
        sinr_min_db = self._linear_to_db(min(state.sinr_ul, state.sinr_dl))
        if sinr_min_db < self.config.sinr_threshold_db:
            return Action(ActionType.PROBE, self._next_probe_beam())
        return Action(ActionType.SERVE)

    def _next_probe_beam(self) -> int:
        """
        Return the next probe beam index, cycling through the codebook.

        Returns:
            Probe beam index in [0, n_probe_beams - 1].
        """
        self._probe_beam_counter = (self._probe_beam_counter + 1) % self.config.n_probe_beams
        return self._probe_beam_counter

    @staticmethod
    def _linear_to_db(sinr_linear: float) -> float:
        """Convert a linear SINR value to dB."""
        return 10.0 * math.log10(max(float(sinr_linear), 1e-30))


class AdaptiveRolloutPolicy(RolloutPolicy):
    """
    Extension placeholder: a rollout policy that adapts its probe threshold
    based on the rate of uncertainty growth (position_uncertainty).

    If reflectors are moving quickly (high process noise), probe more
    aggressively. If the channel is stable, defer probing.

    Implement this after ThresholdRolloutPolicy is working.
    """

    def __init__(self, config: MCTSConfig) -> None:
        self.config = config

    def select_action(self, state: BeamformingState) -> Action:
        """
        Choose SERVE or PROBE based on estimated uncertainty growth rate.

        Args:
            state: Current state.

        Returns:
            Selected Action.
        """
        raise NotImplementedError
