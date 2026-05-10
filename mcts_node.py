from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from action import Action
    from state import BeamformingState


class MCTSNode:
    """
    A single node in the Monte Carlo search tree.

    Each node represents a state reached by taking action_taken from its
    parent. The root node has parent=None and action_taken=None.

    Statistics tracked per node:
        visit_count  N(s)   — number of times this node has been visited
        total_value  W(s)   — sum of all rollout returns passing through here
        mean_value   Q(s)   — W(s) / N(s), the empirical value estimate
    """

    def __init__(
        self,
        state: BeamformingState,
        parent: MCTSNode | None = None,
        action_taken: Action | None = None,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children: list[MCTSNode] = []
        self.untried_actions: list[Action] | None = None  # None = not yet initialised

        self.visit_count: int = 0
        self.total_value: float = 0.0

    # ------------------------------------------------------------------
    # Value estimates
    # ------------------------------------------------------------------

    @property
    def mean_value(self) -> float:
        """Q(s) = W(s) / N(s). Returns 0 if never visited."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    # ------------------------------------------------------------------
    # UCT score
    # ------------------------------------------------------------------

    def ucb_score(self, exploration_c: float) -> float:
        """
        Upper Confidence Bound applied to Trees (UCT):

            UCB(s) = Q(s) + c * sqrt( ln(N_parent) / N(s) )

        An unvisited node returns +infinity to guarantee it is selected
        at least once before any visited sibling.

        Args:
            exploration_c: Exploration constant (typically sqrt(2)).

        Returns:
            UCB score as a float.
        """
        if self.visit_count == 0:
            return math.inf
        assert self.parent is not None, "ucb_score called on root node"
        return self.mean_value + exploration_c * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )

    # ------------------------------------------------------------------
    # Tree navigation helpers
    # ------------------------------------------------------------------

    def is_leaf(self) -> bool:
        """True if this node has no expanded children."""
        return len(self.children) == 0

    def is_fully_expanded(self) -> bool:
        """True if every possible action has a corresponding child node."""
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def best_child(self, exploration_c: float) -> MCTSNode:
        """
        Return the child with the highest UCB score.

        Args:
            exploration_c: Exploration constant forwarded to ucb_score.

        Returns:
            Child MCTSNode with maximum UCB score.
        """
        assert self.children, "best_child called on a leaf node"
        return max(self.children, key=lambda child: child.ucb_score(exploration_c))

    def best_action_child(self) -> MCTSNode:
        """
        Return the child with the highest mean value (no exploration bonus).
        Used at the root after the simulation budget is exhausted to pick
        the action to actually execute.

        Returns:
            Child with maximum mean_value.
        """
        assert self.children, "best_action_child called on a leaf node"
        return max(self.children, key=lambda child: child.mean_value)

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def update(self, value: float) -> None:
        """
        Accumulate a rollout return into this node's statistics.

        Args:
            value: Discounted cumulative reward from the rollout.
        """
        self.visit_count += 1
        self.total_value += value

    def __repr__(self) -> str:
        return (
            f"MCTSNode(action={self.action_taken}, "
            f"N={self.visit_count}, Q={self.mean_value:.4f})"
        )
