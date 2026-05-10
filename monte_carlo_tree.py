from __future__ import annotations

import random

import numpy as np

from action import Action, all_actions
from config import MCTSConfig
from environment import BeamformingEnvironment
from mcts_node import MCTSNode
from rollout_policy import RolloutPolicy
from state import BeamformingState


class MonteCarloTreeSearch:
    """
    Monte Carlo Tree Search for the beam serve/probe scheduling problem.

    At each real timeslot the planner is called with the current observed
    state. It runs n_simulations of the four-phase MCTS loop, then returns
    the action with the highest empirical value estimate at the root.

    The four phases:
        1. Select   — walk the tree from the root using UCT until a leaf.
        2. Expand   — add one unexplored child to the selected leaf.
        3. Rollout  — simulate forward from the new child using the rollout
                      policy for rollout_depth steps; collect discounted return.
        4. Backprop — propagate the return up to the root, updating N and Q.
    """

    def __init__(
        self,
        env: BeamformingEnvironment,
        rollout_policy: RolloutPolicy,
        config: MCTSConfig,
    ) -> None:
        """
        Args:
            env:            Environment that simulates transitions and rewards.
            rollout_policy: Policy used in the simulation phase.
            config:         Hyperparameters (exploration_c, rollout_depth, etc.).
        """
        self.env = env
        self.rollout_policy = rollout_policy
        self.config = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        root_state: BeamformingState,
        true_H_SI: np.ndarray,
    ) -> Action:
        """
        Run the full MCTS budget and return the best action for root_state.

        Builds a fresh tree rooted at root_state, runs config.n_simulations
        iterations of select → expand → rollout → backpropagate, then
        returns the action leading to the child with the highest mean value.

        Args:
            root_state: The current observed system state.
            true_H_SI:  True channel, passed to env.step during simulation.

        Returns:
            The recommended Action to execute at this timeslot.
        """
        root = MCTSNode(state=root_state)
        root.untried_actions = self.env.get_all_actions()

        for _ in range(self.config.n_simulations):
            node = self._select(root)
            if not node.is_fully_expanded():
                node = self._expand(node)
            value = self._rollout(node.state, true_H_SI)
            self._backpropagate(node, value)

        return root.best_action_child().action_taken  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Phase 1 — Selection
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Walk the tree from node using UCT until reaching a leaf or a node
        that is not yet fully expanded.

        Applies the tree policy: at each internal node, move to the child
        with the highest UCB score.

        Args:
            node: Starting node (typically the root).

        Returns:
            The selected leaf or partially expanded node.
        """
        curNode = node
        while not curNode.is_fully_expanded():
            curNode = curNode.best_child(self.config.exploration_c)
        return curNode


    # ------------------------------------------------------------------
    # Phase 2 — Expansion
    # ------------------------------------------------------------------

    def _expand(self, node: MCTSNode, true_H_SI: np.ndarray) -> MCTSNode:
        """
        Add one new child to node by trying an untried action.

        Pops one action from node.untried_actions, simulates one step in
        the environment to obtain the child state, creates an MCTSNode for
        it, and appends it to node.children.

        Args:
            node: A node with at least one untried action.

        Returns:
            The newly created child MCTSNode.
        """
        action = node.untried_actions.pop(0)
        next_state = self.env.step(node.state, action, true_H_SI)
        child = MCTSNode(state=next_state, parent=node, action_taken=action)
        node.children.append(child)
        return child

    # ------------------------------------------------------------------
    # Phase 3 — Rollout (Simulation)
    # ------------------------------------------------------------------

    def _rollout(self, state: BeamformingState, true_H_SI: np.ndarray) -> float:
        """
        Simulate forward from state for config.rollout_depth steps using
        the rollout policy and return the discounted cumulative reward.

        At each step:
            1. Ask rollout_policy for an action.
            2. Call env.step to get (next_state, reward).
            3. Accumulate discounted reward.
            4. Advance true_H_SI according to the true channel dynamics
               (use env.propagate_reflectors or equivalent).

        The true_H_SI passed here is a simulation copy — it does not affect
        the real environment state.

        Args:
            state:      Starting state for the rollout.
            true_H_SI:  Starting true channel for the rollout simulation.

        Returns:
            Discounted cumulative reward over the rollout horizon.
        """
        for _ in range(self.config.rollout_depth):
            action = self.rollout_policy.select_action(state)
            next_state, reward = self.env.step(state, action, true_H_SI)
            state = next_state
            true_H_SI = self.env.propagate_reflectors(true_H_SI)
            reward = reward * self.config.discount_gamma ** _
            total_reward += reward
        return total_reward

    # ------------------------------------------------------------------
    # Phase 4 — Backpropagation
    # ------------------------------------------------------------------

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Walk from node back to the root, calling node.update(value) at
        each ancestor to accumulate visit counts and total values.

        Args:
            node:  Leaf node where the rollout started.
            value: Discounted return obtained from the rollout.
        """
        curNode = node
        while curNode is not None:
            curNode.update(value)
            curNode = curNode.parent
        return None
