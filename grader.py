"""
grader.py — Deterministic reward/scoring for the Production Incident Response Simulator.

Score formula (clamped to [0.0, 1.0]):

  A perfect agent (all required actions completed, no extra steps) scores 1.0
  on every task regardless of task length or ordering mode.

  Per-episode breakdown:
    progress_score  = correct_count * (0.80 / N)   # scales with task size
    completion_bonus = 0.20 if all_done else 0.0    # bonus for full resolution
    extra_step_penalty = extra_steps * 0.05
    wrong_action_penalty = wrong_count * 0.12

  final_score = clamp(progress_score + completion_bonus
                      - extra_step_penalty - wrong_action_penalty, 0.0, 1.0)

  Ordering modes (set via task.order_strict):
    order_strict=True  — actions must be taken in the EXACT listed order (easy, hard)
    order_strict=False — any ordering of required actions earns full credit (medium)

  Intermediate (per-step) reward is returned every step for real-time feedback:
    +per_action_bonus   on each correct action
    +0.20               on the final completing action
    -0.12               on each wrong / unnecessary action
    -0.02               per-step overhead (efficiency nudge)
"""

from typing import List, Dict, Any, Set
from tasks import TaskDefinition


class Grader:
    """
    Scores an agent's performance on a single task episode.

    Usage:
        grader = Grader(task)
        reward = grader.record(action_dict)  # called by env.step()
        score  = grader.final_score()        # [0.0, 1.0], authoritative
        done   = grader.is_complete()
    """

    # Per-step overhead — keeps agents efficient even when correct
    STEP_OVERHEAD    = 0.02
    # Penalty for each extra step BEYOND the minimum required
    EXTRA_PENALTY    = 0.05
    # Penalty for wrong or unnecessary actions
    WRONG_PENALTY    = 0.12
    # Bonus unlocked when ALL required actions are completed
    COMPLETION_BONUS = 0.20

    def __init__(self, task: TaskDefinition):
        self.task         = task
        self.expected: List[Dict] = task.expected_action_sequence
        self._n           = len(self.expected)
        self._order_strict = getattr(task, "order_strict", True)

        # Per-action reward scales so a perfect agent earns exactly 1.0
        self._per_action_bonus = 0.80 / max(self._n, 1)

        self.actions_taken: List[Dict] = []
        self.correct_count: int = 0
        self._last_step_reward: float = 0.0

        # For order-agnostic mode: track which required actions are still pending
        self._pending: Set[tuple] = {
            (a["type"], a["target"]) for a in self.expected
        }

    # ------------------------------------------------------------------
    # Called by the environment on every step
    # ------------------------------------------------------------------
    def record(self, action: Dict[str, Any]) -> float:
        """
        Record action and return the immediate step reward.

        In strict mode  → action must match the NEXT expected action in sequence.
        In agnostic mode → action matches if it is ANY remaining required action.
        """
        self.actions_taken.append(action)
        step_reward = -self.STEP_OVERHEAD

        if self.correct_count < self._n:
            if self._order_strict:
                matched = self._matches_strict(action)
            else:
                matched = self._matches_agnostic(action)

            if matched:
                self.correct_count += 1
                step_reward += self._per_action_bonus
                if self.correct_count == self._n:
                    step_reward += self.COMPLETION_BONUS
            else:
                step_reward -= self.WRONG_PENALTY
        else:
            # All required actions already done — unnecessary extra action
            step_reward -= self.WRONG_PENALTY

        self._last_step_reward = round(step_reward, 4)
        return self._last_step_reward

    def step_reward(self) -> float:
        """Reward returned on the most recent step."""
        return self._last_step_reward

    # ------------------------------------------------------------------
    # Final episode score — authoritative
    # ------------------------------------------------------------------
    def final_score(self) -> float:
        """
        Compute the overall episode score in [0.0, 1.0].
        A perfect agent always scores exactly 1.0.
        """
        total        = len(self.actions_taken)
        extra_steps  = max(0, total - self._n)

        score  = self.correct_count * self._per_action_bonus
        if self.correct_count == self._n:
            score += self.COMPLETION_BONUS
        score -= extra_steps * self.EXTRA_PENALTY
        score -= self._count_wrong() * self.WRONG_PENALTY

        return round(max(0.0, min(1.0, score)), 4)

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------
    def is_complete(self) -> bool:
        """True when every required action has been completed."""
        return self.correct_count >= self._n

    def progress(self) -> str:
        pct = int(100 * self.correct_count / max(self._n, 1))
        return f"{self.correct_count}/{self._n} correct actions ({pct}% resolved)"

    def summary(self) -> Dict[str, Any]:
        """Full scoring breakdown — surfaced in StepResult.info."""
        return {
            "final_score":       self.final_score(),
            "correct_actions":   self.correct_count,
            "total_actions":     len(self.actions_taken),
            "required_actions":  self._n,
            "wrong_actions":     self._count_wrong(),
            "extra_steps":       max(0, len(self.actions_taken) - self._n),
            "order_strict":      self._order_strict,
            "progress":          self.progress(),
            "is_complete":       self.is_complete(),
            "per_action_bonus":  round(self._per_action_bonus, 4),
            "completion_bonus":  self.COMPLETION_BONUS if self.is_complete() else 0.0,
        }

    # ------------------------------------------------------------------
    # Matching helpers
    # ------------------------------------------------------------------
    def _matches_strict(self, action: Dict) -> bool:
        """
        Strict mode: action must match the NEXT expected action in sequence.
        Used by easy and hard tasks where dependency ordering is mandatory.
        """
        if self.correct_count >= self._n:
            return False
        expected = self.expected[self.correct_count]
        return (
            action.get("type")   == expected.get("type") and
            action.get("target") == expected.get("target")
        )

    def _matches_agnostic(self, action: Dict) -> bool:
        """
        Order-agnostic mode: action matches if it is ANY remaining required action.
        Used by medium task where both fixes are independent and either can go first.
        Once matched, the action is removed from the pending set.
        """
        key = (action.get("type"), action.get("target"))
        if key in self._pending:
            self._pending.discard(key)
            return True
        return False

    def _count_wrong(self) -> int:
        """Count actions that were NOT part of the required sequence."""
        expected_set = {(a["type"], a["target"]) for a in self.expected}
        return sum(
            1 for a in self.actions_taken
            if (a.get("type"), a.get("target")) not in expected_set
        )
