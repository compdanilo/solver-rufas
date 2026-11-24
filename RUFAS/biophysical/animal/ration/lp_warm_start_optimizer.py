"""Ration optimizer for RuFaS with a deterministic linear-programming warm start.

The stock ``RationOptimizer`` minimizes feed cost with SLSQP, starting from the previous ration or, when there is
none, from a random point. This module adds ``LPWarmStartRationOptimizer``, a drop-in subclass that replaces the
random start with the result of a sequential linear program (SLP):

1. Every RuFaS constraint ``g_i(x) >= 0`` is linearized numerically around the current point.
2. A cost-minimizing LP (scipy/HiGHS) is solved inside a trust region and inside the feed bounds.
3. Steps 1-2 repeat a few times; the best point found becomes ``x0`` for the unchanged SLSQP solve.

Because the constraints are linearized from RuFaS's own functions, no nutritional equation is re-implemented here:
the scientific content stays in RuFaS. The objective (feed cost) and the final solver are unchanged, so results
differ from the stock optimizer only through the starting point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import OptimizeResult, linprog, minimize

from RUFAS.biophysical.animal.data_types.animal_combination import AnimalCombination
from RUFAS.biophysical.animal.data_types.nutrition_data_structures import NutritionRequirements
from RUFAS.biophysical.animal.ration.ration_optimizer import RationConfig, RationOptimizer
from RUFAS.data_structures.feed_storage_to_animal_connection import RUFAS_ID, Feed, NutrientStandard


@dataclass
class WarmStartResult:
    """Outcome of the sequential-LP warm start."""

    x: npt.NDArray[np.float64]
    cost: float
    max_violation: float
    iterations: int
    feasible: bool


def _evaluate_constraints(constraints: Sequence[dict[str, Any]], x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Evaluates every inequality constraint ``g_i(x)`` (feasible when ``>= 0``)."""
    return np.array([float(c["fun"](x, *c.get("args", ()))) for c in constraints], dtype=float)


def _linearize(
    constraints: Sequence[dict[str, Any]],
    x: npt.NDArray[np.float64],
    step: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Returns ``g(x)`` and its forward-difference Jacobian ``J`` (shape: constraints x feeds)."""
    g0 = _evaluate_constraints(constraints, x)
    jacobian = np.zeros((len(constraints), len(x)), dtype=float)
    for j in range(len(x)):
        h = step * max(1.0, abs(x[j]))
        x_step = x.copy()
        x_step[j] += h
        jacobian[:, j] = (_evaluate_constraints(constraints, x_step) - g0) / h
    return g0, jacobian


def _violation(g: npt.NDArray[np.float64]) -> float:
    """Largest constraint violation (0 when all constraints are satisfied)."""
    return float(max(0.0, -g.min())) if g.size else 0.0


def lp_warm_start(
    prices: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    constraints: Sequence[dict[str, Any]],
    x0: npt.NDArray[np.float64],
    max_iterations: int = 8,
    initial_trust_radius: float = 2.0,
    finite_difference_step: float = 1e-4,
    feasibility_tolerance: float = 1e-6,
) -> WarmStartResult:
    """
    Finds a deterministic starting ration by sequential linear programming.

    Parameters
    ----------
    prices : Sequence[float]
        Cost per kg of each feed (the LP objective).
    bounds : Sequence[tuple[float, float]]
        (lower, upper) limit per feed, kg.
    constraints : Sequence[dict]
        scipy-style ``{"type": "ineq", "fun": f, "args": (...)}`` constraints, feasible when ``f(x, *args) >= 0``.
    x0 : numpy.ndarray
        Point around which the first linearization is made. Clipped into the bounds.
    max_iterations : int
        Maximum number of LP solves.
    initial_trust_radius : float
        Maximum change per feed (kg) allowed in one LP step. Shrinks when a step makes things worse.
    finite_difference_step : float
        Relative step for the numerical Jacobian.
    feasibility_tolerance : float
        Largest constraint violation still counted as feasible.

    Returns
    -------
    WarmStartResult
        Best point found: the cheapest feasible one if any exists, otherwise the least-violating one.
    """
    cost = np.asarray(prices, dtype=float)
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)
    x = np.clip(np.asarray(x0, dtype=float), lower, upper)

    g, jac = _linearize(constraints, x, finite_difference_step)
    current_violation = _violation(g)
    best = WarmStartResult(x.copy(), float(cost @ x), current_violation, 0, current_violation <= feasibility_tolerance)
    radius = initial_trust_radius
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        step_lower = np.maximum(lower, x - radius)
        step_upper = np.minimum(upper, x + radius)
        lp = linprog(
            c=cost,
            A_ub=-jac if len(constraints) else None,
            b_ub=(g - jac @ x) if len(constraints) else None,
            bounds=[(lo, None if np.isinf(hi) else hi) for lo, hi in zip(step_lower, step_upper)],
            method="highs",
        )
        if not lp.success:
            radius *= 2.0  # linearized problem infeasible in this region: look wider
            if radius > 1e3:
                break
            continue

        x_new = np.clip(lp.x, lower, upper)
        g_new, jac_new = _linearize(constraints, x_new, finite_difference_step)
        new_violation = _violation(g_new)

        if new_violation > current_violation + feasibility_tolerance:
            radius *= 0.5  # step made the true constraints worse: reject it and shrink the region
            if radius < 1e-6:
                break
            continue

        step_size = float(np.max(np.abs(x_new - x)))
        x, g, jac, current_violation = x_new, g_new, jac_new, new_violation
        candidate = WarmStartResult(
            x.copy(), float(cost @ x), current_violation, iterations, current_violation <= feasibility_tolerance
        )
        if _is_better(candidate, best, feasibility_tolerance):
            best = candidate
        if step_size < 1e-6:
            break

    best.iterations = iterations
    return best


def _is_better(a: WarmStartResult, b: WarmStartResult, tol: float) -> bool:
    """True when ``a`` should replace ``b``: feasible beats infeasible, then lower cost / lower violation."""
    if a.feasible != b.feasible:
        return a.feasible
    if a.feasible:
        return a.cost < b.cost - tol
    return a.max_violation < b.max_violation - tol


class LPWarmStartRationOptimizer(RationOptimizer):
    """
    ``RationOptimizer`` whose SLSQP solve starts from a sequential-LP solution instead of a random point.

    Use it wherever ``RationOptimizer()`` is instantiated (e.g. ``Pen``); the public interface is identical.

    Attributes
    ----------
    last_warm_start : WarmStartResult | None
        Diagnostics of the most recent warm start (``None`` if it was skipped or failed).
    """

    def __init__(
        self,
        max_lp_iterations: int = 8,
        initial_trust_radius: float = 2.0,
        finite_difference_step: float = 1e-4,
    ) -> None:
        super().__init__()
        self.max_lp_iterations = max_lp_iterations
        self.initial_trust_radius = initial_trust_radius
        self.finite_difference_step = finite_difference_step
        self.last_warm_start: Optional[WarmStartResult] = None

    @staticmethod
    def _deterministic_start(
        previous_ration: Optional[dict[RUFAS_ID | str, float | str]], ration_config: RationConfig
    ) -> npt.NDArray[np.float64]:
        """Previous ration if available, otherwise the dry-matter requirement split evenly across feeds."""
        if previous_ration:
            return np.array(RationOptimizer._build_initial_value(previous_ration, ration_config), dtype=float)
        n_feeds = max(1, len(ration_config.price_list))
        return np.full(n_feeds, ration_config.initial_dry_matter_requirement / n_feeds, dtype=float)

    def attempt_optimization(
        self,
        nutrient_standard: NutrientStandard,
        pen_average_body_weight: float,
        pen_average_enteric_methane: float | None,
        pen_average_urine_nitrogen: float | None,
        requirements: NutritionRequirements,
        initial_dry_matter_requirement: float,
        initial_protein_requirement: float,
        pen_available_feeds: list[Feed],
        animal_combination: AnimalCombination,
        previous_ration: dict[RUFAS_ID | str, float | str] | None = None,
        user_defined_ration_dictionary: dict[RUFAS_ID, float] | None = None,
        user_defined_ration_tolerance: float | None = None,
    ) -> tuple[OptimizeResult, RationConfig]:
        """Same contract as ``RationOptimizer.attempt_optimization``; only the starting point differs."""
        sorted_feeds = sorted(pen_available_feeds, key=lambda feed: feed.rufas_id)
        ration_config = RationConfig(
            nutrient_standard,
            requirements,
            sorted_feeds,
            initial_dry_matter_requirement,
            initial_protein_requirement,
            pen_average_body_weight,
            pen_average_enteric_methane,
            pen_average_urine_nitrogen,
            animal_combination,
        )
        sorted_previous = dict(sorted(previous_ration.items())) if previous_ration else None

        if user_defined_ration_dictionary:
            bounds = self._build_bounds_user_defined_ration(
                ration_config=ration_config,
                user_defined_ration_dictionary=dict(sorted(user_defined_ration_dictionary.items())),
                user_defined_ration_tolerance=cast(float, user_defined_ration_tolerance),
            )
        else:
            bounds = self._build_bounds(ration_config)

        self.set_constraints(ration_config=ration_config)
        constraints = self._select_constraints(animal_combination)

        x0 = self._check_initial_bounds(bounds, self._deterministic_start(sorted_previous, ration_config))
        self.last_warm_start = None
        try:
            warm = lp_warm_start(
                ration_config.price_list,
                bounds,
                constraints,
                x0,
                max_iterations=self.max_lp_iterations,
                initial_trust_radius=self.initial_trust_radius,
                finite_difference_step=self.finite_difference_step,
            )
            self.last_warm_start = warm
            x0 = warm.x
        except Exception:  # noqa: BLE001 - the warm start is an optimization aid; never let it break formulation
            pass

        result = minimize(
            self.objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            args=ration_config,
        )
        return result, ration_config
