"""Base ration + individual concentrate (a hybrid between one pen ration and one ration per cow).

Idea (how TMR + automatic concentrate feeders work on real farms)
-----------------------------------------------------------------
The pen receives one **base ration** with composition ``m`` (fractions that sum to 1). On top of it each cow ``j``
receives an individual amount of concentrate ``c_j`` (kg DM per feed, ``c_j >= 0``) on top of the base she eats
(``b_j`` kg DM)::

    x_j = b_j * m + c_j

Total intake ``b_j + sum(c_j)`` is not fixed: it is bounded by the solver's own intake constraints (in RuFaS, the DMI
constraints), because RuFaS meets the energy of demanding cows by letting them eat more.

For a fixed base ``m`` the problem **decomposes**: one small problem per cow (choose ``c_j`` to minimise cost subject to
the cow's own requirements). The base is chosen by an outer search over a single number ``theta`` in [0, 1]:

* ``theta = 0``: base designed for the average cow (cheap base, the demanding cows need a lot of concentrate);
* ``theta = 1``: base designed for the most demanding cow (dense base, nobody needs much concentrate, but it is
  expensive for everyone).

The total cost as a function of ``theta`` is evaluated on a grid and refined by golden-section search. By construction
``individual <= hybrid <= single ration protected for the most demanding cow`` (when the same solver is used).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol, Sequence

from RUFAS.biophysical.animal.data_types.nutrition_data_structures import NutritionRequirements

_PHI = (5**0.5 - 1) / 2
_DENSITY_ATTRS = ("metabolizable_protein", "calcium", "phosphorus", "process_based_phosphorus")


@dataclass
class BaseSolution:
    success: bool
    cost_per_kg: float = float("inf")  # cost of 1 kg DM of base
    fractions: dict[Any, float] = field(default_factory=dict)  # feed id -> fraction of base DM (sums to 1)


@dataclass
class ConcentrateSolution:
    success: bool
    cost: float = float("inf")  # total cost per day (base share + concentrate)
    concentrate: dict[Any, float] = field(default_factory=dict)  # feed id -> kg DM per day
    ration: dict[Any, float] = field(default_factory=dict)  # feed id -> total kg DM per day


class HybridSolver(Protocol):
    def solve_base(self, requirements: NutritionRequirements) -> BaseSolution:
        """Base ration (fractions) formulated for ``requirements``."""
        ...

    def solve_concentrate(
        self, requirements: NutritionRequirements, base: BaseSolution, index: int = 0
    ) -> ConcentrateSolution:
        """Cheapest concentrate for ONE cow (the ``index``-th of the herd) given the base composition."""
        ...


@dataclass
class HybridPlan:
    theta: float
    base: BaseSolution
    solutions: list[ConcentrateSolution]
    curve: list[tuple[float, float]]  # (theta, total cost); inf if the base itself is infeasible
    infeasible_penalty: float = 1000.0  # cost charged per cow that no concentrate can feed adequately

    @property
    def n_infeasible(self) -> int:
        return sum(1 for s in self.solutions if not s.success)

    @property
    def feasible(self) -> bool:
        """True when the base exists and every cow can be fed adequately."""
        return self.base.success and bool(self.solutions) and self.n_infeasible == 0

    @property
    def total_cost(self) -> float:
        """Cost of the cows that can be fed, plus a penalty for each one that cannot (so ``theta`` minimises both)."""
        if not (self.base.success and self.solutions):
            return float("inf")
        return sum(s.cost for s in self.solutions if s.success) + self.infeasible_penalty * self.n_infeasible

    @property
    def total_concentrate(self) -> float:
        return sum(sum(s.concentrate.values()) for s in self.solutions)


def _density(r: NutritionRequirements, attr: str) -> float:
    value = r.total_energy_requirement if attr == "energy" else getattr(r, attr)
    return float(value) / r.dry_matter


def reference_requirement(cow_requirements: Sequence[NutritionRequirements], theta: float) -> NutritionRequirements:
    """
    Requirement the base is designed for: average intake, with nutrient densities between the herd average
    (``theta=0``) and the most demanding cow (``theta=1``).
    """
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must be in [0, 1]")
    n = len(cow_requirements)
    dmi = sum(r.dry_matter for r in cow_requirements) / n
    ref = replace(cow_requirements[0], dry_matter=dmi)

    def interpolated(attr: str) -> float:
        dens = [_density(r, attr) for r in cow_requirements]
        average = sum(r_ * r.dry_matter for r_, r in zip(dens, cow_requirements)) / sum(
            r.dry_matter for r in cow_requirements
        )
        return float(average + theta * (max(dens) - average))

    energy_total = dmi * interpolated("energy")
    other = energy_total - (ref.total_energy_requirement - ref.lactation_energy)
    kwargs: dict[str, Any] = {a: dmi * interpolated(a) for a in _DENSITY_ATTRS}
    return replace(ref, lactation_energy=other, **kwargs)


def _evaluate(
    theta: float,
    cow_requirements: Sequence[NutritionRequirements],
    solver: HybridSolver,
    cache: dict[float, HybridPlan],
    infeasible_penalty: float = 1000.0,
) -> HybridPlan:
    if theta in cache:
        return cache[theta]
    base = solver.solve_base(reference_requirement(cow_requirements, theta))
    sols = [solver.solve_concentrate(r, base, i) for i, r in enumerate(cow_requirements)] if base.success else []
    plan = HybridPlan(theta, base, sols, [], infeasible_penalty)
    cache[theta] = plan
    return plan


def search_base_density(
    cow_requirements: Sequence[NutritionRequirements],
    solver: HybridSolver,
    n_grid: int = 9,
    refine_iterations: int = 10,
) -> HybridPlan:
    """Outer search over ``theta``: grid, then golden-section refinement around the best grid point."""
    cache: dict[float, HybridPlan] = {}
    grid = [i / (n_grid - 1) for i in range(n_grid)]
    for t in grid:
        _evaluate(t, cow_requirements, solver, cache)
    best_t = min(grid, key=lambda t: cache[t].total_cost)
    if cache[best_t].total_cost < float("inf"):
        step = 1.0 / (n_grid - 1)
        lo, hi = max(0.0, best_t - step), min(1.0, best_t + step)
        a, b = hi - _PHI * (hi - lo), lo + _PHI * (hi - lo)
        for _ in range(refine_iterations):
            if (
                _evaluate(a, cow_requirements, solver, cache).total_cost
                <= _evaluate(b, cow_requirements, solver, cache).total_cost
            ):
                hi, b = b, a
                a = hi - _PHI * (hi - lo)
            else:
                lo, a = a, b
                b = lo + _PHI * (hi - lo)
    curve = sorted((t, p.total_cost) for t, p in cache.items())
    best = min(cache.values(), key=lambda p: p.total_cost)
    return replace(best, curve=curve)


def plan_base_plus_concentrate(
    cow_requirements: Sequence[NutritionRequirements],
    solver: HybridSolver,
    n_grid: int = 9,
    refine_iterations: int = 10,
) -> HybridPlan:
    """Entry point: best base ration + individual concentrate for a list of cow requirements (one per cow)."""
    if not cow_requirements:
        raise ValueError("no cows")
    return search_base_density(list(cow_requirements), solver, n_grid, refine_iterations)
