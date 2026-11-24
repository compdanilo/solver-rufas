"""Lactation-stage and parity aware ration planning for RuFaS.

Why this exists
---------------
RuFaS formulates one ration per pen from the **average** nutrient requirement of the cows in it
(``Pen.average_nutrition_requirements``). Each cow's requirement already depends on days in milk (DIM) and parity
(e.g. the NASEM dry matter intake equation has a parity term and a ``1 - (0.212 + 0.136*p) * exp(-0.053*DIM)`` factor),
but averaging hides the differences: a primiparous cow at 40 DIM eats little and needs a **dense** diet, while a
multiparous cow at 250 DIM eats a lot and needs a **lean** one. A single average ration underfeeds the first and
overfeeds the second.

This module adds the operations-research layer on top of the existing optimizer:

1. **Cohorts.** Lactating cows are classified by parity group (primiparous / multiparous) and lactation stage
   (early / mid / late), and each cohort gets its own average requirement.
2. **Protected group requirement.** A ration fed to several cohorts at once must supply the nutrient *density*
   (per kg of dry matter) demanded by the most demanding cohort in the group.
3. **Pen-grouping problem.** Given ``k`` pens, which cohorts should share a ration so that total feed cost is
   minimal while no cohort is underfed? Cohorts are ordered by energy density demand and split into ``k`` contiguous
   groups by dynamic programming. The cost of each group comes from a ``RationSolver`` (the RuFaS optimizer in
   production, any LP in tests).
4. **Plan comparison.** Average ration (today's behavior) vs. grouped rations vs. one ration per cohort.

Nothing here re-implements a nutritional equation: requirements come from ``Animal.nutrition_requirements`` and feed
costs from the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Optional, Protocol, Sequence

from RUFAS.biophysical.animal.data_types.nutrition_data_structures import NutritionRequirements

EARLY_MAX_DIM = 100
MID_MAX_DIM = 200


class ParityGroup(Enum):
    """Primiparous cows are still growing and produce less; multiparous cows are mature."""

    PRIMIPAROUS = "primípara (1ª lactação)"
    MULTIPAROUS = "multípara (2ª lactação ou mais)"


class LactationStage(Enum):
    """Lactation stage by days in milk (DIM)."""

    EARLY = "início (até 100 DIM)"
    MID = "meio (101 a 200 DIM)"
    LATE = "fim (acima de 200 DIM)"


@dataclass(frozen=True)
class CohortKey:
    parity_group: ParityGroup
    stage: LactationStage

    @property
    def label(self) -> str:
        parity = "1ª lact." if self.parity_group is ParityGroup.PRIMIPAROUS else "2ª+ lact."
        stage = {LactationStage.EARLY: "início", LactationStage.MID: "meio", LactationStage.LATE: "fim"}[self.stage]
        return f"{parity} · {stage}"


def classify(
    days_in_milk: int, parity: int, early_max_dim: int = EARLY_MAX_DIM, mid_max_dim: int = MID_MAX_DIM
) -> Optional[CohortKey]:
    """Returns the cohort of a cow, or ``None`` if the cow is dry (DIM <= 0)."""
    if days_in_milk <= 0:
        return None
    group = ParityGroup.PRIMIPAROUS if parity <= 1 else ParityGroup.MULTIPAROUS
    if days_in_milk <= early_max_dim:
        stage = LactationStage.EARLY
    elif days_in_milk <= mid_max_dim:
        stage = LactationStage.MID
    else:
        stage = LactationStage.LATE
    return CohortKey(group, stage)


def _scaled(req: NutritionRequirements, factor: float) -> NutritionRequirements:
    """``req * factor`` using the division already defined by ``NutritionRequirements``."""
    return req / (1.0 / factor)


def weighted_average(items: Sequence[tuple[NutritionRequirements, int]]) -> NutritionRequirements:
    """Average of requirements weighted by number of animals."""
    total = NutritionRequirements.make_empty_nutrition_requirements()
    count = 0
    for req, n in items:
        total = total + _scaled(req, float(n))
        count += n
    return total / float(count)


@dataclass
class Cohort:
    """Cows sharing parity group and lactation stage, with their average requirement per animal per day."""

    key: CohortKey
    n_animals: int
    mean_dim: float
    mean_parity: float
    requirements: NutritionRequirements
    member_requirements: list[NutritionRequirements] = field(default_factory=list)

    @property
    def energy_density(self) -> float:
        """Total net energy required per kg of dry matter (Mcal/kg DM)."""
        return self.requirements.total_energy_requirement / self.requirements.dry_matter

    @property
    def protein_density(self) -> float:
        """Metabolizable protein required per kg of dry matter (g/kg DM)."""
        return self.requirements.metabolizable_protein / self.requirements.dry_matter


def build_cohorts(
    animals: Iterable[Any], early_max_dim: int = EARLY_MAX_DIM, mid_max_dim: int = MID_MAX_DIM
) -> list[Cohort]:
    """
    Groups lactating cows into cohorts.

    ``animals`` only need ``days_in_milk``, ``calves`` (RuFaS' parity counter) and ``nutrition_requirements``.
    Cohorts are returned ordered by energy density demand (lowest first).
    """
    buckets: dict[CohortKey, list[Any]] = {}
    for animal in animals:
        key = classify(animal.days_in_milk, animal.calves, early_max_dim, mid_max_dim)
        if key is not None:
            buckets.setdefault(key, []).append(animal)
    cohorts = []
    for key, cows in buckets.items():
        n = len(cows)
        cohorts.append(
            Cohort(
                key=key,
                n_animals=n,
                mean_dim=sum(c.days_in_milk for c in cows) / n,
                mean_parity=sum(c.calves for c in cows) / n,
                requirements=weighted_average([(c.nutrition_requirements, 1) for c in cows]),
                member_requirements=[c.nutrition_requirements for c in cows],
            )
        )
    return sorted(cohorts, key=lambda c: (c.energy_density, c.protein_density))


def average_requirements(cohorts: Sequence[Cohort]) -> NutritionRequirements:
    """Pen-average requirement: what RuFaS uses today for a single pen ration."""
    return weighted_average([(c.requirements, c.n_animals) for c in cohorts])


def protected_requirements(cohorts: Sequence[Cohort], by_cow: bool = False) -> NutritionRequirements:
    """
    Requirement for a ration shared by ``cohorts``.

    Starts from the group average (intake, absolute amounts) and raises energy, metabolizable protein, calcium and
    phosphorus so that their **density per kg of dry matter** equals the highest one. Every cow then receives at least
    the density it needs, whatever it eats.

    With ``by_cow=False`` the reference densities are those of the cohort averages; with ``by_cow=True`` they are those
    of the most demanding **individual cow** (safer, more expensive).
    """
    avg = average_requirements(cohorts)
    dmi = avg.dry_matter
    refs = (
        [m for c in cohorts for m in (c.member_requirements or [c.requirements])]
        if by_cow
        else [c.requirements for c in cohorts]
    )
    max_energy_density = max(r.total_energy_requirement / r.dry_matter for r in refs)
    extra_energy = max(0.0, dmi * max_energy_density - avg.total_energy_requirement)

    def dens(attr: str) -> float:
        return float(max(getattr(r, attr) / r.dry_matter for r in refs))

    return replace(
        avg,
        lactation_energy=avg.lactation_energy + extra_energy,
        metabolizable_protein=max(avg.metabolizable_protein, dmi * dens("metabolizable_protein")),
        calcium=max(avg.calcium, dmi * dens("calcium")),
        phosphorus=max(avg.phosphorus, dmi * dens("phosphorus")),
        process_based_phosphorus=max(avg.process_based_phosphorus, dmi * dens("process_based_phosphorus")),
    )


@dataclass
class RationSolution:
    """Result of formulating one ration."""

    success: bool
    cost_per_animal: float  # currency per animal per day
    ration: dict[Any, float] = field(default_factory=dict)  # feed id -> kg DM per animal per day
    detail: Any = None  # solver-specific payload (kept for adequacy checks)


@dataclass
class AdequacyReport:
    ok: bool
    shortfalls: dict[str, float] = field(default_factory=dict)  # nutrient -> relative shortfall (0.05 = 5%)


class RationSolver(Protocol):
    """Anything that can cost a ration for a given requirement (RuFaS optimizer, LP, ...)."""

    def solve(self, requirements: NutritionRequirements) -> RationSolution:
        """Formulates a ration for ``requirements`` and returns its cost."""
        ...

    def check(self, solution: RationSolution, requirements: NutritionRequirements) -> AdequacyReport:
        """Tells whether ``solution`` meets ``requirements`` when eaten at that requirement's intake."""
        ...


@dataclass
class Plan:
    strategy: str
    groups: list[list[Cohort]]
    solutions: list[RationSolution]
    fixed_cost_per_group: float = 0.0
    adequacy: list[list[AdequacyReport]] = field(default_factory=list)  # per group, per cohort

    @property
    def feasible(self) -> bool:
        return all(s.success for s in self.solutions)

    @property
    def feed_cost(self) -> float:
        """Total feed cost per day for all cows."""
        return sum(sum(c.n_animals for c in g) * s.cost_per_animal for g, s in zip(self.groups, self.solutions))

    @property
    def total_cost(self) -> float:
        return self.feed_cost + self.fixed_cost_per_group * len(self.groups)

    @property
    def n_underfed_cohorts(self) -> int:
        return sum(1 for g in self.adequacy for rep in g if not rep.ok)

    @property
    def n_animals(self) -> int:
        return sum(c.n_animals for g in self.groups for c in g)


def _attach_adequacy(plan: Plan, solver: RationSolver) -> Plan:
    for group, sol in zip(plan.groups, plan.solutions):
        plan.adequacy.append(
            [solver.check(sol, c.requirements) if sol.success else AdequacyReport(False) for c in group]
        )
    return plan


def plan_average_ration(cohorts: Sequence[Cohort], solver: RationSolver) -> Plan:
    """Today's RuFaS behavior: one ration from the pen-average requirement."""
    sol = solver.solve(average_requirements(cohorts))
    return _attach_adequacy(Plan("Ração única (média do pen, como o RuFaS faz hoje)", [list(cohorts)], [sol]), solver)


def plan_groups(
    cohorts: Sequence[Cohort],
    solver: RationSolver,
    n_groups: int,
    fixed_cost_per_group: float = 0.0,
    by_cow: bool = False,
) -> Plan:
    """
    Optimal split of the cohorts (ordered by energy density) into ``n_groups`` contiguous groups.

    Dynamic programming: ``best[g][j]`` = minimum cost of feeding the first ``j`` cohorts with ``g`` groups.
    The cost of the group made of cohorts ``i..j-1`` is its protected-requirement ration cost times the number of
    animals. Every group is fed to its most demanding cohort's density, so no cohort is underfed.
    """
    cs = sorted(cohorts, key=lambda c: (c.energy_density, c.protein_density))
    n = len(cs)
    n_groups = max(1, min(n_groups, n))
    cache: dict[tuple[int, int], tuple[float, RationSolution]] = {}

    def group_cost(i: int, j: int) -> tuple[float, RationSolution]:
        if (i, j) not in cache:
            sol = solver.solve(protected_requirements(cs[i:j], by_cow))
            animals = sum(c.n_animals for c in cs[i:j])
            cost = animals * sol.cost_per_animal + fixed_cost_per_group if sol.success else float("inf")
            cache[(i, j)] = (cost, sol)
        return cache[(i, j)]

    inf = float("inf")
    best = [[inf] * (n + 1) for _ in range(n_groups + 1)]
    back = [[-1] * (n + 1) for _ in range(n_groups + 1)]
    best[0][0] = 0.0
    for g in range(1, n_groups + 1):
        for j in range(g, n + 1):
            for i in range(g - 1, j):
                if best[g - 1][i] == inf:
                    continue
                cost = best[g - 1][i] + group_cost(i, j)[0]
                if cost < best[g][j]:
                    best[g][j], back[g][j] = cost, i
    groups: list[list[Cohort]] = []
    sols: list[RationSolution] = []
    j = n
    for g in range(n_groups, 0, -1):
        i = back[g][j]
        if i < 0:  # infeasible: fall back to a failed single plan
            return Plan(
                f"{n_groups} grupos (inviável)", [list(cs)], [RationSolution(False, float("inf"))], fixed_cost_per_group
            )
        groups.append(cs[i:j])
        sols.append(group_cost(i, j)[1])
        j = i
    groups.reverse()
    sols.reverse()
    label = "Ração única protegida" if n_groups == 1 else f"{n_groups} grupos protegidos"
    return _attach_adequacy(Plan(label, groups, sols, fixed_cost_per_group), solver)


def plan_by_cohort(cohorts: Sequence[Cohort], solver: RationSolver, fixed_cost_per_group: float = 0.0) -> Plan:
    """One ration per cohort: the lower bound on feed cost (and the most pens/rations to manage)."""
    plan = plan_groups(cohorts, solver, len(cohorts), fixed_cost_per_group)
    plan.strategy = "Uma ração por coorte (atende a média de cada coorte)"
    return plan


def compare_plans(
    cohorts: Sequence[Cohort], solver: RationSolver, max_groups: int | None = None, fixed_cost_per_group: float = 0.0
) -> list[Plan]:
    """Average ration, then k = 1..max_groups protected groups. The cheapest feasible plan has the lowest total cost."""
    max_groups = max_groups or len(cohorts)
    plans = [plan_average_ration(cohorts, solver)]
    plans += [plan_groups(cohorts, solver, k, fixed_cost_per_group) for k in range(1, max_groups + 1)]
    return plans


def best_plan(plans: Sequence[Plan]) -> Plan:
    """Cheapest plan (feed + fixed cost) among those that feed every cohort adequately."""
    ok = [p for p in plans if p.feasible and p.n_underfed_cohorts == 0]
    return min(ok, key=lambda p: p.total_cost)


def _member_protected(cohort: Cohort) -> NutritionRequirements:
    """Cohort requirement raised to the nutrient density of its most demanding cow."""
    members = cohort.member_requirements or [cohort.requirements]
    singles = [Cohort(cohort.key, 1, cohort.mean_dim, cohort.mean_parity, m) for m in members]
    return protected_requirements(singles)


def plan_cohorts_protected_by_cow(
    cohorts: Sequence[Cohort], solver: RationSolver, fixed_cost_per_group: float = 0.0
) -> Plan:
    """One ration per cohort, each sized for the most demanding **cow** of the cohort (no cow is underfed)."""
    sols = [solver.solve(_member_protected(c)) for c in cohorts]
    plan = Plan(
        "Uma ração por coorte, protegida para a vaca mais exigente", [[c] for c in cohorts], sols, fixed_cost_per_group
    )
    return _attach_adequacy(plan, solver)


def plan_individual(cohorts: Sequence[Cohort], solver: RationSolver, fixed_cost_per_group: float = 0.0) -> Plan:
    """
    One ration per cow, from that cow's own requirement.

    Needs one optimization per cow per formulation interval. In a real barn this is only possible with individual
    feeding (feeder stations / robots); a TMR cannot be individualized. It is the cost floor for adequate feeding.
    """
    singles = [
        Cohort(c.key, 1, c.mean_dim, c.mean_parity, m, [m])
        for c in cohorts
        for m in (c.member_requirements or [c.requirements])
    ]
    sols = [solver.solve(c.requirements) for c in singles]
    plan = Plan("Uma ração por vaca (individual)", [[c] for c in singles], sols, fixed_cost_per_group)
    return _attach_adequacy(plan, solver)


def underfed_animals(plan: Plan, solver: RationSolver) -> int:
    """Number of individual cows whose own requirement is not met by the ration their group receives."""
    count = 0
    for group, sol in zip(plan.groups, plan.solutions):
        for cohort in group:
            for req in cohort.member_requirements or [cohort.requirements]:
                if not sol.success or not solver.check(sol, req).ok:
                    count += 1
    return count
