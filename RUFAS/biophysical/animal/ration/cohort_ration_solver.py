"""Adapter: the RuFaS ration optimizer as a ``RationSolver`` for the cohort planner.

``solve`` formulates a ration with the (warm-started) RuFaS optimizer for a given requirement and returns its cost.
``check`` asks the RuFaS constraints themselves whether that ration would meet another cohort's requirement when
the cohort eats it at its own intake (same composition, scaled to the cohort's dry matter). No nutritional equation is
re-implemented here.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from RUFAS.biophysical.animal.data_types.animal_combination import AnimalCombination
from RUFAS.biophysical.animal.data_types.nutrition_data_structures import NutritionRequirements
from RUFAS.biophysical.animal.ration.lactation_cohorts import (
    AdequacyReport,
    Cohort,
    Plan,
    RationSolution,
    build_cohorts,
    compare_plans,
)
from RUFAS.biophysical.animal.ration.lp_warm_start_optimizer import LPWarmStartRationOptimizer
from RUFAS.biophysical.animal.ration.ration_optimizer import RationConfig, RationOptimizer
from RUFAS.data_structures.feed_storage_to_animal_connection import Feed, NutrientStandard

# Constraints that compare total intake with a target DMI: meaningless when a ration is checked at another intake.
_INTAKE_CONSTRAINTS = {"DMI_constraint_lower", "DMI_constraint_upper"}


class RuFaSCohortSolver:
    """
    Parameters
    ----------
    feeds : list[Feed]
        Feeds available to the pen.
    nutrient_standard : NutrientStandard
        NRC or NASEM, as used by the animals.
    pen_average_body_weight : float
        Average body weight of the animals the rations are formulated for (kg).
    pen_average_enteric_methane, pen_average_urine_nitrogen : float | None
        Only used with NASEM, as in ``Pen._attempt_formulation``.
    optimizer : RationOptimizer | None
        Defaults to ``LPWarmStartRationOptimizer``.
    tolerance : float
        Constraint violation still accepted when checking a cohort.
    """

    def __init__(
        self,
        feeds: list[Feed],
        nutrient_standard: NutrientStandard,
        pen_average_body_weight: float,
        pen_average_enteric_methane: Optional[float] = None,
        pen_average_urine_nitrogen: Optional[float] = None,
        optimizer: Optional[RationOptimizer] = None,
        animal_combination: AnimalCombination = AnimalCombination.LAC_COW,
        tolerance: float = 1e-4,
    ) -> None:
        self.feeds = sorted(feeds, key=lambda f: f.rufas_id)
        self.nutrient_standard = nutrient_standard
        self.body_weight = pen_average_body_weight
        self.methane = pen_average_enteric_methane
        self.urine_nitrogen = pen_average_urine_nitrogen
        self.optimizer = optimizer or LPWarmStartRationOptimizer()
        self.animal_combination = animal_combination
        self.tolerance = tolerance

    def _config(self, requirements: NutritionRequirements) -> RationConfig:
        return RationConfig(
            self.nutrient_standard,
            requirements,
            self.feeds,
            requirements.dry_matter,
            requirements.metabolizable_protein,
            self.body_weight,
            self.methane,
            self.urine_nitrogen,
            self.animal_combination,
        )

    def solve(self, requirements: NutritionRequirements) -> RationSolution:
        result, config = self.optimizer.attempt_optimization(
            nutrient_standard=self.nutrient_standard,
            pen_average_body_weight=self.body_weight,
            pen_average_enteric_methane=self.methane,
            pen_average_urine_nitrogen=self.urine_nitrogen,
            requirements=requirements,
            initial_dry_matter_requirement=requirements.dry_matter,
            initial_protein_requirement=requirements.metabolizable_protein,
            pen_available_feeds=self.feeds,
            animal_combination=self.animal_combination,
        )
        if not result.success:
            return RationSolution(False, float("inf"), detail=result)
        ration = {feed.rufas_id: float(x) for feed, x in zip(self.feeds, result.x)}
        return RationSolution(True, float(result.fun), ration, detail=np.asarray(result.x, dtype=float))

    def check(self, solution: RationSolution, requirements: NutritionRequirements) -> AdequacyReport:
        x = np.asarray(solution.detail, dtype=float)
        total = float(x.sum())
        if total <= 0:
            return AdequacyReport(False, {"ração vazia": 1.0})
        x_cohort = x * (requirements.dry_matter / total)
        config = self._config(requirements)
        self.optimizer.set_constraints(config)
        shortfalls: dict[str, float] = {}
        for constraint in self.optimizer._select_constraints(self.animal_combination):
            name = constraint["fun"].__name__
            if name in _INTAKE_CONSTRAINTS:
                continue
            value = float(constraint["fun"](x_cohort, *constraint.get("args", ())))
            if value < -self.tolerance:
                shortfalls[name] = -value
        return AdequacyReport(not shortfalls, shortfalls)


def plan_pen(
    pen: Any,
    feeds: list[Feed],
    max_groups: Optional[int] = None,
    fixed_cost_per_group: float = 0.0,
    optimizer: Optional[RationOptimizer] = None,
) -> tuple[list[Cohort], list[Plan]]:
    """
    Integration entry point: compares ration strategies for a lactating-cow ``Pen`` (read-only).

    Builds the pen's cohorts from each cow's DIM, parity and requirement, then compares the average ration (what
    ``Pen.formulate_optimized_ration`` does today) against protected rations for 1..k groups. Use
    ``best_plan(plans)`` to pick the cheapest plan that feeds every cohort adequately.
    """
    animals = list(pen.animals_in_pen.values())
    cohorts = build_cohorts(animals)
    nutrient_standard = animals[0].nutrient_standard
    methane, urine_n = (
        pen._calculate_pen_nasem_averages() if nutrient_standard is NutrientStandard.NASEM else (None, None)
    )
    solver = RuFaSCohortSolver(feeds, nutrient_standard, pen.average_body_weight, methane, urine_n, optimizer)
    return cohorts, compare_plans(cohorts, solver, max_groups, fixed_cost_per_group)
