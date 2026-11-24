"""Pen that feeds each lactating cow according to her own requirement (opt-in extension of ``Pen``).

``Pen`` formulates ONE ration from the pen-average requirement and gives it to every cow. ``IndividualFeedingPen``
keeps that behavior as the starting point and, when a mode is configured, replaces the per-animal nutrient supply:

* ``INDIVIDUAL``: one ration per cow, from her own ``nutrition_requirements`` (DIM and parity are already inside them);
* ``BASE_CONCENTRATE``: one base ration for the pen plus an individual concentrate per cow
  (see ``base_concentrate_planner``).

``Pen.ration`` becomes the **mean** of the individual rations, so the feed demand computed from the pen ration (per
animal x number of animals) stays equal to what the cows are actually offered. If anything fails, the cow keeps the
pen ration, i.e. the stock RuFaS behavior.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from RUFAS.biophysical.animal.animal import Animal
from RUFAS.biophysical.animal.data_types.animal_combination import AnimalCombination
from RUFAS.biophysical.animal.nutrients.nutrition_evaluator import NutritionEvaluator
from RUFAS.biophysical.animal.nutrients.nutrition_supply_calculator import NutritionSupplyCalculator
from RUFAS.biophysical.animal.pen import Pen
from RUFAS.biophysical.animal.ration.base_concentrate_planner import plan_base_plus_concentrate
from RUFAS.biophysical.animal.ration.cohort_ration_solver import RuFaSCohortSolver
from RUFAS.biophysical.animal.ration.hybrid_ration_solver import RuFaSHybridSolver
from RUFAS.data_structures.feed_storage_to_animal_connection import RUFAS_ID, Feed, NutrientStandard

INDIVIDUAL = "individual"
BASE_CONCENTRATE = "base_concentrate"
Ration = dict[RUFAS_ID, float]


def mean_ration(rations: Sequence[Ration]) -> Ration:
    """Feed-by-feed mean of several rations (kg DM per animal per day)."""
    ids = sorted({i for r in rations for i in r})
    return {i: sum(r.get(i, 0.0) for r in rations) / len(rations) for i in ids}


def apply_individual_rations(animals: dict[int, Animal], feeds_used: list[Feed], rations: dict[int, Ration]) -> None:
    """Sets each animal's nutrient supply from ITS ration (animals without one are left untouched)."""
    for animal_id, ration in rations.items():
        animal = animals[animal_id]
        animal.previous_nutrition_supply = animal.nutrition_supply
        animal.nutrition_supply = NutritionSupplyCalculator.calculate_nutrient_supply(
            feeds_used=feeds_used,
            ration_formulation=ration,
            body_weight=animal.body_weight,
            enteric_methane=animal.enteric_methane,
            urinary_nitrogen=animal.digestive_system.manure_excretion.urine_nitrogen,
        )
        animal.nutrients.set_dry_matter_intake(animal.nutrition_supply.dry_matter)
        animal.nutrients.set_phosphorus_intake(animal.nutrition_supply.phosphorus)


class IndividualFeedingPen(Pen):
    """``Pen`` whose lactating cows can be fed individually (see module docstring)."""

    def configure_individual_feeding(
        self,
        mode: Optional[str],
        concentrate_feed_ids: Sequence[RUFAS_ID] = (),
        base_feed_ids: Optional[Sequence[RUFAS_ID]] = None,
        max_concentrate: float = 8.0,
    ) -> None:
        if mode not in (None, INDIVIDUAL, BASE_CONCENTRATE):
            raise ValueError(f"unknown individual feeding mode: {mode}")
        if mode == BASE_CONCENTRATE and not concentrate_feed_ids:
            raise ValueError("BASE_CONCENTRATE needs concentrate_feed_ids")
        self.individual_mode = mode
        self.concentrate_feed_ids = list(concentrate_feed_ids)
        self.base_feed_ids = base_feed_ids
        self.max_concentrate = max_concentrate
        self.individual_costs: dict[int, float] = {}
        self.individual_fallbacks: list[int] = []
        self.individual_theta: Optional[float] = None
        self.individual_error: Optional[str] = None

    def formulate_optimized_ration(
        self, is_ration_defined_by_user: bool, pen_available_feeds: list[Feed], *args: Any, **kwargs: Any
    ) -> None:
        super().formulate_optimized_ration(is_ration_defined_by_user, pen_available_feeds, *args, **kwargs)
        mode = getattr(self, "individual_mode", None)
        if (
            mode is None
            or self.animal_combination is not AnimalCombination.LAC_COW
            or is_ration_defined_by_user
            or not self.animals_in_pen
            or not self.ration
        ):
            return
        rations = self._individual_rations(pen_available_feeds, mode)
        if not rations:
            return
        full = {i: dict(rations.get(i, self.ration)) for i in self.animals_in_pen}
        apply_individual_rations(self.animals_in_pen, pen_available_feeds, full)
        self.ration = mean_ration(list(full.values()))
        _, evaluation = NutritionEvaluator.evaluate_nutrition_supply(
            self.average_nutrition_requirements, self.average_nutrition_supply, True
        )
        self.average_nutrition_evaluation = evaluation

    def _cow_solver(self, feeds: list[Feed], animal: Animal) -> RuFaSCohortSolver:
        standard = animal.nutrient_standard
        methane = animal.enteric_methane if standard is NutrientStandard.NASEM else None
        urine = animal.digestive_system.manure_excretion.urine_nitrogen if standard is NutrientStandard.NASEM else None
        return RuFaSCohortSolver(feeds, standard, animal.body_weight, methane, urine, self.ration_optimizer)

    def _individual_rations(self, feeds: list[Feed], mode: str) -> dict[int, Ration]:
        self.individual_costs, self.individual_fallbacks, self.individual_theta = {}, [], None
        self.individual_error = None
        animals = self.animals_in_pen
        out: dict[int, Ration] = {}
        try:
            if mode == INDIVIDUAL:
                for animal_id, animal in animals.items():
                    sol = self._cow_solver(feeds, animal).solve(animal.nutrition_requirements)
                    self._store(out, animal_id, sol.success, sol.ration, sol.cost_per_animal)
            else:
                ids = list(animals)
                first = animals[ids[0]]
                nasem = first.nutrient_standard is NutrientStandard.NASEM
                methane, urine = self._calculate_pen_nasem_averages() if nasem else (None, None)
                hybrid = RuFaSHybridSolver(
                    feeds,
                    self.concentrate_feed_ids,
                    self.base_feed_ids,
                    self.max_concentrate,
                    nutrient_standard=first.nutrient_standard,
                    pen_average_body_weight=self.average_body_weight,
                    pen_average_enteric_methane=methane,
                    pen_average_urine_nitrogen=urine,
                    optimizer=self.ration_optimizer,
                )
                hybrid.cow_contexts = [
                    (
                        animals[i].body_weight,
                        animals[i].enteric_methane if nasem else None,
                        animals[i].digestive_system.manure_excretion.urine_nitrogen if nasem else None,
                    )
                    for i in ids
                ]
                plan = plan_base_plus_concentrate(
                    [animals[i].nutrition_requirements for i in ids], hybrid, n_grid=5, refine_iterations=4
                )
                self.individual_theta = plan.theta
                if plan.base.success:  # cows no concentrate can feed adequately keep the pen ration
                    for animal_id, csol in zip(ids, plan.solutions):
                        self._store(out, animal_id, csol.success, csol.ration, csol.cost)
                else:
                    self.individual_fallbacks = ids
        except Exception as error:  # keep the simulation running with the stock pen ration
            import traceback

            self.individual_error = (
                f"{type(error).__name__}: {error} @ {traceback.format_exc().splitlines()[-3].strip()}"
            )
            self.om.add_warning(
                "Individual feeding failed; the pen ration was kept.",
                self.individual_error,
                {"class": "IndividualFeedingPen", "function": "_individual_rations"},
            )
            return {}
        return out

    def _store(self, out: dict[int, Ration], animal_id: int, ok: bool, ration: Ration, cost: float) -> None:
        if ok:
            out[animal_id] = dict(ration)
            self.individual_costs[animal_id] = cost
        else:
            self.individual_fallbacks.append(animal_id)

    def individual_feeding_report(self) -> dict[str, Any]:
        n = len(self.animals_in_pen)
        return {
            "mode": getattr(self, "individual_mode", None),
            "n_animals": n,
            "n_individual": len(getattr(self, "individual_costs", {})),
            "n_fallback_to_pen_ration": len(getattr(self, "individual_fallbacks", [])),
            "total_cost": sum(getattr(self, "individual_costs", {}).values()),
            "theta": getattr(self, "individual_theta", None),
            "error": getattr(self, "individual_error", None),
        }
