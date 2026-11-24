"""Runs a RuFaS simulation and records, at every lactating-pen ration formulation, cost and cows underfed.

Usage: python compare_run.py RESULT.jsonl -- <main.py arguments>
"""
import json
import runpy
import sys

from RUFAS.biophysical.animal.data_types.animal_combination import AnimalCombination
from RUFAS.biophysical.animal.nutrients.nutrition_evaluator import NutritionEvaluator
from RUFAS.biophysical.animal.pen import Pen

result_path, args = sys.argv[1], sys.argv[3:]
original = Pen.formulate_optimized_ration


def make(original_fn, outer_only):
    def recorder(self, is_ration_defined_by_user, pen_available_feeds, temperature, *rest, **kw):
        original_fn(self, is_ration_defined_by_user, pen_available_feeds, temperature, *rest, **kw)
        if outer_only and isinstance(self, IndividualFeedingPen):
            return  # the subclass wrapper records after applying the individual rations
        record(self, pen_available_feeds)

    return recorder


def record(self, pen_available_feeds):
    if self.animal_combination is not AnimalCombination.LAC_COW or not self.animals_in_pen:
        return
    price = {f.rufas_id: f.purchase_cost for f in pen_available_feeds}
    under, e_sum = 0, 0.0
    for a in self.animals_in_pen.values():
        _, ev = NutritionEvaluator.evaluate_nutrition_supply(a.nutrition_requirements, a.nutrition_supply, True)
        # underfed = real energy deficit (> 0.05 Mcal) or protein outside bounds (is_valid_* also demands exact DMI, etc.)
        under += 1 if (ev.total_energy < -0.05 or ev.metabolizable_protein != 0.0) else 0
        e_sum += ev.total_energy
    n = len(self.animals_in_pen)
    cost = sum(price.get(i, 0.0) * q for i, q in self.ration.items())
    rec = {"n_cows": n, "cost_per_cow_day": cost, "cost_pen_day": cost * n, "underfed_cows": under,
           "energy_balance_mean": e_sum / len(self.animals_in_pen), "dmi_mean": sum(a.nutrition_supply.dry_matter for a in self.animals_in_pen.values()) / n}
    if hasattr(self, "individual_feeding_report"):
        rec["report"] = self.individual_feeding_report()
    with open(result_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


from RUFAS.biophysical.animal.individual_feeding_pen import IndividualFeedingPen  # noqa: E402

sub_original = IndividualFeedingPen.formulate_optimized_ration
IndividualFeedingPen.formulate_optimized_ration = make(sub_original, False)
Pen.formulate_optimized_ration = make(original, True)
sys.argv = ["main.py"] + args
runpy.run_path("main.py", run_name="__main__")
