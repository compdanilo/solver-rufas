"""Demo: cohort planning with a synthetic herd (illustrative numbers, not calibrated to a real farm)."""
import json
import random
from math import exp
from types import SimpleNamespace

from RUFAS.biophysical.animal.data_types.nutrition_data_structures import NutritionRequirements
from RUFAS.biophysical.animal.nutrients.nasem_requirements_calculator import NASEMRequirementsCalculator
from RUFAS.biophysical.animal.ration.lactation_cohorts import (
    best_plan, build_cohorts, compare_plans, plan_by_cohort,
)
from tests.test_biophysical.test_animal.test_ration.toy_solver import FEEDS, ToyLPSolver

WOOD = {"prim": (19.0, 0.18, 0.0023), "multi": (22.0, 0.22, 0.0032)}  # illustrative Wood (a, b, c)
BW = {"prim": 590.0, "multi": 690.0}
NEL_MILK = 0.70  # Mcal per kg of milk (typical value)


def milk(group, dim):
    a, b, c = WOOD[group]
    return a * dim**b * exp(-c * dim)


def requirement(group, dim, parity):
    bw = BW[group]
    y = milk(group, dim)
    nel_milk = y * NEL_MILK
    dmi = NASEMRequirementsCalculator._calculate_dry_matter_intake(
        body_weight=bw, mature_body_weight=700.0, days_in_milk=dim, lactating=True,
        net_energy_lactation=nel_milk, parity=parity, body_condition_score_5=3.0, NDF_conc=30.0)
    r = NutritionRequirements.make_empty_nutrition_requirements()
    r.dry_matter = dmi
    r.maintenance_energy = 0.08 * bw**0.75
    r.activity_energy = 0.4
    r.lactation_energy = nel_milk
    r.metabolizable_protein = 600.0 + 46.0 * y + (150.0 if group == "prim" else 0.0)
    r.calcium = 30.0 + 1.2 * y
    r.phosphorus = 20.0 + 0.9 * y
    return r


def make_herd(n=200, seed=7):
    rnd = random.Random(seed)
    cows = []
    for _ in range(n):
        group = "prim" if rnd.random() < 0.38 else "multi"
        parity = 1 if group == "prim" else rnd.choice([2, 2, 3, 4])
        dim = rnd.randint(21, 340)
        cows.append(SimpleNamespace(days_in_milk=dim, calves=parity, group=group,
                                    nutrition_requirements=requirement(group, dim, parity)))
    return cows


def run():
    herd = make_herd()
    cohorts = build_cohorts(herd)
    solver = ToyLPSolver()
    plans = compare_plans(cohorts, solver, max_groups=len(cohorts))
    chosen = best_plan(plans)
    out = {"cohorts": [], "plans": []}
    for c in cohorts:
        sol = solver.solve(c.requirements)
        out["cohorts"].append({"label": c.key.label, "n": c.n_animals, "dim": round(c.mean_dim), "dmi": round(c.requirements.dry_matter, 1),
                               "energy": round(c.requirements.total_energy_requirement, 1), "e_dens": round(c.energy_density, 3),
                               "mp": round(c.requirements.metabolizable_protein), "mp_dens": round(c.protein_density, 1),
                               "cost": round(sol.cost_per_animal, 2), "ration": {k: float(v) for k, v in sol.ration.items()}})
    for p in plans:
        out["plans"].append({"strategy": p.strategy, "n_groups": len(p.groups), "feed_cost": round(p.feed_cost, 2),
                             "underfed": p.n_underfed_cohorts, "feasible": p.feasible,
                             "groups": [[c.key.label for c in g] for g in p.groups]})
    out["best"] = chosen.strategy
    out["n_animals"] = sum(c.n_animals for c in cohorts)
    return herd, cohorts, plans, chosen, out


if __name__ == "__main__":
    herd, cohorts, plans, chosen, out = run()
    print(json.dumps(out, ensure_ascii=False, indent=1))
