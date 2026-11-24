"""Demo: single ration vs base + individual concentrate vs one ration per cow (toy feeds, fictitious herd)."""
import json

from demo_cohorts import make_herd
from RUFAS.biophysical.animal.ration.base_concentrate_planner import plan_base_plus_concentrate, reference_requirement
from tests.test_biophysical.test_animal.test_ration.toy_solver import ToyHybridSolver, ToyLPSolver


def run():
    herd = [c for c in make_herd(120)]
    reqs = [c.nutrition_requirements for c in herd]
    # RuFaS-style intake: cows may eat up to 8% above their estimated DMI
    solver = ToyHybridSolver(max_concentrate=8.0, intake_slack=0.08)
    plan = plan_base_plus_concentrate(reqs, solver, n_grid=11, refine_iterations=8)
    individual = sum(ToyLPSolver().solve(r).cost_per_animal for r in reqs)
    avg_dmi = sum(r.dry_matter for r in reqs) / len(reqs)
    top = solver.solve_base(reference_requirement(reqs, 1.0))
    single_protected = top.cost_per_kg * avg_dmi * len(reqs) if top.success else float("inf")
    avg = solver.solve_base(reference_requirement(reqs, 0.0)).cost_per_kg * avg_dmi * len(reqs)
    return {
        "n_cows": len(reqs), "theta": round(plan.theta, 3), "feasible": plan.feasible, "n_infeasible": plan.n_infeasible,
        "hybrid_cost": round(plan.total_cost, 2), "individual_cost": round(individual, 2),
        "single_protected_cost": round(single_protected, 2), "single_average_cost": round(avg, 2),
        "total_concentrate_kg": round(plan.total_concentrate, 1),
        "curve": [(round(t, 3), None if c == float("inf") else round(c, 1)) for t, c in plan.curve],
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=1))
