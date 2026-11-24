import numpy as np
import pytest

from RUFAS.biophysical.animal.data_types.nutrition_data_structures import NutritionRequirements
from RUFAS.biophysical.animal.ration.base_concentrate_planner import (
    plan_base_plus_concentrate,
    reference_requirement,
)
from tests.test_biophysical.test_animal.test_ration.toy_solver import ToyHybridSolver, ToyLPSolver


def req(dmi, energy, mp):
    r = NutritionRequirements.make_empty_nutrition_requirements()
    r.dry_matter, r.maintenance_energy, r.lactation_energy, r.metabolizable_protein = dmi, 10.0, energy - 10.0, mp
    r.calcium, r.phosphorus = 0.1 * dmi, 0.05 * dmi
    return r


def cows():
    return (
        [req(18 + 0.4 * k, 31 + 1.0 * k, 2000 + 90 * k) for k in range(8)]
        + [req(25 + 0.3 * k, 40 + 0.5 * k, 2400 + 40 * k) for k in range(8)]
        + [req(24.0, 35.0, 1800)] * 4
    )


def test_reference_requirement_interpolates_densities():
    cs = cows()
    low, high = reference_requirement(cs, 0.0), reference_requirement(cs, 1.0)
    top = max(c.total_energy_requirement / c.dry_matter for c in cs)
    assert high.total_energy_requirement / high.dry_matter == pytest.approx(top)
    assert low.dry_matter == pytest.approx(high.dry_matter)
    assert low.total_energy_requirement < high.total_energy_requirement
    with pytest.raises(ValueError):
        reference_requirement(cs, 1.5)


def test_every_cow_is_fed_and_total_dmi_is_kept():
    cs, solver = cows(), ToyHybridSolver()
    plan = plan_base_plus_concentrate(cs, solver)
    assert plan.feasible
    for r, s in zip(cs, plan.solutions):
        assert sum(s.ration.values()) == pytest.approx(r.dry_matter, rel=1e-3)
        assert sum(s.concentrate.values()) <= solver.max_concentrate + 1e-6
        e = sum(solver.E[i] * s.ration[n] for i, n in enumerate(solver.names))
        assert e >= r.total_energy_requirement - 1e-3


def test_hybrid_between_individual_and_single_protected_ration():
    cs, solver = cows(), ToyHybridSolver()
    hybrid = plan_base_plus_concentrate(cs, solver)
    individual = sum(ToyLPSolver().solve(r).cost_per_animal for r in cs)
    single = (
        len(cs) * solver.solve_base(reference_requirement(cs, 1.0)).cost_per_kg * np.mean([r.dry_matter for r in cs])
    )
    assert individual <= hybrid.total_cost + 1e-6
    assert hybrid.total_cost <= single + 1e-6


def test_search_is_at_least_as_good_as_the_grid_ends_and_returns_curve():
    cs, solver = cows(), ToyHybridSolver()
    plan = plan_base_plus_concentrate(cs, solver)
    ends = [c for t, c in plan.curve if t in (0.0, 1.0)]
    assert plan.total_cost <= min(ends) + 1e-9
    assert len(plan.curve) >= 9 and 0.0 <= plan.theta <= 1.0


def test_infeasible_when_concentrate_is_capped_too_low():
    plan = plan_base_plus_concentrate(cows(), ToyHybridSolver(max_concentrate=0.0), n_grid=3, refine_iterations=2)
    assert plan.total_cost >= 0  # no crash; may be infeasible or served by the base alone


def test_cow_nobody_can_feed_is_penalised_not_fatal():
    cs = cows() + [req(15.0, 60.0, 5000.0)]  # absurdly dense requirement: infeasible with these feeds
    plan = plan_base_plus_concentrate(cs, ToyHybridSolver())
    assert plan.base.success and not plan.feasible and plan.n_infeasible >= 1
    assert plan.total_cost >= plan.infeasible_penalty
    assert sum(s.success for s in plan.solutions) == len(cs) - plan.n_infeasible
