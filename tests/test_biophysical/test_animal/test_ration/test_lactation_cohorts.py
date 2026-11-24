import itertools
from types import SimpleNamespace

import pytest

from RUFAS.biophysical.animal.data_types.nutrition_data_structures import NutritionRequirements
from RUFAS.biophysical.animal.ration.lactation_cohorts import (
    LactationStage,
    ParityGroup,
    average_requirements,
    best_plan,
    build_cohorts,
    classify,
    compare_plans,
    plan_average_ration,
    plan_by_cohort,
    plan_groups,
    protected_requirements,
)
from tests.test_biophysical.test_animal.test_ration.toy_solver import ToyLPSolver


def req(dmi, energy, mp):
    r = NutritionRequirements.make_empty_nutrition_requirements()
    r.dry_matter, r.maintenance_energy, r.lactation_energy, r.metabolizable_protein = dmi, 10.0, energy - 10.0, mp
    r.calcium, r.phosphorus = 0.1 * dmi, 0.05 * dmi
    return r


def cow(dim, parity, dmi, energy, mp):
    return SimpleNamespace(days_in_milk=dim, calves=parity, nutrition_requirements=req(dmi, energy, mp))


def herd():
    # (dim, parity, dmi kg, NEL Mcal, MP g): early primiparous eats little and needs a dense diet
    cows = [cow(40, 1, 18.0, 31.0, 2000), cow(60, 1, 19.0, 33.0, 2100)]
    cows += (
        [cow(50, 3, 27.0, 47.0, 2900)] * 3 + [cow(150, 2, 26.0, 43.0, 2500)] * 4 + [cow(250, 2, 24.0, 35.0, 1800)] * 3
    )
    cows += [cow(0, 2, 12.0, 20.0, 900)]  # dry cow: must be ignored
    return cows


@pytest.mark.parametrize(
    "dim,parity,group,stage",
    [
        (1, 1, ParityGroup.PRIMIPAROUS, LactationStage.EARLY),
        (100, 1, ParityGroup.PRIMIPAROUS, LactationStage.EARLY),
        (101, 2, ParityGroup.MULTIPAROUS, LactationStage.MID),
        (200, 4, ParityGroup.MULTIPAROUS, LactationStage.MID),
        (201, 2, ParityGroup.MULTIPAROUS, LactationStage.LATE),
    ],
)
def test_classify_boundaries(dim, parity, group, stage):
    key = classify(dim, parity)
    assert (key.parity_group, key.stage) == (group, stage)


def test_dry_cows_are_not_classified():
    assert classify(0, 3) is None


def test_build_cohorts_ignores_dry_and_averages():
    cohorts = build_cohorts(herd())
    assert sum(c.n_animals for c in cohorts) == 12
    prim = next(c for c in cohorts if c.key.parity_group is ParityGroup.PRIMIPAROUS)
    assert prim.n_animals == 2 and prim.requirements.dry_matter == pytest.approx(18.5)
    densities = [c.energy_density for c in cohorts]
    assert densities == sorted(densities)  # ordered by demand


def test_protected_requirement_covers_densest_cohort():
    cohorts = build_cohorts(herd())
    avg, prot = average_requirements(cohorts), protected_requirements(cohorts)
    top = max(c.energy_density for c in cohorts)
    assert prot.dry_matter == pytest.approx(avg.dry_matter)
    assert prot.total_energy_requirement / prot.dry_matter == pytest.approx(top)
    assert prot.total_energy_requirement >= avg.total_energy_requirement
    assert prot.metabolizable_protein >= avg.metabolizable_protein


def test_average_ration_underfeeds_demanding_cohort_but_protected_does_not():
    cohorts, solver = build_cohorts(herd()), ToyLPSolver()
    avg_plan = plan_average_ration(cohorts, solver)
    prot_plan = plan_groups(cohorts, solver, 1)
    assert avg_plan.n_underfed_cohorts > 0
    assert prot_plan.n_underfed_cohorts == 0
    assert prot_plan.feed_cost > avg_plan.feed_cost  # protection costs money


def test_dp_matches_brute_force_partition():
    cohorts, solver = build_cohorts(herd()), ToyLPSolver()
    cs = sorted(cohorts, key=lambda c: (c.energy_density, c.protein_density))
    for k in (1, 2, 3):
        best = float("inf")
        for cuts in itertools.combinations(range(1, len(cs)), k - 1):
            edges = (0, *cuts, len(cs))
            total = sum(
                sum(c.n_animals for c in cs[a:b]) * solver.solve(protected_requirements(cs[a:b])).cost_per_animal
                for a, b in zip(edges, edges[1:])
            )
            best = min(best, total)
        assert plan_groups(cohorts, solver, k).total_cost == pytest.approx(best, rel=1e-9)


def test_more_groups_never_cost_more_without_fixed_cost():
    cohorts, solver = build_cohorts(herd()), ToyLPSolver()
    costs = [plan_groups(cohorts, solver, k).total_cost for k in range(1, len(cohorts) + 1)]
    assert all(a >= b - 1e-9 for a, b in zip(costs, costs[1:]))
    assert plan_by_cohort(cohorts, solver).total_cost == pytest.approx(costs[-1])


def test_fixed_cost_per_group_can_make_one_group_best():
    cohorts, solver = build_cohorts(herd()), ToyLPSolver()
    free = best_plan(compare_plans(cohorts, solver, fixed_cost_per_group=0.0))
    pricey = best_plan(compare_plans(cohorts, solver, fixed_cost_per_group=10_000.0))
    assert len(pricey.groups) == 1 and len(free.groups) >= len(pricey.groups)


def test_groups_cover_every_cohort_exactly_once():
    cohorts, solver = build_cohorts(herd()), ToyLPSolver()
    plan = plan_groups(cohorts, solver, 3)
    ids = [id(c) for g in plan.groups for c in g]
    assert sorted(ids) == sorted(id(c) for c in cohorts)
    assert plan.n_animals == 12


def varied_herd():
    """Cows inside the same cohort differ a little (as real cows do)."""
    cows = []
    for k in range(6):
        cows.append(cow(40 + k, 1, 18.0 + 0.3 * k, 30.0 + 0.6 * k, 2000 + 60 * k))
    for k in range(8):
        cows.append(cow(50 + k, 3, 26.0 + 0.2 * k, 44.0 + 0.4 * k, 2800 + 40 * k))
    for k in range(6):
        cows.append(cow(240 + k, 2, 23.0 + 0.4 * k, 33.0 + 0.5 * k, 1700 + 50 * k))
    return cows


def test_individual_plan_feeds_every_cow_and_costs_most():
    from RUFAS.biophysical.animal.ration.lactation_cohorts import (
        plan_cohorts_protected_by_cow,
        plan_individual,
        underfed_animals,
    )

    cohorts, solver = build_cohorts(varied_herd()), ToyLPSolver()
    ind, prot_cow = plan_individual(cohorts, solver), plan_cohorts_protected_by_cow(cohorts, solver)
    by_cohort = plan_by_cohort(cohorts, solver)
    assert len(ind.groups) == 20 and underfed_animals(ind, solver) == 0
    assert underfed_animals(prot_cow, solver) == 0
    # fair comparison (every cow adequately fed): individual <= cohort protected by cow
    assert ind.total_cost <= prot_cow.total_cost + 1e-9
    # cohort-average ration is cheaper (convexity of the LP value) but underfeeds some individual cows
    assert by_cohort.total_cost <= ind.total_cost + 1e-9
    assert underfed_animals(by_cohort, solver) > 0
    assert by_cohort.n_underfed_cohorts == 0  # cohort averages are met: the cow-level gap is invisible there


def test_groups_protected_by_cow_leave_no_cow_underfed():
    from RUFAS.biophysical.animal.ration.lactation_cohorts import underfed_animals

    cohorts, solver = build_cohorts(varied_herd()), ToyLPSolver()
    by_cohort_level = plan_groups(cohorts, solver, 2)
    by_cow_level = plan_groups(cohorts, solver, 2, by_cow=True)
    assert underfed_animals(by_cow_level, solver) == 0
    assert underfed_animals(by_cohort_level, solver) > 0
    assert by_cow_level.feed_cost >= by_cohort_level.feed_cost
