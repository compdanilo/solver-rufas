from unittest.mock import MagicMock

import numpy as np
from scipy.optimize import OptimizeResult

from RUFAS.biophysical.animal.data_types.nutrition_data_structures import NutritionRequirements
from RUFAS.biophysical.animal.ration import cohort_ration_solver as mod
from RUFAS.biophysical.animal.ration.cohort_ration_solver import RuFaSCohortSolver
from RUFAS.biophysical.animal.ration.lactation_cohorts import RationSolution


def requirement(dmi=20.0):
    r = NutritionRequirements.make_empty_nutrition_requirements()
    r.dry_matter, r.metabolizable_protein = dmi, 2000.0
    return r


def make_solver(mocker, result):
    optimizer = MagicMock()
    optimizer.attempt_optimization.return_value = (result, MagicMock())
    feeds = [MagicMock(rufas_id=i) for i in (3, 1, 2)]  # unsorted on purpose
    mocker.patch.object(mod, "RationConfig", return_value=MagicMock())
    return RuFaSCohortSolver(feeds, MagicMock(), 650.0, optimizer=optimizer), optimizer


def test_solve_success_returns_cost_and_ration_by_feed_id(mocker):
    solver, optimizer = make_solver(mocker, OptimizeResult(success=True, x=np.array([5.0, 6.0, 9.0]), fun=7.5))
    sol = solver.solve(requirement())
    assert sol.success and sol.cost_per_animal == 7.5
    assert sol.ration == {1: 5.0, 2: 6.0, 3: 9.0}  # feeds are sorted by id, like the optimizer does
    kwargs = optimizer.attempt_optimization.call_args.kwargs
    assert kwargs["initial_dry_matter_requirement"] == 20.0 and kwargs["pen_average_body_weight"] == 650.0


def test_solve_failure(mocker):
    solver, _ = make_solver(mocker, OptimizeResult(success=False, x=np.zeros(3), fun=0.0))
    sol = solver.solve(requirement())
    assert not sol.success and sol.cost_per_animal == float("inf")


def test_check_scales_to_cohort_intake_and_ignores_dmi_constraints(mocker):
    solver, optimizer = make_solver(mocker, OptimizeResult(success=True, x=np.array([5.0, 5.0, 10.0]), fun=1.0))
    seen = []

    def energy(x, cfg):
        seen.append(x.copy())
        return -0.5  # violated

    def DMI_constraint_upper(x, cfg):
        return -99.0  # must be ignored

    def calcium(x, cfg):
        return 0.2

    optimizer._select_constraints.return_value = [
        {"fun": f, "args": (None,)} for f in (energy, DMI_constraint_upper, calcium)
    ]
    sol = RationSolution(True, 1.0, {}, detail=np.array([5.0, 5.0, 10.0]))  # 20 kg DM
    report = solver.check(sol, requirement(dmi=10.0))  # cohort eats 10 kg
    assert np.allclose(seen[0], [2.5, 2.5, 5.0])
    assert not report.ok and list(report.shortfalls) == ["energy"]


def test_check_ok_when_all_constraints_satisfied(mocker):
    solver, optimizer = make_solver(mocker, OptimizeResult(success=True, x=np.ones(3), fun=1.0))
    optimizer._select_constraints.return_value = [{"fun": lambda x, c: 0.0, "args": (None,)}]
    assert solver.check(RationSolution(True, 1.0, {}, detail=np.ones(3)), requirement()).ok


def test_plan_pen_builds_cohorts_and_compares_plans(mocker):
    from types import SimpleNamespace

    cows = {
        i: SimpleNamespace(
            days_in_milk=dim,
            calves=par,
            nutrient_standard=mod.NutrientStandard.NRC,
            nutrition_requirements=requirement(20.0 + i),
        )
        for i, (dim, par) in enumerate([(30, 1), (90, 3), (250, 2)])
    }
    pen = SimpleNamespace(animals_in_pen=cows, average_body_weight=640.0)
    fake_plans = [MagicMock()]
    cmp_mock = mocker.patch.object(mod, "compare_plans", return_value=fake_plans)
    mocker.patch.object(mod, "RuFaSCohortSolver", return_value=MagicMock())
    cohorts, plans = mod.plan_pen(pen, [MagicMock(rufas_id=1)], max_groups=2, fixed_cost_per_group=5.0)
    assert len(cohorts) == 3 and plans is fake_plans
    assert cmp_mock.call_args.args[2:] == (2, 5.0)
