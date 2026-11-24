from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from RUFAS.biophysical.animal.data_types.nutrition_data_structures import NutritionRequirements
from RUFAS.biophysical.animal.ration import cohort_ration_solver as csmod
from RUFAS.biophysical.animal.ration.base_concentrate_planner import BaseSolution, plan_base_plus_concentrate
from RUFAS.biophysical.animal.ration.hybrid_ration_solver import RuFaSHybridSolver

ENERGY = np.array([1.3, 2.0, 1.8])
PROTEIN = np.array([60.0, 70.0, 280.0])
PRICE = [0.10, 0.30, 0.60]


def requirement(dmi, energy, mp):
    r = NutritionRequirements.make_empty_nutrition_requirements()
    r.dry_matter, r.maintenance_energy, r.metabolizable_protein = dmi, energy, mp
    return r


def make_solver(mocker, max_concentrate=8.0):
    config = MagicMock(price_list=PRICE, feed_minimum_list=[0.0] * 3, feed_maximum_list=[30.0, 12.0, 6.0])
    mocker.patch.object(csmod, "RationConfig", return_value=config)

    def energy(x, _cfg=None):
        return float(ENERGY @ x) - state["req"].total_energy_requirement

    def protein(x, _cfg=None):
        return float(PROTEIN @ x) - state["req"].metabolizable_protein

    def dmi_low(x, _cfg=None):
        return float(x.sum()) - state["req"].dry_matter

    def dmi_up(x, _cfg=None):
        return 1.10 * state["req"].dry_matter - float(x.sum())

    state = {"req": None}
    optimizer = MagicMock()
    optimizer._select_constraints.return_value = [
        {"type": "ineq", "fun": energy, "args": (None,)},
        {"type": "ineq", "fun": protein, "args": (None,)},
        {"type": "ineq", "fun": dmi_low, "args": (None,)},
        {"type": "ineq", "fun": dmi_up, "args": (None,)},
    ]

    def attempt(**kw):
        state["req"] = kw["requirements"]
        x = np.array([kw["initial_dry_matter_requirement"] * 0.9, kw["initial_dry_matter_requirement"] * 0.1])
        return OptimizeResult(success=True, x=x, fun=float(np.dot(PRICE[: len(x)], x))), MagicMock()

    optimizer.attempt_optimization.side_effect = attempt
    feeds = [MagicMock(rufas_id=i, purchase_cost=p) for i, p in zip((1, 2, 3), PRICE)]
    solver = RuFaSHybridSolver(
        feeds,
        [2, 3],
        base_feed_ids=[1, 2],
        max_concentrate=max_concentrate,
        nutrient_standard=MagicMock(),
        pen_average_body_weight=650.0,
        optimizer=optimizer,
    )

    def set_constraints(cfg):
        pass

    optimizer.set_constraints.side_effect = set_constraints
    return solver, state


def test_base_fractions_sum_to_one_and_use_only_base_feeds(mocker):
    solver, _ = make_solver(mocker)
    base = solver.solve_base(requirement(20.0, 30.0, 1900.0))
    assert base.success and sum(base.fractions.values()) == pytest.approx(1.0) and set(base.fractions) == {1, 2}


def test_concentrate_meets_requirements_within_intake_bounds(mocker):
    solver, state = make_solver(mocker)
    base = BaseSolution(True, 0.12, {1: 0.9, 2: 0.1})
    req = requirement(22.0, 33.0, 1800.0)
    state["req"] = req
    sol = solver.solve_concentrate(req, base)
    assert sol.success
    x = np.array([sol.ration[i] for i in (1, 2, 3)])
    assert 22.0 - 1e-3 <= x.sum() <= 1.10 * 22.0 + 1e-3  # intake is bounded by the DMI constraints, not fixed
    assert ENERGY @ x >= req.total_energy_requirement - 1e-2 and PROTEIN @ x >= req.metabolizable_protein - 1
    assert sum(sol.concentrate.values()) <= 8.0 + 1e-6


def test_more_demanding_cow_gets_more_concentrate(mocker):
    solver, state = make_solver(mocker)
    base = BaseSolution(True, 0.12, {1: 0.9, 2: 0.1})
    amounts = []
    for energy in (33.0, 37.0):
        state["req"] = requirement(22.0, energy, 1700.0)
        amounts.append(sum(solver.solve_concentrate(state["req"], base).concentrate.values()))
    assert amounts[1] > amounts[0]


def test_infeasible_when_concentrate_cap_is_zero(mocker):
    solver, state = make_solver(mocker, max_concentrate=0.0)
    state["req"] = requirement(22.0, 45.0, 2300.0)
    assert not solver.solve_concentrate(state["req"], BaseSolution(True, 0.12, {1: 0.9, 2: 0.1})).success


def test_unknown_concentrate_id_is_rejected(mocker):
    with pytest.raises(ValueError):
        RuFaSHybridSolver([MagicMock(rufas_id=1)], [99], nutrient_standard=MagicMock(), pen_average_body_weight=600.0)
