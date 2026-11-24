from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock

import pytest

from RUFAS.biophysical.animal import individual_feeding_pen as mod
from RUFAS.biophysical.animal.data_types.animal_combination import AnimalCombination
from RUFAS.biophysical.animal.individual_feeding_pen import (
    BASE_CONCENTRATE,
    INDIVIDUAL,
    IndividualFeedingPen,
    apply_individual_rations,
    mean_ration,
)
from RUFAS.biophysical.animal.pen import Pen
from RUFAS.biophysical.animal.ration.lactation_cohorts import RationSolution


def make_animal():
    return MagicMock(body_weight=650.0, enteric_methane=400.0)


def make_pen(mocker, n=3, combination=AnimalCombination.LAC_COW):
    pen = object.__new__(IndividualFeedingPen)
    pen.animal_combination = combination
    pen.animals_in_pen = {i: make_animal() for i in range(n)}
    pen.ration = {1: 10.0, 2: 8.0}
    pen.ration_optimizer = MagicMock()
    pen.om = MagicMock()
    mocker.patch.object(Pen, "formulate_optimized_ration")
    mocker.patch.object(Pen, "average_nutrition_requirements", new_callable=PropertyMock)
    mocker.patch.object(Pen, "average_nutrition_supply", new_callable=PropertyMock)
    mocker.patch.object(Pen, "average_body_weight", new_callable=PropertyMock, return_value=650.0)
    mocker.patch.object(mod.NutritionEvaluator, "evaluate_nutrition_supply", return_value=(None, "EVAL"))
    mocker.patch.object(
        mod.NutritionSupplyCalculator,
        "calculate_nutrient_supply",
        side_effect=lambda **kw: SimpleNamespace(dry_matter=1.0, phosphorus=0.1, **kw),
    )
    return pen


def formulate(pen, feeds=None, user=False):
    pen.formulate_optimized_ration(
        user, feeds or [MagicMock(rufas_id=1), MagicMock(rufas_id=2)], 20.0, {}, MagicMock(), 1
    )


def test_mean_ration():
    assert mean_ration([{1: 10.0, 2: 0.0}, {1: 20.0, 3: 6.0}]) == {1: 15.0, 2: 0.0, 3: 3.0}


def test_apply_sets_supply_per_animal_from_its_own_ration(mocker):
    mocker.patch.object(
        mod.NutritionSupplyCalculator,
        "calculate_nutrient_supply",
        side_effect=lambda **kw: SimpleNamespace(dry_matter=1.0, phosphorus=0.1, **kw),
    )
    animals = {0: make_animal(), 1: make_animal()}
    apply_individual_rations(animals, [], {0: {1: 5.0}, 1: {1: 9.0}})
    assert animals[0].nutrition_supply.ration_formulation == {1: 5.0}
    assert animals[1].nutrition_supply.ration_formulation == {1: 9.0}


def test_mode_none_keeps_stock_behavior(mocker):
    pen = make_pen(mocker)
    pen.configure_individual_feeding(None)
    formulate(pen)
    assert pen.ration == {1: 10.0, 2: 8.0}


def test_validation_of_mode():
    pen = object.__new__(IndividualFeedingPen)
    with pytest.raises(ValueError):
        pen.configure_individual_feeding("banana")
    with pytest.raises(ValueError):
        pen.configure_individual_feeding(BASE_CONCENTRATE)


def test_individual_mode_applies_each_cow_ration_and_averages(mocker):
    pen = make_pen(mocker)
    pen.configure_individual_feeding(INDIVIDUAL)
    rations = iter([{1: 6.0, 2: 12.0}, {1: 10.0, 2: 8.0}, {1: 14.0, 2: 4.0}])
    solver = MagicMock()
    solver.solve.side_effect = lambda req: RationSolution(True, 3.0, next(rations))
    mocker.patch.object(IndividualFeedingPen, "_cow_solver", return_value=solver)
    formulate(pen)
    assert pen.ration == {1: 10.0, 2: 8.0}
    assert {a.nutrition_supply.ration_formulation[1] for a in pen.animals_in_pen.values()} == {6.0, 10.0, 14.0}
    assert pen.average_nutrition_evaluation == "EVAL"
    assert pen.individual_feeding_report()["n_individual"] == 3


def test_failed_cow_falls_back_to_pen_ration(mocker):
    pen = make_pen(mocker, n=2)
    pen.configure_individual_feeding(INDIVIDUAL)
    results = iter([RationSolution(True, 3.0, {1: 4.0, 2: 4.0}), RationSolution(False, float("inf"))])
    solver = MagicMock()
    solver.solve.side_effect = lambda req: next(results)
    mocker.patch.object(IndividualFeedingPen, "_cow_solver", return_value=solver)
    formulate(pen)
    assert pen.animals_in_pen[1].nutrition_supply.ration_formulation == {1: 10.0, 2: 8.0}
    report = pen.individual_feeding_report()
    assert report["n_fallback_to_pen_ration"] == 1 and report["n_individual"] == 1


def test_solver_exception_keeps_pen_ration(mocker):
    pen = make_pen(mocker)
    pen.configure_individual_feeding(INDIVIDUAL)
    mocker.patch.object(IndividualFeedingPen, "_cow_solver", side_effect=RuntimeError("boom"))
    formulate(pen)
    assert pen.ration == {1: 10.0, 2: 8.0}


@pytest.mark.parametrize("combination,user", [(AnimalCombination.GROWING, False), (AnimalCombination.LAC_COW, True)])
def test_only_lactating_pens_without_user_ration_are_individualized(mocker, combination, user):
    pen = make_pen(mocker, combination=combination)
    pen.configure_individual_feeding(INDIVIDUAL)
    spy = mocker.patch.object(IndividualFeedingPen, "_individual_rations")
    formulate(pen, user=user)
    spy.assert_not_called()


def test_base_concentrate_mode_uses_plan(mocker):
    pen = make_pen(mocker, n=2)
    pen.configure_individual_feeding(BASE_CONCENTRATE, concentrate_feed_ids=[2])
    plan = SimpleNamespace(
        theta=0.4,
        base=SimpleNamespace(success=True),
        solutions=[
            SimpleNamespace(success=True, ration={1: 12.0, 2: 6.0}, cost=5.0),
            SimpleNamespace(success=True, ration={1: 8.0, 2: 10.0}, cost=6.0),
        ],
    )
    mocker.patch.object(mod, "RuFaSHybridSolver")
    mocker.patch.object(mod, "plan_base_plus_concentrate", return_value=plan)
    formulate(pen)
    assert pen.ration == {1: 10.0, 2: 8.0}
    report = pen.individual_feeding_report()
    assert report["theta"] == 0.4 and report["total_cost"] == 11.0
