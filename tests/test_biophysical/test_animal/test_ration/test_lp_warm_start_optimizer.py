import numpy as np
from scipy.optimize import minimize

from RUFAS.biophysical.animal.ration.lp_warm_start_optimizer import (
    LPWarmStartRationOptimizer,
    lp_warm_start,
)
from RUFAS.biophysical.animal.ration.ration_optimizer import RationOptimizer

# Toy diet: corn silage, hay, corn, soybean meal. Nutrient content per kg DM.
PRICES = [0.10, 0.15, 0.30, 0.55]
ENERGY = np.array([1.35, 1.10, 1.95, 1.80])  # Mcal/kg
PROTEIN = np.array([70.0, 80.0, 60.0, 280.0])  # g MP/kg
BOUNDS = [(0.0, 15.0), (0.0, 6.0), (0.0, 8.0), (0.0, 4.0)]


def _constraints(nonlinear: bool = False) -> list[dict]:
    def energy(x, _=None):  # type: ignore[no-untyped-def]
        return float(ENERGY @ x) - 28.0

    def protein(x, _=None):  # type: ignore[no-untyped-def]
        return float(PROTEIN @ x) - 1900.0 + (0.5 * x[3] ** 2 if nonlinear else 0.0)

    def dmi_low(x, _=None):  # type: ignore[no-untyped-def]
        return float(x.sum()) - 19.0

    def dmi_up(x, _=None):  # type: ignore[no-untyped-def]
        return 23.0 - float(x.sum())

    return [{"type": "ineq", "fun": f, "args": (None,)} for f in (energy, protein, dmi_low, dmi_up)]


def test_warm_start_is_deterministic_and_feasible() -> None:
    cons = _constraints()
    x0 = np.full(4, 5.0)
    a = lp_warm_start(PRICES, BOUNDS, cons, x0)
    b = lp_warm_start(PRICES, BOUNDS, cons, x0)
    assert a.feasible
    assert np.allclose(a.x, b.x)


def test_warm_start_matches_exact_lp_on_linear_problem() -> None:
    from scipy.optimize import linprog

    cons = _constraints()
    exact = linprog(
        c=PRICES,
        A_ub=-np.vstack([ENERGY, PROTEIN, np.ones(4), -np.ones(4)]),
        b_ub=-np.array([28.0, 1900.0, 19.0, -23.0]),
        bounds=BOUNDS,
        method="highs",
    )
    warm = lp_warm_start(PRICES, BOUNDS, cons, np.full(4, 5.0), max_iterations=15)
    assert warm.feasible
    assert abs(warm.cost - exact.fun) < 1e-3


def test_warm_start_handles_nonlinear_constraint() -> None:
    cons = _constraints(nonlinear=True)
    warm = lp_warm_start(PRICES, BOUNDS, cons, np.full(4, 5.0), max_iterations=15)
    assert warm.feasible
    final = minimize(lambda x: float(np.dot(PRICES, x)), warm.x, method="SLSQP", bounds=BOUNDS, constraints=cons)
    assert final.success


def test_warm_start_reports_infeasible_problem() -> None:
    cons = _constraints() + [{"type": "ineq", "fun": lambda x, _=None: -1.0, "args": (None,)}]
    warm = lp_warm_start(PRICES, BOUNDS, cons, np.full(4, 5.0))
    assert not warm.feasible
    assert warm.max_violation >= 1.0 - 1e-6


def test_subclass_is_drop_in_replacement() -> None:
    assert issubclass(LPWarmStartRationOptimizer, RationOptimizer)
    assert LPWarmStartRationOptimizer().last_warm_start is None


def test_attempt_optimization_passes_warm_start_to_slsqp(mocker) -> None:  # type: ignore[no-untyped-def]
    """The SLSQP solve must receive the LP warm start (not a random point) as its starting vector."""
    from unittest.mock import MagicMock

    optimizer = LPWarmStartRationOptimizer()
    config = MagicMock()
    config.price_list = PRICES
    config.initial_dry_matter_requirement = 20.0
    mocker.patch("RUFAS.biophysical.animal.ration.lp_warm_start_optimizer.RationConfig", return_value=config)
    mocker.patch.object(optimizer, "_build_bounds", return_value=BOUNDS)
    mocker.patch.object(optimizer, "set_constraints")
    mocker.patch.object(optimizer, "_select_constraints", return_value=_constraints())
    minimize_mock = mocker.patch(
        "RUFAS.biophysical.animal.ration.lp_warm_start_optimizer.minimize", return_value=MagicMock()
    )
    feeds = [MagicMock(rufas_id=i) for i in range(4)]

    def run() -> np.ndarray:
        optimizer.attempt_optimization(
            nutrient_standard=MagicMock(),
            pen_average_body_weight=600.0,
            pen_average_enteric_methane=0.0,
            pen_average_urine_nitrogen=0.0,
            requirements=MagicMock(),
            initial_dry_matter_requirement=20.0,
            initial_protein_requirement=1.0,
            pen_available_feeds=feeds,
            animal_combination=MagicMock(),
        )
        return minimize_mock.call_args[0][1]

    first, second = run(), run()
    assert np.allclose(first, second)  # deterministic, unlike the stock random start
    assert optimizer.last_warm_start is not None and optimizer.last_warm_start.feasible
    assert np.allclose(first, optimizer.last_warm_start.x)


def test_attempt_optimization_survives_warm_start_failure(mocker) -> None:  # type: ignore[no-untyped-def]
    """If the LP step raises, formulation still proceeds from the deterministic fallback start."""
    from unittest.mock import MagicMock

    optimizer = LPWarmStartRationOptimizer()
    config = MagicMock()
    config.price_list = PRICES
    mocker.patch("RUFAS.biophysical.animal.ration.lp_warm_start_optimizer.RationConfig", return_value=config)
    mocker.patch.object(optimizer, "_build_bounds", return_value=BOUNDS)
    mocker.patch.object(optimizer, "set_constraints")
    mocker.patch.object(optimizer, "_select_constraints", return_value=[])
    mocker.patch("RUFAS.biophysical.animal.ration.lp_warm_start_optimizer.lp_warm_start", side_effect=RuntimeError)
    minimize_mock = mocker.patch(
        "RUFAS.biophysical.animal.ration.lp_warm_start_optimizer.minimize", return_value=MagicMock()
    )
    config.initial_dry_matter_requirement = 20.0
    optimizer.attempt_optimization(
        nutrient_standard=MagicMock(),
        pen_average_body_weight=600.0,
        pen_average_enteric_methane=0.0,
        pen_average_urine_nitrogen=0.0,
        requirements=MagicMock(),
        initial_dry_matter_requirement=20.0,
        initial_protein_requirement=1.0,
        pen_available_feeds=[MagicMock(rufas_id=i) for i in range(4)],
        animal_combination=MagicMock(),
    )
    assert optimizer.last_warm_start is None
    assert minimize_mock.called
