"""RuFaS-backed solver for the base ration + individual concentrate model (see ``base_concentrate_planner``).

* ``solve_base`` runs the (warm-started) RuFaS optimizer on the **base feeds** only and returns fractions of dry matter.
* ``solve_concentrate`` chooses, for one cow, the concentrate amounts ``c`` (kg DM) that minimise cost using the RuFaS
  constraints themselves (energy, MP, Ca, P, NDF, fat, ...). The cow's total ration is ``(DMI - sum(c)) * m + c``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from RUFAS.biophysical.animal.data_types.nutrition_data_structures import NutritionRequirements
from RUFAS.biophysical.animal.ration.base_concentrate_planner import BaseSolution, ConcentrateSolution
from RUFAS.biophysical.animal.ration.cohort_ration_solver import RuFaSCohortSolver
from RUFAS.biophysical.animal.ration.ration_optimizer import RationConfig
from RUFAS.data_structures.feed_storage_to_animal_connection import RUFAS_ID, Feed


class RuFaSHybridSolver(RuFaSCohortSolver):
    def __init__(
        self,
        feeds: list[Feed],
        concentrate_feed_ids: Sequence[RUFAS_ID],
        base_feed_ids: Optional[Sequence[RUFAS_ID]] = None,
        max_concentrate: float = 8.0,
        violation_tolerance: float = 1e-3,
        starts: Sequence[tuple[float, float]] = ((0.25, 1.0), (0.0, 1.25)),
        **kwargs: Any,
    ) -> None:
        super().__init__(feeds, **kwargs)
        ids = [f.rufas_id for f in self.feeds]
        unknown = set(concentrate_feed_ids) - set(ids)
        if unknown:
            raise ValueError(f"concentrate feeds not available in the pen: {sorted(unknown)}")
        self.concentrate_ids = list(concentrate_feed_ids)
        self.base_ids = list(base_feed_ids) if base_feed_ids is not None else ids
        self.max_concentrate = max_concentrate
        self.violation_tolerance = violation_tolerance
        self.cow_contexts: list[tuple[float, Optional[float], Optional[float]]] = []  # per cow: (BW, CH4, urine N)
        self.starts = list(starts)  # (concentrate share, intake scale) initial points for SLSQP
        self._base_solver = RuFaSCohortSolver(
            [f for f in self.feeds if f.rufas_id in set(self.base_ids)],
            self.nutrient_standard,
            self.body_weight,
            self.methane,
            self.urine_nitrogen,
            self.optimizer,
            self.animal_combination,
            self.tolerance,
        )

    def solve_base(self, requirements: NutritionRequirements) -> BaseSolution:
        sol = self._base_solver.solve(requirements)
        if not sol.success:
            return BaseSolution(False)
        total = sum(sol.ration.values())
        fractions = {i: sol.ration.get(i, 0.0) / total for i in self.base_ids}
        price = {f.rufas_id: f.purchase_cost for f in self.feeds}
        return BaseSolution(True, sum(price[i] * q for i, q in fractions.items()), fractions)

    def solve_concentrate(
        self, requirements: NutritionRequirements, base: BaseSolution, index: int = 0
    ) -> ConcentrateSolution:
        """
        Variables: ``b`` (kg DM of base the cow eats) and ``c`` (kg DM of each concentrate). Ration ``x = b*m + c``.
        Total intake is not fixed: the RuFaS DMI constraints bound it, as in the pen formulation.
        """
        ids = [f.rufas_id for f in self.feeds]
        dmi = requirements.dry_matter
        m = np.array([base.fractions.get(i, 0.0) for i in ids])
        conc_idx = [ids.index(i) for i in self.concentrate_ids]
        k = len(conc_idx)

        def ration_of(v: NDArray[np.float64]) -> NDArray[np.float64]:
            x: NDArray[np.float64] = np.asarray(v[0] * m, dtype=float)
            x[conc_idx] += v[1:]
            return x

        # each cow is checked with its OWN body weight / methane / urine N (the pen average would let the optimizer
        # ride the constraint boundary and leave the real cow short)
        if self.cow_contexts:
            self.body_weight, self.methane, self.urine_nitrogen = self.cow_contexts[index]
        config: RationConfig = self._config(requirements)
        self.optimizer.set_constraints(config)
        price = np.asarray(config.price_list, dtype=float)
        lo, hi = np.asarray(config.feed_minimum_list, float), np.asarray(config.feed_maximum_list, float)
        rufas = list(self.optimizer._select_constraints(self.animal_combination))
        cons: list[dict[str, Any]] = [
            {"type": "ineq", "fun": (lambda v, f=con["fun"], a=con.get("args", ()): f(ration_of(v), *a))}
            for con in rufas
        ]
        cons += [
            {"type": "ineq", "fun": lambda v: self.max_concentrate - float(v[1:].sum())},
            {"type": "ineq", "fun": lambda v: hi - ration_of(v)},
            {"type": "ineq", "fun": lambda v: ration_of(v) - lo},
        ]
        bounds = [(0.0, 3.0 * dmi)] + [(0.0, self.max_concentrate)] * k
        best: Optional[OptimizeResult] = None
        for share, scale in self.starts:
            c0 = np.full(k, share * min(self.max_concentrate, dmi) / max(k, 1))
            start = np.concatenate([[max(dmi * scale - c0.sum(), 0.0)], c0])
            res = minimize(
                lambda v: float(price @ ration_of(v)), start, method="SLSQP", bounds=bounds, constraints=cons
            )
            violation = max((max(0.0, -float(np.min(con["fun"](res.x)))) for con in cons), default=0.0)
            if violation <= self.violation_tolerance and (best is None or res.fun < best.fun):
                best = res
        if best is None:
            return ConcentrateSolution(False)
        x = ration_of(best.x)
        return ConcentrateSolution(
            True,
            float(price @ x),
            {self.concentrate_ids[j]: float(best.x[1 + j]) for j in range(k)},
            {i: float(q) for i, q in zip(ids, x)},
        )
