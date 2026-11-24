"""Toy LP ration solver used to test and demo the cohort planner (fictitious feeds and prices)."""

import numpy as np
from scipy.optimize import linprog

from RUFAS.biophysical.animal.ration.lactation_cohorts import AdequacyReport, RationSolution

# name: (NEL Mcal/kg DM, MP g/kg DM, price R$/kg DM, max kg DM/cow/day, is_forage)
FEEDS = {
    "silagem de milho": (1.45, 55.0, 0.18, 14.0, True),
    "feno de alfafa": (1.25, 85.0, 0.50, 6.0, True),
    "milho moído": (2.05, 70.0, 0.60, 12.0, False),
    "farelo de soja": (1.95, 260.0, 1.60, 8.0, False),
    "polpa cítrica": (1.85, 50.0, 0.85, 6.0, False),
    "gordura protegida": (4.5, 0.0, 3.5, 0.6, False),
}


class ToyLPSolver:
    def __init__(self, min_forage_fraction: float = 0.40):
        self.names = list(FEEDS)
        self.E = np.array([FEEDS[n][0] for n in self.names])
        self.MP = np.array([FEEDS[n][1] for n in self.names])
        self.price = np.array([FEEDS[n][2] for n in self.names])
        self.ub = np.array([FEEDS[n][3] for n in self.names])
        self.forage = np.array([1.0 if FEEDS[n][4] else 0.0 for n in self.names])
        self.min_forage_fraction = min_forage_fraction

    def solve(self, req):
        dmi = req.dry_matter
        r = linprog(
            self.price,
            A_ub=np.vstack([-self.E, -self.MP, -self.forage]),
            b_ub=[-req.total_energy_requirement, -req.metabolizable_protein, -self.min_forage_fraction * dmi],
            A_eq=[np.ones(len(self.names))],
            b_eq=[dmi],
            bounds=[(0, u) for u in self.ub],
            method="highs",
        )
        if not r.success:
            return RationSolution(False, float("inf"))
        return RationSolution(True, float(r.fun), dict(zip(self.names, r.x.round(3))), detail=r.x)

    def check(self, sol, req):
        x = sol.detail
        dmi = float(np.sum(x))
        short = {}
        # densities: a cow that eats less/more than the ration's DMI still gets the same concentration
        e_dens, mp_dens = float(self.E @ x) / dmi, float(self.MP @ x) / dmi
        need_e = req.total_energy_requirement / req.dry_matter
        need_mp = req.metabolizable_protein / req.dry_matter
        if e_dens < need_e * (1 - 1e-6):
            short["energia"] = (need_e - e_dens) / need_e
        if mp_dens < need_mp * (1 - 1e-6):
            short["proteína metabolizável"] = (need_mp - mp_dens) / need_mp
        return AdequacyReport(not short, short)


class ToyHybridSolver(ToyLPSolver):
    """Toy solver for base + individual concentrate: concentrate feeds are the non-forage ones."""

    def __init__(self, min_forage_fraction: float = 0.40, max_concentrate: float = 8.0, intake_slack: float = 0.0):
        super().__init__(min_forage_fraction)
        self.intake_slack = intake_slack
        self.max_concentrate = max_concentrate
        self.is_conc = 1.0 - self.forage

    def solve_base(self, req):
        from RUFAS.biophysical.animal.ration.base_concentrate_planner import BaseSolution

        sol = self.solve(req)
        if not sol.success:
            return BaseSolution(False)
        x = np.asarray(sol.detail, dtype=float)
        return BaseSolution(True, float(self.price @ x) / req.dry_matter, dict(zip(self.names, x / req.dry_matter)))

    def solve_concentrate(self, req, base, index=0):
        """Variables v = [b, c_1..c_n]: x = b*m + c (b = kg of base eaten); total intake in [dmi, dmi*(1+slack)]."""
        from RUFAS.biophysical.animal.ration.base_concentrate_planner import ConcentrateSolution

        n, dmi = len(self.names), req.dry_matter
        m = np.array([base.fractions[k] for k in self.names])
        mat = np.hstack([m.reshape(-1, 1), np.eye(n)])  # x = mat @ v
        one = np.ones(n + 1)
        one[0] = 1.0
        total = np.concatenate([[1.0], np.ones(n)])  # total intake = b + sum(c)
        conc_row = np.concatenate([[0.0], np.ones(n)])
        a_ub = np.vstack(
            [
                -self.E @ mat,
                -self.MP @ mat,
                -self.forage @ mat + self.min_forage_fraction * total,
                mat,
                total,
                -total,
                conc_row,
            ]
        )
        b_ub = np.concatenate(
            [
                [-req.total_energy_requirement, -req.metabolizable_protein, 0.0],
                self.ub,
                [dmi * (1 + self.intake_slack), -dmi, self.max_concentrate],
            ]
        )
        cost = self.price @ mat
        r = linprog(
            cost,
            A_ub=a_ub,
            b_ub=b_ub,
            bounds=[(0, None)] + [(0, u * c) for u, c in zip(self.ub, self.is_conc)],
            method="highs",
        )
        if not r.success:
            return ConcentrateSolution(False)
        x = mat @ r.x
        return ConcentrateSolution(
            True, float(self.price @ x), dict(zip(self.names, r.x[1:].round(4))), dict(zip(self.names, x.round(4)))
        )
