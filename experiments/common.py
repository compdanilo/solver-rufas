"""Shared helpers for the solver experiments (real RuFaS objects captured from a simulation)."""
import pickle
import random
import time
from dataclasses import replace

import numpy as np

from RUFAS.biophysical.animal.data_types.animal_combination import AnimalCombination
from RUFAS.biophysical.animal.nutrients.nutrition_supply_calculator import NutritionSupplyCalculator
from RUFAS.biophysical.animal.ration.lp_warm_start_optimizer import LPWarmStartRationOptimizer
from RUFAS.biophysical.animal.ration.ration_optimizer import RationOptimizer
from RUFAS.biophysical.animal.animal import Animal

COMBO = AnimalCombination.LAC_COW


# R$/kg DM. CEPEA/Conab/Agrolink quotes of 19-24/11/2026 where available, otherwise ESTIMATES (see experiments/README_precos.md)
PRICES_BRL = {44: 1.298, 167: 2.263, 31: 1.77, 163: 1.05, 23: 4.50, 50: 0.55, 104: 0.45, 110: 0.50, 301: 4.00, 302: 0.50}


def reprice(feeds):
    out = []
    for f in feeds:
        p = PRICES_BRL.get(f.rufas_id)
        out.append(replace(f, purchase_cost=p, on_farm_cost=p) if p is not None else f)
    return out


def load(path="experiments/herd2.pkl", brl=True):
    d = pickle.load(open(path, "rb"))
    if brl:
        d["feeds"] = reprice(d["feeds"])
    NutritionSupplyCalculator.nutrient_standard = d["standard"]
    Animal.set_nutrient_standard(d["standard"])
    return d


def solve(opt, d, cow, feeds=None, seed=0, req=None):
    """One optimizer run for one cow. Returns dict(success, cost, x, nit, time, violation, active)."""
    feeds = feeds or d["feeds"]
    req = req or cow["req"]
    random.seed(seed)
    t = time.perf_counter()
    r, cfg = opt.attempt_optimization(
        nutrient_standard=d["standard"],
        pen_average_body_weight=cow["bw"],
        pen_average_enteric_methane=cow["methane"],
        pen_average_urine_nitrogen=cow["urine_n"],
        requirements=req,
        initial_dry_matter_requirement=req.dry_matter,
        initial_protein_requirement=req.metabolizable_protein,
        pen_available_feeds=feeds,
        animal_combination=COMBO,
    )
    dt = time.perf_counter() - t
    x = np.asarray(r.x, dtype=float)
    opt.set_constraints(cfg)
    cons = opt._select_constraints(COMBO)
    g = {c["fun"].__name__: float(c["fun"](x, *c.get("args", ()))) for c in cons}
    viol = max(0.0, -min(g.values()))
    active = sorted(k for k, v in g.items() if abs(v) < 1e-3)
    return {"success": bool(r.success), "cost": float(r.fun), "x": x, "nit": int(r.nit), "time": dt,
            "violation": viol, "active": active, "g": g, "msg": str(r.message)}
