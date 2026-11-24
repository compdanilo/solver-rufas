"""Validation experiments for the per-cow ration solver (real RuFaS objects, NASEM 2021).

Usage: PYTHONPATH=. python experiments/run_experiments.py STAGE [STAGE ...]   (stages: e1 e2 e3 e5)
Results go to experiments/results_<stage>.json
"""
import json
import statistics as st
import sys
from dataclasses import replace

import numpy as np
from scipy.optimize import linprog

from experiments.common import COMBO, LPWarmStartRationOptimizer, RationOptimizer, load, solve
from RUFAS.biophysical.animal.ration.lp_warm_start_optimizer import _linearize

SEEDS = [0, 1, 2, 3, 4]


def dump(name, obj):
    with open(f"experiments/results_{name}.json", "w") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=float)


def e1_e4(d):
    """E1: LP warm start + SLSQP vs stock SLSQP (random start). E4: dispersion of costs over random starts."""
    lp, stock = LPWarmStartRationOptimizer(), RationOptimizer()
    rows = []
    for c in d["cows"]:
        a = solve(lp, d, c)
        b = [solve(stock, d, c, seed=s) for s in SEEDS]
        rows.append({"id": c["id"], "lp": {k: a[k] for k in ("success", "cost", "time", "nit", "violation")},
                     "stock": [{k: r[k] for k in ("success", "cost", "time", "nit", "violation")} for r in b]})
    n = len(rows)
    lp_ok = [r for r in rows if r["lp"]["success"]]
    stock_runs = [x for r in rows for x in r["stock"]]
    both = [(r["lp"]["cost"], [x["cost"] for x in r["stock"] if x["success"]]) for r in lp_ok]
    rel = [(min(cs) - lc) / lc * 100 for lc, cs in both if cs]
    disp = []
    for r in lp_ok:
        ok = [x["cost"] for x in r["stock"] if x["success"]]
        if len(ok) >= 2:
            disp.append({"id": r["id"], "n_ok": len(ok), "min": min(ok), "max": max(ok), "std": st.pstdev(ok),
                         "lp": r["lp"]["cost"], "spread_pct": (max(ok) - min(ok)) / r["lp"]["cost"] * 100})
    out = {
        "n_cows": n, "seeds": len(SEEDS),
        "lp_success": len(lp_ok), "lp_success_rate": len(lp_ok) / n,
        "stock_success_runs": sum(x["success"] for x in stock_runs), "stock_runs": len(stock_runs),
        "stock_success_rate": sum(x["success"] for x in stock_runs) / len(stock_runs),
        "stock_all_seeds_ok": sum(all(x["success"] for x in r["stock"]) for r in rows),
        "stock_any_ok": sum(any(x["success"] for x in r["stock"]) for r in rows),
        "stock_ok_where_lp_failed": sum(any(x["success"] for x in r["stock"]) for r in rows if not r["lp"]["success"]),
        "lp_ok_where_stock_all_failed": sum(not any(x["success"] for x in r["stock"]) for r in lp_ok),
        "time_lp_mean_s": st.mean(r["lp"]["time"] for r in rows),
        "time_stock_mean_s": st.mean(x["time"] for x in stock_runs),
        "nit_lp_mean": st.mean(r["lp"]["nit"] for r in rows),
        "nit_stock_mean": st.mean(x["nit"] for x in stock_runs),
        "best_stock_vs_lp_cost_pct_median": st.median(rel) if rel else None,
        "best_stock_vs_lp_cost_pct_max": max(rel) if rel else None,
        "best_stock_vs_lp_cost_pct_min": min(rel) if rel else None,
        "dispersion_cows": len(disp),
        "spread_pct_median": st.median(x["spread_pct"] for x in disp) if disp else None,
        "spread_pct_max": max(x["spread_pct"] for x in disp) if disp else None,
        "std_median": st.median(x["std"] for x in disp) if disp else None,
        "rows": rows, "dispersion": disp,
    }
    dump("e1", out)
    print({k: v for k, v in out.items() if k not in ("rows", "dispersion")}, flush=True)


def e2(d):
    """E2: active constraints and shadow prices by re-solving with a perturbed requirement."""
    lp = LPWarmStartRationOptimizer()
    rows = []
    for c in d["cows"]:
        base = solve(lp, d, c)
        if not base["success"]:
            continue
        req = c["req"]
        e = solve(lp, d, c, req=replace(req, lactation_energy=req.lactation_energy + 1.0))
        p = solve(lp, d, c, req=replace(req, metabolizable_protein=req.metabolizable_protein + 100.0))
        rows.append({
            "id": c["id"], "cost": base["cost"], "active": base["active"],
            "d_cost_per_mcal": (e["cost"] - base["cost"]) if e["success"] else None,
            "d_cost_per_100g_mp": (p["cost"] - base["cost"]) if p["success"] else None,
        })
    names = {}
    for r in rows:
        for a in r["active"]:
            names[a] = names.get(a, 0) + 1
    ec = [r["d_cost_per_mcal"] for r in rows if r["d_cost_per_mcal"] is not None]
    pc = [r["d_cost_per_100g_mp"] for r in rows if r["d_cost_per_100g_mp"] is not None]
    en_active = [r["d_cost_per_mcal"] for r in rows if "NASEM_net_energy_constraint" in r["active"] and r["d_cost_per_mcal"] is not None]
    mp_active = [r["d_cost_per_100g_mp"] for r in rows if "protein_constraint_lower" in r["active"] and r["d_cost_per_100g_mp"] is not None]
    out = {"n_feasible": len(rows), "active_counts": names,
           "energy_price_all": {"n": len(ec), "median": st.median(ec), "mean": st.mean(ec), "min": min(ec), "max": max(ec)},
           "energy_price_when_active": {"n": len(en_active), "median": st.median(en_active) if en_active else None},
           "mp_price_all": {"n": len(pc), "median": st.median(pc), "mean": st.mean(pc), "min": min(pc), "max": max(pc)},
           "mp_price_when_active": {"n": len(mp_active), "median": st.median(mp_active) if mp_active else None},
           "rows": rows}
    dump("e2", out)
    print({k: v for k, v in out.items() if k != "rows"}, flush=True)


def e3(d):
    """E3: LP from linearizing NASEM constraints at the average cow's ration vs the exact nonlinear solve."""
    lp = LPWarmStartRationOptimizer()
    avg = {"bw": d["pen_avg_bw"], "methane": d["pen_methane"], "urine_n": d["pen_urine_n"], "req": d["pen_avg_requirements"]}
    ref = solve(lp, d, avg)
    x_ref = ref["x"]
    rows = []
    for c in d["cows"]:
        exact = solve(lp, d, c)
        # constraints and bounds for this cow, then linearize at x_ref
        probe = solve(lp, d, c)  # (re-run to get cfg-bound constraints below)
        _, cfg = lp.attempt_optimization(
            nutrient_standard=d["standard"], pen_average_body_weight=c["bw"], pen_average_enteric_methane=c["methane"],
            pen_average_urine_nitrogen=c["urine_n"], requirements=c["req"],
            initial_dry_matter_requirement=c["req"].dry_matter, initial_protein_requirement=c["req"].metabolizable_protein,
            pen_available_feeds=d["feeds"], animal_combination=COMBO)
        lp.set_constraints(cfg)
        cons = lp._select_constraints(COMBO)
        bounds = lp._build_bounds(cfg)
        g0, jac = _linearize(cons, x_ref, 1e-4)
        res = linprog(c=np.asarray(cfg.price_list, float), A_ub=-jac, b_ub=g0 - jac @ x_ref,
                      bounds=[(lo, None if np.isinf(hi) else hi) for lo, hi in bounds], method="highs")
        if not res.success:
            rows.append({"id": c["id"], "lp_feasible": False, "exact_success": exact["success"]})
            continue
        x = res.x
        g = {k["fun"].__name__: float(k["fun"](x, *k.get("args", ()))) for k in cons}
        worst = min(g, key=g.get)
        rows.append({"id": c["id"], "lp_feasible": True, "exact_success": exact["success"],
                     "lp_cost": float(res.fun), "exact_cost": exact["cost"] if exact["success"] else None,
                     "violation": max(0.0, -g[worst]), "worst": worst,
                     "energy_g": g.get("NASEM_net_energy_constraint"), "mp_g": g.get("protein_constraint_lower")})
    ok = [r for r in rows if r["lp_feasible"] and r["exact_success"]]
    viol = [r for r in ok if r["violation"] > 1e-3]
    out = {"n_cows": len(rows), "lp_infeasible": sum(not r["lp_feasible"] for r in rows),
           "compared": len(ok), "lp_violates_true_constraints": len(viol),
           "lp_violates_energy": sum(1 for r in ok if r["energy_g"] is not None and r["energy_g"] < -1e-3),
           "violation_median_when_violating": st.median(r["violation"] for r in viol) if viol else 0,
           "violation_max": max((r["violation"] for r in ok), default=0),
           "cost_gap_pct_median": st.median((r["lp_cost"] - r["exact_cost"]) / r["exact_cost"] * 100 for r in ok),
           "cost_gap_pct_min": min((r["lp_cost"] - r["exact_cost"]) / r["exact_cost"] * 100 for r in ok),
           "cost_gap_pct_max": max((r["lp_cost"] - r["exact_cost"]) / r["exact_cost"] * 100 for r in ok),
           "worst_counts": {w: sum(1 for r in viol if r["worst"] == w) for w in {r["worst"] for r in viol}},
           "rows": rows}
    dump("e3", out)
    print({k: v for k, v in out.items() if k != "rows"}, flush=True)




def e5(d):
    """E5: total cost vs theta for the base + concentrate mode, and sensitivity to the grid size."""
    from RUFAS.biophysical.animal.ration.base_concentrate_planner import _evaluate, search_base_density
    from RUFAS.biophysical.animal.ration.hybrid_ration_solver import RuFaSHybridSolver

    h = RuFaSHybridSolver(d["feeds"], [44, 23], None, 6.0, nutrient_standard=d["standard"],
                          pen_average_body_weight=d["pen_avg_bw"], pen_average_enteric_methane=d["pen_methane"],
                          pen_average_urine_nitrogen=d["pen_urine_n"], optimizer=LPWarmStartRationOptimizer())
    h.cow_contexts = [(c["bw"], c["methane"], c["urine_n"]) for c in d["cows"]]
    reqs = [c["req"] for c in d["cows"]]
    cache = {}
    curve = []
    for i in range(41):
        t = i / 40
        p = _evaluate(t, reqs, h, cache)
        curve.append({"theta": t, "total": p.total_cost if p.total_cost != float("inf") else None,
                      "base_feasible": p.total_cost != float("inf"), "n_infeasible": p.n_infeasible,
                      "cost_feasible": (p.total_cost - 1000 * p.n_infeasible) if p.total_cost != float("inf") else None})
    grids = []
    for n, r in [(3, 4), (5, 4), (9, 4), (21, 0)]:
        plan = search_base_density(reqs, h, n_grid=n, refine_iterations=r)
        grids.append({"n_grid": n, "refine": r, "theta": plan.theta, "total": plan.total_cost,
                      "n_infeasible": plan.n_infeasible, "evaluations": len(plan.curve)})
    fin = [c for c in curve if c["total"] is not None]
    totals = [c["total"] for c in fin]
    # unimodality on the grid with feasible base: no local minimum other than the global one
    minima = [i for i in range(len(totals)) if (i == 0 or totals[i] <= totals[i - 1]) and (i == len(totals) - 1 or totals[i] <= totals[i + 1])]
    out = {"n_cows": len(reqs), "curve": curve, "grids": grids, "n_feasible_thetas": len(fin),
           "local_minima_indices_on_feasible_grid": minima,
           "monotone_nondecreasing": all(totals[i] <= totals[i + 1] for i in range(len(totals) - 1))}
    dump("e5", out)
    print({k: v for k, v in out.items() if k != "curve"}, flush=True)
    print([(round(c["theta"], 3), c["total"] and round(c["total"], 1), c["n_infeasible"]) for c in curve], flush=True)


# prices on the same scale as the captured menu (corn 0.5, mineral 1.0); ASSUMED, not market data
from experiments.common import PRICES_BRL
EXTRA = {167: PRICES_BRL[167], 31: PRICES_BRL[31], 163: PRICES_BRL[163]}


def extra_feeds(d):
    import pandas as pd
    from RUFAS.data_structures.feed_storage_to_animal_connection import (
        FeedCategorization, FeedComponentType, MeasurementUnits, NASEMFeed)
    df = pd.read_csv("input/data/feed/NASEM_Comp_with_TDN_urea.csv")
    out = []
    for rid, price in EXTRA.items():
        row = df[df["rufas_id"] == rid].iloc[0].to_dict()
        row.pop("rufas_id")
        row["feed_type"] = FeedComponentType(row["feed_type"])
        row["Fd_Category"] = FeedCategorization(row["Fd_Category"])
        row["units"] = MeasurementUnits(row["units"])
        out.append(NASEMFeed(rufas_id=rid, amount_available=0.0, on_farm_cost=price * 0.01, purchase_cost=price,
                             buffer=0.15, **row))
    return out


def e6(d):
    """E6: add 3 feeds to the menu; re-solve every cow, report feasibility and cost change."""
    lp = LPWarmStartRationOptimizer()
    feeds2 = sorted(d["feeds"] + extra_feeds(d), key=lambda f: f.rufas_id)
    rows = []
    for i, c in enumerate(d["cows"]):
        a = solve(lp, d, c)
        b = solve(lp, d, c, feeds=feeds2)
        rows.append({"i": i, "dim": c["dim"], "ok7": a["success"], "ok10": b["success"], "c7": a["cost"],
                     "c10": b["cost"], "viol7": a["violation"], "viol10": b["violation"],
                     "used_extra": {f.rufas_id: float(x) for f, x in zip(feeds2, b["x"]) if f.rufas_id in EXTRA}})
    n7 = sum(r["ok7"] for r in rows)
    n10 = sum(r["ok10"] for r in rows)
    both = [r for r in rows if r["ok7"] and r["ok10"]]
    gains = [100 * (r["c10"] - r["c7"]) / r["c7"] for r in both if r["c7"] > 0]
    out = {"n_cows": len(rows), "ok_7_feeds": n7, "ok_10_feeds": n10, "recovered": sum(1 for r in rows if not r["ok7"] and r["ok10"]),
           "lost": sum(1 for r in rows if r["ok7"] and not r["ok10"]),
           "median_cost_change_pct_both_ok": st.median(gains) if gains else None,
           "mean_cost_change_pct_both_ok": st.mean(gains) if gains else None,
           "extra_prices": EXTRA, "rows": rows}
    dump("e6", out)
    print({k: v for k, v in out.items() if k != "rows"}, flush=True)


if __name__ == "__main__":
    d = load()
    for stage in sys.argv[1:]:
        {"e1": e1_e4, "e2": e2, "e3": e3, "e5": e5, "e6": e6}[stage](d)
