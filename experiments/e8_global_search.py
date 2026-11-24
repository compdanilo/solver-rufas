"""E8: independent global search (differential evolution, constraint-handling by scipy) vs LP->SLSQP, on a sample of cows."""
import json, time, sys
import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from experiments.common import COMBO, LPWarmStartRationOptimizer, load, solve

d = load(); opt = LPWarmStartRationOptimizer()
rows = []
SH, NSH = int(sys.argv[1]), int(sys.argv[2])
ids = [i for i in range(len(d["cows"])) if i % NSH == SH]
for ci in ids:
    c = d["cows"][ci]; a = solve(opt, d, c)
    if not a["success"]:
        continue
    req = c["req"]
    r, cfg = opt.attempt_optimization(nutrient_standard=d["standard"], pen_average_body_weight=c["bw"],
        pen_average_enteric_methane=c["methane"], pen_average_urine_nitrogen=c["urine_n"], requirements=req,
        initial_dry_matter_requirement=req.dry_matter, initial_protein_requirement=req.metabolizable_protein,
        pen_available_feeds=d["feeds"], animal_combination=COMBO)
    opt.set_constraints(cfg)
    cons = opt._select_constraints(COMBO)
    bnds = [(b[0], min(b[1], 40.0)) for b in opt._build_bounds(cfg)]
    price = np.array([f.purchase_cost for f in d["feeds"]])
    ncs = [NonlinearConstraint(lambda x, k=k: float(k["fun"](x, *k.get("args", ()))), 0.0, np.inf) for k in cons]
    t = time.time()
    best = None
    for seed in range(2):
        def pen(x):
            v = sum(max(0.0, -float(k["fun"](x, *k.get("args", ())))) for k in cons)
            return float(price @ x) + 1e3 * v
        res = differential_evolution(pen, bnds, seed=seed, popsize=30, maxiter=500, tol=1e-12, polish=False,
                                     init="sobol", updating="deferred", mutation=(0.4, 1.0), recombination=0.8)
        # polish the DE point with the exact-constraint SLSQP (what a global-then-local scheme does)
        from scipy.optimize import minimize
        rr = minimize(lambda x: float(price @ x), res.x, jac=lambda x: price, bounds=bnds, constraints=cons, method="SLSQP",
                      options={"maxiter": 300, "ftol": 1e-12})
        viol = max(0.0, -min(float(k["fun"](rr.x, *k.get("args", ()))) for k in cons))
        if viol < 1e-4 and (best is None or rr.fun < best):
            best = float(rr.fun)
    row = {"cow": ci, "slsqp_cost": a["cost"], "de_best_cost": best, "diff_pct": None if best is None else 100 * (best - a["cost"]) / a["cost"], "t": time.time() - t}
    rows.append(row); print(row, flush=True)
json.dump(rows, open(f"experiments/results_e8_{SH}.json", "w"), indent=1, default=float)
