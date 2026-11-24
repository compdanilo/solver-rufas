"""E7: numerical test of concavity of the nonlinear NASEM constraints (energy, metabolizable protein) in the decision vector."""
import json, sys
import numpy as np
from experiments.common import COMBO, LPWarmStartRationOptimizer, load

d = load()
opt = LPWarmStartRationOptimizer()
rng = np.random.default_rng(0)


def hessian(f, x, h):
    n = len(x); H = np.zeros((n, n)); f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ej = np.zeros(n); ei[i] = h[i]; ej[j] = h[j]
            v = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
            H[i, j] = H[j, i] = v
    return H


out = []
for ci in range(0, len(d["cows"]), 6):
    c = d["cows"][ci]
    req = c["req"]
    r, cfg = opt.attempt_optimization(nutrient_standard=d["standard"], pen_average_body_weight=c["bw"],
        pen_average_enteric_methane=c["methane"], pen_average_urine_nitrogen=c["urine_n"], requirements=req,
        initial_dry_matter_requirement=req.dry_matter, initial_protein_requirement=req.metabolizable_protein,
        pen_available_feeds=d["feeds"], animal_combination=COMBO)
    opt.set_constraints(cfg)
    cons = {k["fun"].__name__: k for k in opt._select_constraints(COMBO)}
    bnds = opt._build_bounds(cfg)
    ub = np.array([min(b[1], 25.0) if b[1] is not None else 25.0 for b in bnds]); lb = np.array([b[0] for b in bnds])
    res = {"cow": ci}
    for name in ["NASEM_net_energy_constraint", "protein_constraint_lower", "protein_constraint_upper"]:
        if name not in cons:
            continue
        k = cons[name]; f = lambda x, k=k: float(k["fun"](x, *k.get("args", ())))
        worst_max, worst_min, nviol = -1e9, 1e9, 0
        N = 150
        for _ in range(N):
            # random ration with total DMI in a realistic band
            x = rng.dirichlet(np.ones(len(lb))) * rng.uniform(12, 28)
            x = np.clip(x, lb + 1e-3, ub)
            h = np.maximum(1e-2, 0.02 * x)
            H = hessian(f, x, h)
            ev = np.linalg.eigvalsh(H)
            worst_max = max(worst_max, ev[-1]); worst_min = min(worst_min, ev[0])
        res[name] = {"max_eig": worst_max, "min_eig": worst_min}
    out.append(res); print(res, flush=True)
json.dump(out, open("experiments/results_e7.json", "w"), indent=1, default=float)
