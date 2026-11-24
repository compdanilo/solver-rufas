"""Dumps per-cow solver results for the presentation front end (frontend/data.json)."""
import json
from dataclasses import replace
import numpy as np
from experiments.common import COMBO, LPWarmStartRationOptimizer, load, solve

d = load()
lp = LPWarmStartRationOptimizer()
feeds = d["feeds"]
pen = d["pen_ration"]
print(type(pen), pen if not hasattr(pen, "__dict__") else vars(pen))

NAMES = {23: "Farelo de sangue", 44: "Milho grão moído", 50: "Silagem de milho", 104: "Silagem de gramínea",
         110: "Silagem de leguminosa", 301: "Mistura mineral", 302: "Blend da fazenda"}
ids = [f.rufas_id for f in feeds]
price = [float(f.purchase_cost) for f in feeds]
xpen = np.array([float(pen[i]) for i in ids])


def g_at(cow, x, req=None):
    req = req or cow["req"]
    _, cfg = lp.attempt_optimization(
        nutrient_standard=d["standard"], pen_average_body_weight=cow["bw"], pen_average_enteric_methane=cow["methane"],
        pen_average_urine_nitrogen=cow["urine_n"], requirements=req, initial_dry_matter_requirement=req.dry_matter,
        initial_protein_requirement=req.metabolizable_protein, pen_available_feeds=feeds, animal_combination=COMBO)
    lp.set_constraints(cfg)
    cons = lp._select_constraints(COMBO)
    return {c["fun"].__name__: float(c["fun"](x, *c.get("args", ()))) for c in cons}


rows = []
for c in d["cows"]:
    r = solve(lp, d, c)
    req = c["req"]
    row = {"id": int(c["id"]), "dim": int(c["dim"]), "parity": int(c["parity"]), "bw": round(float(c["bw"]), 1),
           "req": {"energia": round(float(req.lactation_energy), 2), "pm": round(float(req.metabolizable_protein), 0),
                   "ca": round(float(req.calcium), 1), "p": round(float(req.phosphorus), 1),
                   "ms": round(float(req.dry_matter), 2)},
           "ok": bool(r["success"]), "g_pen": g_at(c, xpen * float(req.dry_matter) / xpen.sum()),
           "pen_cost": round(float(np.dot(price, xpen)) * float(req.dry_matter) / float(xpen.sum()), 3)}
    if r["success"]:
        e = solve(lp, d, c, req=replace(req, lactation_energy=req.lactation_energy + 1.0))
        p = solve(lp, d, c, req=replace(req, metabolizable_protein=req.metabolizable_protein + 100.0))
        row.update(x=[round(float(v), 3) for v in r["x"]], cost=round(float(r["cost"]), 3), g=r["g"], active=r["active"],
                   nit=r["nit"], lam_e=round(e["cost"] - r["cost"], 3) if e["success"] else None,
                   lam_mp=round(p["cost"] - r["cost"], 3) if p["success"] else None)
    rows.append(row)
    print(row["id"], row["ok"], flush=True)
import os
os.makedirs("frontend", exist_ok=True)
json.dump({"feeds": [{"id": i, "nome": NAMES[i], "preco": p} for i, p in zip(ids, price)],
           "pen": [round(float(v), 3) for v in xpen], "pen_cost": round(float(price @ xpen), 3) if False else round(float(np.dot(price, xpen)), 3),
           "cows": rows}, open("frontend/data.json", "w"), ensure_ascii=False)
