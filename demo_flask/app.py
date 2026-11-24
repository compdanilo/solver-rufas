"""Demonstração em Flask do solver de ração (código Python real do trabalho).

Rodar, na raiz do repositório RuFaS:
    pip install flask
    python -m demo_flask.app          # abre em http://127.0.0.1:5000

O que roda aqui: os módulos reais ``lactation_cohorts`` (coortes, DP, plano individual) e
``base_concentrate_planner`` (base + concentrado, busca em θ). O resolvedor de ração é o de demonstração
(``toy_solver``: PL com HiGHS, alimentos e preços fictícios), para responder em segundos.
A aba "Simulação real" mostra os resultados guardados da simulação do RuFaS com o solver completo.
"""

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from demo_cohorts import make_herd
from RUFAS.biophysical.animal.ration.base_concentrate_planner import plan_base_plus_concentrate
from RUFAS.biophysical.animal.ration.lactation_cohorts import (
    build_cohorts,
    compare_plans,
    plan_individual,
    underfed_animals,
)
from tests.test_biophysical.test_animal.test_ration.toy_solver import FEEDS, ToyHybridSolver, ToyLPSolver

DATA = Path(__file__).parent / "data"
app = Flask(__name__)


def _num(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(request.args.get(name, default))
    except ValueError:
        value = default
    return max(low, min(high, value))


@app.get("/")
def index() -> str:
    return render_template("index.html", feeds=FEEDS)


@app.get("/api/run")
def run():
    n, seed = int(_num("n", 120, 20, 300)), int(_num("seed", 7, 1, 9999))
    min_forage, max_conc = _num("min_forage", 40, 0, 80) / 100, _num("max_conc", 8, 0, 20)
    herd = make_herd(n, seed)
    cohorts = build_cohorts(herd)
    solver = ToyLPSolver(min_forage)

    plans = compare_plans(cohorts, solver, max_groups=len(cohorts))
    cohort_plans = [
        {
            "estrategia": p.strategy,
            "grupos": len(p.groups),
            "custo": round(p.total_cost, 2),
            "vacas_em_deficit": underfed_animals(p, solver),
            "viavel": p.feasible,
        }
        for p in plans
    ]
    individual = plan_individual(cohorts, solver)
    cows = []
    for cow in herd:
        sol = solver.solve(cow.nutrition_requirements)
        cows.append(
            {
                "lactacao": "1ª" if cow.calves == 1 else f"{cow.calves}ª",
                "dim": cow.days_in_milk,
                "ms": round(cow.nutrition_requirements.dry_matter, 1),
                "energia_por_kg": round(
                    cow.nutrition_requirements.total_energy_requirement / cow.nutrition_requirements.dry_matter, 2
                ),
                "custo": round(sol.cost_per_animal, 2) if sol.success else None,
                "racao": {k: float(v) for k, v in sol.ration.items() if v > 0.01} if sol.success else {},
            }
        )
    hybrid = plan_base_plus_concentrate(
        [c.nutrition_requirements for c in herd], ToyHybridSolver(min_forage, max_conc), n_grid=11
    )
    return jsonify(
        {
            "n_vacas": n,
            "coortes": [
                {"nome": c.key.label, "vacas": c.n_animals, "dim_medio": round(c.mean_dim),
                 "mcal_por_kg": round(c.energy_density, 2)}
                for c in cohorts
            ],
            "planos": cohort_plans,
            "individual": {"custo": round(individual.total_cost, 2), "vacas_em_deficit": underfed_animals(individual, solver)},
            "hibrido": {
                "theta": round(hybrid.theta, 3),
                "custo": round(hybrid.total_cost, 2),
                "vacas_sem_solucao": hybrid.n_infeasible,
                "concentrado_total_kg": round(hybrid.total_concentrate, 1),
                "curva": [[round(t, 3), None if c == float("inf") else round(c, 1)] for t, c in hybrid.curve],
            },
            "vacas": cows,
        }
    )


@app.get("/api/real")
def real():
    out = {}
    for name in ("base", "ind", "hyb"):
        path = DATA / f"res_{name}.jsonl"
        out[name] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    return jsonify(out)


if __name__ == "__main__":
    app.run(debug=False)
