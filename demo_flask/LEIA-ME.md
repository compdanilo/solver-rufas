# Demonstração Flask do solver de ração

Na raiz do repositório RuFaS (com as dependências do RuFaS instaladas):

    pip install flask
    python -m demo_flask.app

Abra http://127.0.0.1:5000

- Aba "Demonstração": roda o código real do trabalho (`lactation_cohorts`, `base_concentrate_planner`) com o
  resolvedor linear de demonstração (`toy_solver`, alimentos e preços fictícios).
- Aba "Simulação real": resultados guardados da simulação do RuFaS com o solver completo (`data/*.jsonl`).
  Para refazer: `python compare_run.py res.jsonl -- -g -c -s -v errors -p <task_manager.json> -o out -l logs`
  (as entradas estão em `simulacao_real/entradas`).

## Padrão nutricional e escopo

- O solver define a ração diária de uma vaca. Os modos por coorte e por base com concentrado são extensões dele para um curral.
- Na simulação real (aba "Simulação real"), as exigências seguem o NASEM 2021, no padrão `nutrient_standard: NASEM` do RuFaS.
- A aba de demonstração usa um solver simplificado com alimentos e preços fictícios. Ela não reproduz o NASEM.
