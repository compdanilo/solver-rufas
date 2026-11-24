# Preços usados nos experimentos (R$ por kg de matéria seca)

| rufas_id | Alimento | R$/kg MS | Tipo | Base |
|---|---|---|---|---|
| 44 | Milho grão moído | 1,298 | cotação | CEPEA/Esalq milho, 19/11/2026: R$ 67,68/saca 60 kg = R$ 1,128/kg; / 0,869 de MS |
| 167 | Farelo de soja 48 | 2,263 | cotação | Agrolink, Rio Verde (GO), 24/11/2026: R$ 2.020/t; / 0,893 de MS (referencial, sem auditoria) |
| 31 | Farelo de canola | 1,77 | estimado | ~78% do preço do farelo de soja |
| 163 | Casca de soja | 1,05 | estimado | ~80% do preço do milho |
| 23 | Farelo de sangue | 4,50 | estimado | ~2x o farelo de soja |
| 50 | Silagem de milho | 0,55 | estimado | custo de produção típico |
| 104 | Silagem gramínea/leguminosa | 0,45 | estimado | idem |
| 110 | Silagem de leguminosa | 0,50 | estimado | idem |
| 301 | Mistura mineral | 4,00 | estimado | ordem de grandeza de mercado |
| 302 | Blend da fazenda | 0,50 | estimado | idem |

Fontes: CEPEA/Esalq via Canal Rural (21/11/2026); Agrolink (24/11/2026). Preços de ureia e caroço de algodão
(Conab/Agrolink) não entram no cardápio. Os itens "estimado" são hipóteses do trabalho, não cotações.
Resultados com os preços fictícios do exemplo do RuFaS ficam em `results_*_exemplo.json`.
