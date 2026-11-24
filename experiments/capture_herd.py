"""Captures the lactating pen at its first ration formulation (real RuFaS objects) into a pickle for offline experiments.

Usage: python experiments/capture_herd.py OUT.pkl -- <RUFAS main arguments>
"""
import os
import pickle
import runpy
import sys

from RUFAS.biophysical.animal.data_types.animal_combination import AnimalCombination
from RUFAS.biophysical.animal.pen import Pen

out, args = sys.argv[1], sys.argv[3:]
original = Pen.formulate_optimized_ration
done = {"ok": False, "n": 0}
WANT = int(os.environ.get("CAPTURE_N", "1"))


def hook(self, is_ration_defined_by_user, pen_available_feeds, *rest, **kw):
    original(self, is_ration_defined_by_user, pen_available_feeds, *rest, **kw)
    if done["ok"] or self.animal_combination is not AnimalCombination.LAC_COW or len(self.animals_in_pen) < 20:
        return
    done["n"] += 1
    if done["n"] < WANT:
        return
    animals = self.animals_in_pen
    first = next(iter(animals.values()))
    nasem_avg = self._calculate_pen_nasem_averages()
    data = {
        "feeds": pen_available_feeds,
        "standard": first.nutrient_standard,
        "pen_ration": dict(self.ration),
        "pen_avg_bw": self.average_body_weight,
        "pen_methane": nasem_avg[0],
        "pen_urine_n": nasem_avg[1],
        "pen_avg_requirements": self.average_nutrition_requirements,
        "cows": [
            {
                "id": i,
                "dim": getattr(a, "days_in_milk", None),
                "parity": getattr(a, "calves", None),
                "bw": a.body_weight,
                "methane": a.enteric_methane,
                "urine_n": a.digestive_system.manure_excretion.urine_nitrogen,
                "req": a.nutrition_requirements,
            }
            for i, a in animals.items()
        ],
    }
    with open(out, "wb") as fh:
        pickle.dump(data, fh)
    print(f"CAPTURED {len(data['cows'])} cows, {len(pen_available_feeds)} feeds -> {out}", flush=True)
    done["ok"] = True
    os._exit(0)


Pen.formulate_optimized_ration = hook
sys.argv = ["main.py"] + args
runpy.run_path("main.py", run_name="__main__")
