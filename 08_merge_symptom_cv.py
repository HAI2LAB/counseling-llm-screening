# -*- coding: utf-8 -*-
"""fold별 metrics.json을 평균해 증상별 CV 성능 요약"""
import json
from pathlib import Path
from collections import defaultdict
import statistics as st

folds = []
for k in range(5):
    p = Path(f"runs/cvsym_fold{k}/metrics.json")
    if p.exists():
        folds.append(json.loads(p.read_text(encoding="utf-8")))
    else:
        print(f"[경고] fold{k} 없음")
assert folds
mac = [f["macro_f1"] for f in folds]
mic = [f["micro_f1"] for f in folds]
print(f"folds={len(folds)}  macro-F1 {st.mean(mac):.4f}±{st.stdev(mac):.4f}  "
      f"micro-F1 {st.mean(mic):.4f}±{st.stdev(mic):.4f}")
agg = defaultdict(lambda: {"F1": [], "P": [], "R": [], "n": 0})
for f in folds:
    for k, v in f["labels"].items():
        agg[k]["F1"].append(v["F1"]); agg[k]["P"].append(v["P"])
        agg[k]["R"].append(v["R"]); agg[k]["n"] += v["n"]
rows = sorted(agg.items(), key=lambda x: -st.mean(x[1]["F1"]))
print(f"\n{'증상':32s} {'F1(mean±sd)':16s} {'P':7s} {'R':7s} {'n(test합)'}")
for k, v in rows:
    print(f"{k:32s} {st.mean(v['F1']):.3f}±{st.stdev(v['F1']):.3f}   "
          f"{st.mean(v['P']):.3f}  {st.mean(v['R']):.3f}  {v['n']}")
