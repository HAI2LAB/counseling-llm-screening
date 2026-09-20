# -*- coding: utf-8 -*-
"""실험③ k회기 곡선 (v4 확정: 피험자=subject, 회기=id의 그룹명 뒤 첫 숫자)
   id 예: resource_depression_1_check_D073 → 회기 1, 피험자 D073"""
import json, re, collections, sys
from sklearn.metrics import f1_score, accuracy_score

PRED = sys.argv[1] if len(sys.argv) > 1 else "runs/cv14b_final/all_predictions.jsonl"
rows = [json.loads(l) for l in open(PRED, encoding="utf-8")]
assert "subject" in rows[0]
def sess(sid):
    m = re.search(r"resource_[a-z]+_(\d+)_", str(sid)) or re.search(r"_(\d+)_", str(sid))
    assert m, f"회기번호 파싱 실패: {sid}"
    return int(m.group(1))
byp = collections.defaultdict(list)
for r in rows: byp[r["subject"]].append((sess(r["id"]), r["pred"], r["true"]))
ns = [len(v) for v in byp.values()]
ex = list(byp)[0]
print(f"[탐지] 피험자 {len(byp)}명, 인당 세션 min/med/max = "
      f"{min(ns)}/{sorted(ns)[len(ns)//2]}/{max(ns)}")
print(f"[탐지] 회기순서 예시({ex}): {sorted(x[0] for x in byp[ex])}  ← 1,2,3... 오름차순이어야 정상")
def vote(preds):
    c = collections.Counter(preds).most_common()
    best = [x for x, n in c if n == c[0][1]]
    return preds[0] if len(best) > 1 else best[0]
print(f"\n{'k':>5} {'macro-F1':>9} {'acc':>7}")
for k in [1, 2, 3, 5, None]:
    yt, yp = [], []
    for p, lst in byp.items():
        lst = sorted(lst, key=lambda x: x[0])
        use = lst if k is None else lst[:k]
        yt.append(use[0][2]); yp.append(vote([x[1] for x in use]))
    tag = "all" if k is None else str(k)
    print(f"{tag:>5} {f1_score(yt,yp,average='macro'):>9.3f} {accuracy_score(yt,yp):>7.3f}")
