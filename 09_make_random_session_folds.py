# -*- coding: utf-8 -*-
"""실험① 무작위 세션 분할 (v2: id 기준 중복제거)"""
import json, glob, os, random, collections

SRC, DST = "data/folds", "data/folds_randsess"
random.seed(42)
f0 = sorted(glob.glob(f"{SRC}/fold0/*.jsonl")); assert f0
names = [os.path.basename(p) for p in f0]
seen, records = set(), []
for p in f0:
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        key = r.get("id") or json.dumps(r, ensure_ascii=False, sort_keys=True)
        if key in seen: continue
        seen.add(key); records.append(r)
print(f"[탐지] 전체 세션 수: {len(records)} (1500이어야 정상)")
lab = next(k for k in ("label","cls","y","group","클래스") if k in records[0])
print(f"[탐지] 분포: {dict(collections.Counter(r[lab] for r in records))}")
byc = collections.defaultdict(list)
for r in records: byc[r[lab]].append(r)
folds = [[] for _ in range(5)]
for c, rs in byc.items():
    random.shuffle(rs)
    for i, r in enumerate(rs): folds[i % 5].append(r)
def dump(path, rs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
for k in range(5):
    test = folds[k]; rest = [r for j in range(5) if j != k for r in folds[j]]
    random.shuffle(rest); nval = max(1, int(len(rest) * 0.1))
    d = f"{DST}/fold{k}"
    dump(f"{d}/train.jsonl", rest[nval:]); dump(f"{d}/val.jsonl", rest[:nval]); dump(f"{d}/test.jsonl", test)
    print(f"fold{k}: train={len(rest)-nval} val={nval} test={len(test)}")
print("[완료]", DST)
