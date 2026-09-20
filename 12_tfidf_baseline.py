# -*- coding: utf-8 -*-
"""실험④ TF-IDF(word1-2 + char2-4) + LogisticRegression 기준선.
사용: python 12_tfidf_baseline.py data/folds        (원본)
     python 12_tfidf_baseline.py data/folds_masked  (마스킹판)"""
import json, glob, os, sys, collections, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import f1_score, accuracy_score

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/folds"
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
r0 = load(glob.glob(f"{ROOT}/fold0/*.jsonl")[0])[0]
txt = max(((k, v) for k, v in r0.items() if isinstance(v, str)), key=lambda kv: len(kv[1]))[0]
lab = next(k for k in ("label","cls","y","group","클래스") if k in r0)
idk = next((k for k in ("id","file","session","name","fname") if k in r0), None)
def pid_of(r):
    if "subject" in r: return r["subject"]
    sid = str(r[idk])
    g = re.search(r"resource_([a-z]+)_", sid)
    d = re.search(r"([A-Z]+\d+)\s*$", sid)
    assert g and d, f"피험자키 파싱 실패: {sid}"
    return f"{g.group(1)}_{d.group(1)}"
print(f"[탐지] 텍스트='{txt}' 라벨='{lab}' id='{idk}' subject필드={'subject' in r0}  (root={ROOT})")
print(f"[탐지] 피험자키 예시: {r0[idk]} -> {pid_of(r0)}")

sf1, pf1, pacc = [], [], []
for k in range(5):
    d = f"{ROOT}/fold{k}"
    fs = {os.path.basename(p): p for p in glob.glob(f"{d}/*.jsonl")}
    tr = load(fs.get("train.jsonl", sorted(fs.values())[0]))
    if "val.jsonl" in fs: tr += load(fs["val.jsonl"])
    te = load(fs.get("test.jsonl", sorted(fs.values())[-1]))
    vec = FeatureUnion([("w", TfidfVectorizer(ngram_range=(1,2), max_features=200000, min_df=2)),
                        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), max_features=200000, min_df=2))])
    X = vec.fit_transform([r[txt] for r in tr]); Xt = vec.transform([r[txt] for r in te])
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit(X, [r[lab] for r in tr])
    yp = clf.predict(Xt); yt = [r[lab] for r in te]
    sf1.append(f1_score(yt, yp, average="macro"))
    if idk:
        byp = collections.defaultdict(list)
        for r, p in zip(te, yp): byp[pid_of(r)].append((p, r[lab]))
        pt = [v[0][1] for v in byp.values()]
        pp = [collections.Counter(x[0] for x in v).most_common(1)[0][0] for v in byp.values()]
        pf1.append(f1_score(pt, pp, average="macro")); pacc.append(accuracy_score(pt, pp))
    print(f"fold{k}: 세션F1={sf1[-1]:.3f}" + (f" 피험자F1={pf1[-1]:.3f}" if pf1 else ""))
print(f"\n[{ROOT}] 세션 macro-F1 {np.mean(sf1):.3f}±{np.std(sf1):.3f}" +
      (f" | 피험자 macro-F1 {np.mean(pf1):.3f}±{np.std(pf1):.3f} acc {np.mean(pacc):.3f}" if pf1 else ""))
