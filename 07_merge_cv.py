# -*- coding: utf-8 -*-
"""
07_merge_cv.py : 5-fold 예측 통합 최종 리포트
- 각 fold의 predictions.jsonl을 합쳐 전체 209명 기준 세션/피험자 수준 성능 산출
- 부트스트랩 95% CI(피험자 리샘플링), fold별 편차, 혼동행렬 CSV 저장
사용: python 07_merge_cv.py --pattern "runs/cv14b_fold{k}" --out runs/cv14b_final
"""
import argparse, json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

CLASSES = ["우울증", "불안장애", "중독", "일반군"]

def bootstrap_ci(y, p, subjects=None, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    stats = []
    if subjects is not None:
        subjects = np.asarray(subjects)
        uniq = np.unique(subjects)
        idx_by_s = {s: np.where(subjects == s)[0] for s in uniq}
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by_s[s] for s in pick])
            stats.append(f1_score(y[idx], p[idx], average="macro", zero_division=0))
    else:
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            stats.append(f1_score(y[idx], p[idx], average="macro", zero_division=0))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="runs/cv14b_fold{k}",
                    help="{k} 자리에 0~4가 들어가는 fold 결과 경로 패턴")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="runs/cv_final")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    all_res, fold_f1 = [], []
    for k in range(args.k):
        p = Path(args.pattern.format(k=k)) / "predictions.jsonl"
        if not p.exists():
            print(f"[경고] fold{k} 예측 없음: {p} — 건너뜀")
            continue
        res = [json.loads(l) for l in open(p, encoding="utf-8")]
        for r in res:
            r["fold"] = k
        all_res += res
        y = [r["true"] for r in res]; pr = [r["pred"] for r in res]
        fold_f1.append((k, f1_score(y, pr, average="macro", zero_division=0)))
    assert all_res, "취합할 예측이 없습니다"

    # 중복 세션 검증 (각 세션은 정확히 한 fold의 test)
    ids = [r["id"] for r in all_res]
    dup = [i for i, c in Counter(ids).items() if c > 1]
    assert not dup, f"세션 중복 발견: {dup[:5]}"

    lines = [f"folds={len(fold_f1)}  총 세션={len(all_res)}  "
             f"총 피험자={len({r['subject'] for r in all_res})}", ""]
    lines.append("[fold별 세션 macro-F1] " +
                 "  ".join(f"f{k}:{v:.3f}" for k, v in fold_f1))
    vals = [v for _, v in fold_f1]
    lines.append(f"  평균 {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    lines.append("")

    # 세션 수준 (전체 통합)
    y = np.array([r["true"] for r in all_res])
    p = np.array([r["pred"] for r in all_res])
    subs = [r["subject"] for r in all_res]
    lo, hi = bootstrap_ci(y, p, subjects=np.array(subs))
    lines.append(f"[세션 수준·통합] n={len(y)}  acc={accuracy_score(y,p):.4f}  "
                 f"macro-F1={f1_score(y,p,average='macro',zero_division=0):.4f}  "
                 f"(95% CI {lo:.3f}–{hi:.3f})")
    lines.append(classification_report(y, p, target_names=CLASSES, digits=4,
                                       zero_division=0))
    np.savetxt(out / "confusion_session.csv", confusion_matrix(y, p),
               fmt="%d", delimiter=",")

    # 피험자 수준 (전체 209명)
    by_s = defaultdict(list); true_s = {}
    for r in all_res:
        by_s[r["subject"]].append(r["pred"]); true_s[r["subject"]] = r["true"]
    ys, ps = [], []
    for s, preds in by_s.items():
        ys.append(true_s[s]); ps.append(Counter(preds).most_common(1)[0][0])
    ys, ps = np.array(ys), np.array(ps)
    lo2, hi2 = bootstrap_ci(ys, ps)
    lines.append(f"[피험자 수준·통합] n={len(ys)}명  acc={accuracy_score(ys,ps):.4f}  "
                 f"macro-F1={f1_score(ys,ps,average='macro',zero_division=0):.4f}  "
                 f"(95% CI {lo2:.3f}–{hi2:.3f})")
    lines.append(classification_report(ys, ps, target_names=CLASSES, digits=4,
                                       zero_division=0))
    np.savetxt(out / "confusion_subject.csv", confusion_matrix(ys, ps),
               fmt="%d", delimiter=",")

    rep = "\n".join(lines)
    print(rep)
    (out / "final_report.txt").write_text(rep, encoding="utf-8")
    (out / "all_predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_res),
        encoding="utf-8")
    print(f"[저장 완료] {out}/final_report.txt")

if __name__ == "__main__":
    main()
