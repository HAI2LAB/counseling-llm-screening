# -*- coding: utf-8 -*-
"""
06_make_folds.py : 피험자 층화 5-fold 교차검증 분할 생성
- 전체(train+val+test) 209명을 클래스별 층화로 5개 fold에 배정
- fold k 실행 시: test = fold k 피험자, val = fold (k+1)%5 피험자, train = 나머지
- 산출물: data/folds/fold{0..4}/{train,val,test}.jsonl + label_map.json + folds_stats.txt
사용: python 06_make_folds.py --data data/processed --out data/folds
"""
import argparse, json, random
from pathlib import Path
from collections import Counter, defaultdict

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="data/folds")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # 전체 데이터 통합
    recs = []
    for k in ("train", "val", "test"):
        recs += load_jsonl(data / f"{k}.jsonl")
    by_subj = defaultdict(list)
    for r in recs:
        by_subj[r["subject_id"]].append(r)
    print(f"[전체] 세션 {len(recs)}건 / 피험자 {len(by_subj)}명")

    # 클래스별 층화 fold 배정 (피험자 단위)
    fold_of = {}
    by_cls = defaultdict(list)
    for s in by_subj:
        by_cls[by_subj[s][0]["cls"]].append(s)
    for c, subjects in by_cls.items():
        rng.shuffle(subjects)
        for i, s in enumerate(subjects):
            fold_of[s] = i % args.k
    fold_cnt = Counter(fold_of.values())
    print(f"[fold별 피험자 수] {dict(sorted(fold_cnt.items()))}")

    # fold별 train/val/test 생성
    lines = [f"seed={args.seed} k={args.k}  피험자 {len(by_subj)}명 / 세션 {len(recs)}건", ""]
    for k in range(args.k):
        te_f, va_f = k, (k + 1) % args.k
        splits = {"train": [], "val": [], "test": []}
        for s, rs in by_subj.items():
            key = ("test" if fold_of[s] == te_f
                   else "val" if fold_of[s] == va_f else "train")
            splits[key].extend(rs)
        # 누수 검증
        subj = {key: {r["subject_id"] for r in v} for key, v in splits.items()}
        for a in subj:
            for b in subj:
                if a < b:
                    assert not (subj[a] & subj[b]), f"fold{k} 누수: {a}∩{b}"
        d = out / f"fold{k}"
        d.mkdir(exist_ok=True)
        for key, v in splits.items():
            with open(d / f"{key}.jsonl", "w", encoding="utf-8") as f:
                for r in v:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # label_map 복사
        (d / "label_map.json").write_text(
            (data / "label_map.json").read_text(encoding="utf-8"), encoding="utf-8")
        msg = (f"[fold{k}] " + "  ".join(
            f"{key}={len(v)}건/{len(subj[key])}명"
            f"({dict(Counter(r['cls'] for r in v))})"[:120]
            for key, v in splits.items()))
        print(msg)
        lines.append(msg)
    (out / "folds_stats.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[저장 완료] {out}/fold0..{args.k-1}")

if __name__ == "__main__":
    main()
