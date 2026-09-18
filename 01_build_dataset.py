# -*- coding: utf-8 -*-
"""
01_build_dataset.py : 세션(txt 1건) 단위 데이터셋 구축 + 피험자 독립 분할
- 원천데이터 txt를 1레코드로, 클래스는 폴더명(우울증/불안장애/중독/일반군)에서 추출
- 동일 피험자(subject_id)가 train/val/test에 절대 섞이지 않도록 GroupShuffleSplit
- AI Hub 최상위 Training/Validation 폴더가 모두 있으면:
    Training -> (피험자 단위) train/val,  Validation -> test
  Validation만 있으면: 전체를 피험자 단위 70/15/15 분할
출력: out_dir/{train,val,test}.jsonl, label_map.json, stats.txt
사용: python 01_build_dataset.py --root /home/jovyan/HJW_counseling --out data/processed
"""
import argparse, json, re, random
from pathlib import Path
from collections import Counter, defaultdict

CLASSES = ["우울증", "불안장애", "중독", "일반군"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}

def infer_class(path: Path):
    s = str(path)
    for c in CLASSES:
        if c in s:
            return c
    return None

def infer_subject(fname: str):
    m = re.search(r"_([A-Za-z]+\d+)\.(txt|json)$", fname)
    return m.group(1) if m else None

def infer_session(path: Path):
    m = re.search(r"(\d+)회기", str(path))
    if m:
        return int(m.group(1))
    m = re.search(r"_(\d+)_check", path.name)
    return int(m.group(1)) if m else None

def read_text(p: Path):
    for enc in ("utf-8", "cp949"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return p.read_text(encoding="utf-8", errors="ignore")

def collect(root: Path):
    """원천데이터 하위 txt만 수집 (라벨링 json 폴더의 txt 중복 방지)"""
    recs, skipped = [], []
    txts = [p for p in root.rglob("*.txt") if "원천데이터" in str(p)]
    if not txts:  # 폴더명이 다르면 전체 txt로 폴백
        txts = list(root.rglob("*.txt"))
    for p in sorted(txts):
        c, s = infer_class(p), infer_subject(p.name)
        if c is None or s is None:
            skipped.append(str(p)); continue
        text = read_text(p).strip()
        if len(text) < 50:  # 빈/손상 파일 제외 (9~10회기 소용량 파일 검출용)
            skipped.append(str(p) + f"  [len={len(text)}]"); continue
        client, counselor = split_speakers(text)
        recs.append({
            "id": p.stem,
            "path": str(p),
            "top": p.relative_to(root).parts[0],   # Training / Validation 등
            "cls": c,
            "label": LABEL_MAP[c],
            "subject_id": f"{c}_{s}",  # 클래스 간 ID 충돌 방지
            "session": infer_session(p),
            "text": text,
            "client_text": client,       # 내담자 발화만 (라벨 누수 차단 설정)
            "counselor_text": counselor,
        })
    return recs, skipped

def split_speakers(text):
    """'상담사 :' / '내담자 :' 턴 분리. 화자 표기 없는 줄은 직전 화자에 귀속"""
    client, counselor, cur = [], [], None
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(상담사|내담자)\s*:\s*(.*)$", line)
        if m:
            cur = m.group(1)
            utt = m.group(2)
        else:
            utt = line
        (client if cur == "내담자" else counselor).append(utt)
    return "\n".join(client), "\n".join(counselor)

DISEASE_KW = {"우울": "우울증", "불안": "불안장애", "중독": "중독"}

def leakage_report(recs):
    """상담사/내담자 발화 내 질환 키워드 언급 빈도 → 라벨 누수 정도 정량화"""
    rows = ["[라벨 누수 점검] 발화 내 질환 키워드 평균 언급 횟수 (세션당)"]
    for who, field in (("상담사", "counselor_text"), ("내담자", "client_text")):
        by_cls = defaultdict(lambda: Counter())
        n_cls = Counter()
        for r in recs:
            n_cls[r["cls"]] += 1
            for kw in DISEASE_KW:
                by_cls[r["cls"]][kw] += r[field].count(kw)
        for c in sorted(n_cls):
            avg = {kw: round(by_cls[c][kw] / n_cls[c], 2) for kw in DISEASE_KW}
            rows.append(f"  {who} | {c}: {avg}")
    return "\n".join(rows)

def subject_split(recs, ratios=(0.7, 0.15, 0.15), seed=42):
    """피험자 단위 계층적(클래스별) 분할"""
    rng = random.Random(seed)
    by_cls_subj = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by_cls_subj[r["cls"]][r["subject_id"]].append(r)
    out = {"train": [], "val": [], "test": []}
    for c, subj_map in by_cls_subj.items():
        subjects = sorted(subj_map)
        rng.shuffle(subjects)
        n = len(subjects)
        n_tr = max(1, int(round(n * ratios[0])))
        n_va = max(1, int(round(n * ratios[1])))
        splits = {"train": subjects[:n_tr],
                  "val": subjects[n_tr:n_tr + n_va],
                  "test": subjects[n_tr + n_va:]}
        for k, ss in splits.items():
            for s in ss:
                out[k].extend(subj_map[s])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/jovyan/HJW_counseling")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    recs, skipped = collect(root)
    tops = sorted({r["top"] for r in recs})
    has_train = any("Train" in t for t in tops)
    has_val = any("Valid" in t for t in tops)

    if has_train and has_val:
        tr_all = [r for r in recs if "Train" in r["top"]]
        test = [r for r in recs if "Valid" in r["top"]]
        # 공식 분할 간 피험자 누수 검사 → 겹치면 test에서 제거
        tr_subj = {r["subject_id"] for r in tr_all}
        leak = sorted({r["subject_id"] for r in test} & tr_subj)
        if leak:
            print(f"[경고] Training/Validation 피험자 중복 {len(leak)}명 → test에서 제외: {leak[:10]}")
            test = [r for r in test if r["subject_id"] not in leak]
        sp = subject_split(tr_all, ratios=(0.85, 0.15, 0.0), seed=args.seed)
        splits = {"train": sp["train"], "val": sp["val"], "test": test}
    else:
        print("[안내] Training/Validation 최상위 구분 미발견 → 전체 피험자 단위 70/15/15 분할")
        splits = subject_split(recs, seed=args.seed)

    # 최종 누수 검증 (실패 시 즉시 중단)
    subj = {k: {r["subject_id"] for r in v} for k, v in splits.items()}
    for a in subj:
        for b in subj:
            if a < b:
                assert not (subj[a] & subj[b]), f"피험자 누수 발견: {a} ∩ {b}"

    lines = [f"[수집] 총 {len(recs)}건, 제외 {len(skipped)}건"]
    for k, v in splits.items():
        c = Counter(r["cls"] for r in v)
        s = len({r["subject_id"] for r in v})
        lines.append(f"[{k}] {len(v)}건 / 피험자 {s}명 / 클래스 {dict(c)}")
    report = "\n".join(lines) + "\n\n" + leakage_report(recs)
    print(report)
    (out / "stats.txt").write_text(report + "\n\n[제외 목록]\n" + "\n".join(skipped),
                                   encoding="utf-8")
    (out / "label_map.json").write_text(
        json.dumps(LABEL_MAP, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in splits.items():
        with open(out / f"{k}.jsonl", "w", encoding="utf-8") as f:
            for r in v:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[저장 완료] {out}/")

if __name__ == "__main__":
    main()
