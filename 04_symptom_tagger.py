# -*- coding: utf-8 -*-
"""
04_symptom_tagger.py : 단락(발화) 수준 증상 멀티라벨 탐지
- 라벨링 JSON의 단락별 증상 라벨(예: depressive_mood, suicidal, sleep_disturbance...)로
  "어느 발화가 어떤 증상의 근거인지" 탐지하는 모델 학습
- 클래스(우울/불안/중독/일반)마다 증상 키셋이 다를 수 있어 자동 탐색 후 합집합 사용
- 분할은 01에서 만든 train/val/test.jsonl의 피험자 집합을 그대로 상속 (누수 차단 일관성)
- 산출물: 증상별 F1 리포트 + 세션 근거 하이라이트 예시(병원 데모용)

사용:
  python 04_symptom_tagger.py --scan_only                 # 클래스별 증상 키 스키마 확인
  python 04_symptom_tagger.py                             # 학습 + 평가 + 데모 리포트
"""
import argparse, json, re, random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

CLASSES = ["우울증", "불안장애", "중독", "일반군"]
# 비라벨(구조/메타) 키
STRUCT_KEYS = {"start_point", "end_point", "character_count", "cps",
               "index", "paragraph_text", "paragraph_speaker"}

# 상담사 개입기법 라벨 (내담자 증상 라벨과 구분)
TECHNIQUE_KEYS = {
    "clarification_reflection", "cognitive_restructuring", "information_provision",
    "structuring", "sympathy_support", "task_assignment", "goal_setting",
    "process_feedback", "behavioral_intervention", "training_of_coping_skills",
    "emotional_regulation_education_training", "enhancement_of_motivation",
    "facilitation_of_motivation", "accepting_attitude", "self_management",
}
# 변화/성과 라벨 (개선 요인)
OUTCOME_KEYS = {
    "acceptance_change", "behavioral_change", "cognitive_change",
    "emotional_change", "motivation_for_change", "coping",
    "emotional_regulation", "emotional_requlation", "anxiety_control",
    "self_control",
}

def infer_class(path: Path):
    for c in CLASSES:
        if c in str(path):
            return c
    return None

def infer_subject(fname: str):
    m = re.search(r"_([A-Za-z]+\d+)\.json$", fname)
    return m.group(1) if m else None

def load_split_subjects(data_dir: Path):
    """01 산출물에서 split별 subject_id(cls_ID) 집합 로드"""
    split_of = {}
    for k in ("train", "val", "test"):
        for line in open(data_dir / f"{k}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            split_of[r["subject_id"]] = k
    return split_of

def scan_schema(root: Path):
    """클래스별 단락 증상 키(정수형, 구조키 제외) 집계"""
    keys_by_cls = defaultdict(Counter)
    n_files = Counter()
    for p in root.rglob("label_*.json"):
        c = infer_class(p)
        if c is None:
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        n_files[c] += 1
        for para in obj.get("paragraph", [])[:5]:  # 파일당 5단락이면 키 파악 충분
            for k, v in para.items():
                if k in STRUCT_KEYS or not isinstance(v, int):
                    continue
                keys_by_cls[c][k] += 1
    return keys_by_cls, n_files

def collect_paragraphs(root: Path, split_of, symptom_keys, speaker="내담자",
                       with_context=True):
    """단락 레코드 수집: (context, text, y[멀티라벨], split)"""
    out = {"train": [], "val": [], "test": []}
    pos_cnt = Counter()
    n_bad = 0

    def as_text(v):
        return v.strip() if isinstance(v, str) else ""

    for p in sorted(root.rglob("label_*.json")):
        c, s = infer_class(p), infer_subject(p.name)
        if c is None or s is None:
            continue
        split = split_of.get(f"{c}_{s}")
        if split is None:
            continue  # 01에서 제외된 세션
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        paras = obj.get("paragraph", [])
        for i, para in enumerate(paras):
            if not isinstance(para, dict):
                n_bad += 1; continue
            if speaker and para.get("paragraph_speaker") != speaker:
                continue
            text = as_text(para.get("paragraph_text"))
            if len(text) < 2:
                n_bad += 1; continue
            y = []
            for k in symptom_keys:
                v = para.get(k, 0)
                y.append(1 if (isinstance(v, (int, float)) and v) else 0)
            ctx = ""
            if with_context and i > 0 and isinstance(paras[i-1], dict):
                ctx = as_text(paras[i-1].get("paragraph_text"))
            out[split].append({"cls": c, "subject": f"{c}_{s}",
                               "session": p.stem, "ctx": ctx, "text": text, "y": y})
            for k, v in zip(symptom_keys, y):
                pos_cnt[k] += v
    if n_bad:
        print(f"[안내] 비정상 단락 {n_bad}건 제외 (paragraph_text 결측/비문자열)")
    return out, pos_cnt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/jovyan/HJW_counseling")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs/symptom")
    ap.add_argument("--model", default="klue/roberta-base")
    ap.add_argument("--scan_only", action="store_true")
    ap.add_argument("--speaker", default="내담자", choices=["내담자", "상담사", ""])
    ap.add_argument("--label_group", default="symptom",
                    choices=["symptom", "technique", "outcome", "all"],
                    help="symptom=증상(내담자), technique=상담기법(상담사), "
                         "outcome=변화지표, all=전체")
    ap.add_argument("--pw_cap", type=float, default=10.0,
                    help="pos_weight 상한 (과잉 예측 완화)")
    ap.add_argument("--min_pos", type=int, default=50,
                    help="양성 예시가 이 값 미만인 증상 키는 학습에서 제외")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)

    # ── 1) 스키마 탐색 ─────────────────────────────────────
    keys_by_cls, n_files = scan_schema(root)
    print("[클래스별 라벨 JSON 파일 수]", dict(n_files))
    union = sorted({k for c in keys_by_cls for k in keys_by_cls[c]})
    sym = [k for k in union if k not in TECHNIQUE_KEYS and k not in OUTCOME_KEYS]
    tech = [k for k in union if k in TECHNIQUE_KEYS]
    outc = [k for k in union if k in OUTCOME_KEYS]
    print("\n[클래스별 라벨 키]")
    for c in CLASSES:
        print(f"  {c}: {len(keys_by_cls[c])}개")
    print(f"\n[증상/특성 라벨] {len(sym)}개: {sym}")
    print(f"\n[상담기법 라벨] {len(tech)}개: {tech}")
    print(f"\n[변화지표 라벨] {len(outc)}개: {outc}")
    if args.scan_only:
        return

    group = {"symptom": sym, "technique": tech, "outcome": outc, "all": union}
    all_keys = group[args.label_group]
    print(f"\n[선택된 라벨 그룹] {args.label_group} — {len(all_keys)}개")

    # ── 2) 단락 수집 (피험자 독립 분할 상속) ────────────────
    split_of = load_split_subjects(Path(args.data))
    data, pos_cnt = collect_paragraphs(root, split_of, all_keys,
                                       speaker=args.speaker)
    print(f"\n[단락 수] train={len(data['train'])} val={len(data['val'])} "
          f"test={len(data['test'])}  (speaker={args.speaker or '전체'})")
    print("[증상별 양성 단락 수]", dict(pos_cnt.most_common()))

    keep = [k for k in all_keys if pos_cnt[k] >= args.min_pos]
    kidx = [all_keys.index(k) for k in keep]
    print(f"[학습 대상 증상] {len(keep)}개 (양성 {args.min_pos}건 이상): {keep}")
    if not keep:
        print("학습 가능한 증상 키가 없습니다. --min_pos를 낮추세요."); return

    # ── 3) 학습 ────────────────────────────────────────────
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(keep),
        problem_type="multi_label_classification").to(device)

    class ParaDS(Dataset):
        def __init__(self, recs):
            self.recs = recs
        def __len__(self):
            return len(self.recs)
        def __getitem__(self, i):
            r = self.recs[i]
            if r["ctx"]:
                enc = tok(r["ctx"], r["text"], truncation="longest_first",
                          max_length=args.max_len)
            else:
                enc = tok(r["text"], truncation=True, max_length=args.max_len)
            y = [float(r["y"][j]) for j in kidx]
            return {"input_ids": enc["input_ids"], "y": y}

    def coll(batch):
        mx = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), mx), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), mx), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"]); att[i, :n] = 1
        return {"input_ids": ids, "attention_mask": att,
                "y": torch.tensor([b["y"] for b in batch])}

    dl_tr = DataLoader(ParaDS(data["train"]), batch_size=args.bs, shuffle=True,
                       collate_fn=coll)
    dl_va = DataLoader(ParaDS(data["val"]), batch_size=args.bs, collate_fn=coll)
    dl_te = DataLoader(ParaDS(data["test"]), batch_size=args.bs, collate_fn=coll)

    # 희소 양성 대응: pos_weight = 음성/양성 비 (상한 30)
    n_tr = len(data["train"])
    pw = torch.tensor([min(args.pw_cap, (n_tr - pos_cnt[k]) / max(1, pos_cnt[k]))
                       for k in keep]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(dl_tr) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(total * 0.06), total)

    @torch.no_grad()
    def run_eval(dl):
        model.eval()
        Y, P = [], []
        for b in dl:
            lg = model(input_ids=b["input_ids"].to(device),
                       attention_mask=b["attention_mask"].to(device)).logits
            P.append(torch.sigmoid(lg).cpu().numpy()); Y.append(b["y"].numpy())
        return np.vstack(Y), np.vstack(P)

    best, best_state = -1, None
    for ep in range(1, args.epochs + 1):
        model.train()
        for step, b in enumerate(dl_tr):
            lg = model(input_ids=b["input_ids"].to(device),
                       attention_mask=b["attention_mask"].to(device)).logits
            loss = crit(lg, b["y"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            if step % 100 == 0:
                print(f"ep{ep} step{step}/{len(dl_tr)} loss={loss.item():.4f}")
        Y, P = run_eval(dl_va)
        f1 = f1_score(Y, (P > 0.5).astype(int), average="macro", zero_division=0)
        print(f"[ep{ep}] val macro-F1={f1:.4f}")
        if f1 > best:
            best = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # 라벨별 임계값 최적화 (val 기준 F1 최대화) — 희소 멀티라벨 필수 절차
    Yv, Pv = run_eval(dl_va)
    grid = np.arange(0.05, 0.96, 0.05)
    ths = np.full(len(keep), 0.5)
    for j in range(len(keep)):
        if Yv[:, j].sum() == 0:
            continue
        f1s_j = [f1_score(Yv[:, j], (Pv[:, j] > t).astype(int), zero_division=0)
                 for t in grid]
        ths[j] = grid[int(np.argmax(f1s_j))]

    Y, P = run_eval(dl_te)
    Pb = (P > ths[None, :]).astype(int)
    pr, rc, f1s, sup = precision_recall_fscore_support(Y, Pb, zero_division=0)
    lines = [f"model={args.model}  speaker={args.speaker}  group={args.label_group}  val best macro-F1={best:.4f}",
             f"[TEST] macro-F1={f1_score(Y, Pb, average='macro', zero_division=0):.4f}  "
             f"micro-F1={f1_score(Y, Pb, average='micro', zero_division=0):.4f}", ""]
    for i, k in enumerate(keep):
        lines.append(f"  {k:24s} P={pr[i]:.3f} R={rc[i]:.3f} F1={f1s[i]:.3f} "
                     f"th={ths[i]:.2f} (n={int(sup[i])})")
    rep = "\n".join(lines)
    print(rep)
    (out / f"report_{args.label_group}.txt").write_text(rep, encoding="utf-8")
    metrics = {"macro_f1": float(f1_score(Y, Pb, average="macro", zero_division=0)),
               "micro_f1": float(f1_score(Y, Pb, average="micro", zero_division=0)),
               "labels": {k: {"P": float(pr[i]), "R": float(rc[i]),
                              "F1": float(f1s[i]), "n": int(sup[i]),
                              "th": float(ths[i])}
                          for i, k in enumerate(keep)}}
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    model.save_pretrained(out / "model"); tok.save_pretrained(out / "model")

    # ── 4) 병원 데모용: test 세션 3건 근거 하이라이트 ────────
    demo = []
    by_sess = defaultdict(list)
    for i, r in enumerate(data["test"]):
        by_sess[r["session"]].append((i, r))
    for sess in list(by_sess)[:3]:
        items = by_sess[sess]
        demo.append(f"\n===== {sess} ({items[0][1]['cls']}) =====")
        for i, r in items:
            hits = [(keep[j], P[i, j]) for j in range(len(keep)) if P[i, j] > ths[j]]
            if hits:
                tag = ", ".join(f"{k}({v:.2f})" for k, v in hits)
                demo.append(f"[{tag}] {r['text'][:120]}")
    (out / f"demo_evidence_{args.label_group}.txt").write_text("\n".join(demo), encoding="utf-8")
    print(f"[저장] {out}/report_{args.label_group}.txt, demo_evidence_{args.label_group}.txt")

if __name__ == "__main__":
    main()
