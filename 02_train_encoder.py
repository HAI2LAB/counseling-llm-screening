# -*- coding: utf-8 -*-
"""
02_train_encoder.py : 인코더 베이스라인 (KLUE-RoBERTa / KoELECTRA)
- 긴 상담 전사를 512토큰 청크로 분할해 학습, 평가는 세션(문서) 단위 로짓 평균 집계
- 클래스 불균형 대응: class-weighted CE
- 산출물: out_dir/report.txt (macro-F1, per-class F1), confusion_matrix.png
사용:
  pip install "transformers>=4.41" scikit-learn matplotlib
  python 02_train_encoder.py --data data/processed --model klue/roberta-base
"""
import argparse, json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

class ChunkDS(Dataset):
    """세션 텍스트를 max_len 청크로 분할. 각 청크는 (doc_idx, label) 보유"""
    def __init__(self, recs, tok, max_len=512, stride=128, max_chunks=8, field="text"):
        self.items = []
        for di, r in enumerate(recs):
            ids = tok(r[field], add_special_tokens=False)["input_ids"]
            step = max_len - 2 - stride
            starts = list(range(0, max(1, len(ids)), step))[:max_chunks]
            for st in starts:
                chunk = ids[st:st + max_len - 2]
                if len(chunk) < 20 and st > 0:
                    continue
                self.items.append({"ids": chunk, "label": r["label"], "doc": di})
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        ids = [self.tok.cls_token_id] + it["ids"] + [self.tok.sep_token_id]
        return {"input_ids": ids, "label": it["label"], "doc": it["doc"]}

def collate(batch, pad_id):
    mx = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), mx), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), mx), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        att[i, :n] = 1
    return {"input_ids": ids, "attention_mask": att,
            "labels": torch.tensor([b["label"] for b in batch]),
            "docs": torch.tensor([b["doc"] for b in batch])}

@torch.no_grad()
def eval_docs(model, loader, n_docs, n_cls, device):
    """청크 로짓을 문서 단위 평균 → 문서 예측"""
    model.eval()
    logit_sum = np.zeros((n_docs, n_cls)); cnt = np.zeros(n_docs)
    labels = np.full(n_docs, -1)
    for b in loader:
        out = model(input_ids=b["input_ids"].to(device),
                    attention_mask=b["attention_mask"].to(device))
        lg = out.logits.float().cpu().numpy()
        for i, d in enumerate(b["docs"].numpy()):
            logit_sum[d] += lg[i]; cnt[d] += 1
            labels[d] = b["labels"][i].item()
        del out
    mask = cnt > 0
    preds = (logit_sum[mask] / cnt[mask, None]).argmax(1)
    return labels[mask], preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--model", default="klue/roberta-base")
    ap.add_argument("--out", default="runs/encoder")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_chunks", type=int, default=8)
    ap.add_argument("--field", default="client_text",
                    choices=["text", "client_text"],
                    help="client_text=내담자 발화만(권장, 라벨 누수 차단), text=전체 대화")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    data = Path(args.data)
    label_map = json.loads((data / "label_map.json").read_text(encoding="utf-8"))
    id2cls = {v: k for k, v in label_map.items()}
    n_cls = len(label_map)
    train, val, test = (load_jsonl(data / f"{k}.jsonl") for k in ("train", "val", "test"))
    print(f"[데이터] train={len(train)} val={len(val)} test={len(test)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=n_cls).to(device)

    ds_tr = ChunkDS(train, tok, args.max_len, max_chunks=args.max_chunks, field=args.field)
    ds_va = ChunkDS(val, tok, args.max_len, max_chunks=args.max_chunks, field=args.field)
    ds_te = ChunkDS(test, tok, args.max_len, max_chunks=args.max_chunks, field=args.field)
    coll = lambda b: collate(b, tok.pad_token_id)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, collate_fn=coll)
    dl_va = DataLoader(ds_va, batch_size=args.bs, collate_fn=coll)
    dl_te = DataLoader(ds_te, batch_size=args.bs, collate_fn=coll)

    # class weight (청크 기준)
    cc = Counter(it["label"] for it in ds_tr.items)
    w = torch.tensor([sum(cc.values()) / (n_cls * cc[i]) for i in range(n_cls)],
                     dtype=torch.float).to(device)
    print(f"[class weights] { {id2cls[i]: round(w[i].item(),3) for i in range(n_cls)} }")
    crit = nn.CrossEntropyLoss(weight=w)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(dl_tr) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(total * 0.06), total)

    best_f1, best_state = -1, None
    for ep in range(1, args.epochs + 1):
        model.train()
        for step, b in enumerate(dl_tr):
            out_ = model(input_ids=b["input_ids"].to(device),
                         attention_mask=b["attention_mask"].to(device))
            loss = crit(out_.logits, b["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            if step % 50 == 0:
                print(f"ep{ep} step{step}/{len(dl_tr)} loss={loss.item():.4f}")
        y, p = eval_docs(model, dl_va, len(val), n_cls, device)
        f1 = f1_score(y, p, average="macro")
        print(f"[ep{ep}] val macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    y, p = eval_docs(model, dl_te, len(test), n_cls, device)
    names = [id2cls[i] for i in range(n_cls)]
    rep = (f"model={args.model}  field={args.field}\nval best macro-F1={best_f1:.4f}\n\n"
           f"[TEST] acc={accuracy_score(y,p):.4f}  macro-F1={f1_score(y,p,average='macro'):.4f}\n\n"
           + classification_report(y, p, target_names=names, digits=4))
    print(rep)
    (out / "report.txt").write_text(rep, encoding="utf-8")

    cm = confusion_matrix(y, p)
    np.savetxt(out / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "DejaVu Sans"]
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n_cls), names, rotation=45)
        ax.set_yticks(range(n_cls), names)
        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(out / "confusion_matrix.png", dpi=200)
        print(f"[저장] {out}/confusion_matrix.png")
    except Exception as e:
        print(f"[안내] 혼동행렬 이미지는 생략(csv는 저장됨): {e}")

    # 오분류 목록 (병원 설득용 정성 분석 소재)
    wrong = [{"id": test[i]["id"], "true": id2cls[int(y[i])], "pred": id2cls[int(p[i])]}
             for i in range(len(y)) if y[i] != p[i]]
    (out / "errors.jsonl").write_text(
        "\n".join(json.dumps(w, ensure_ascii=False) for w in wrong), encoding="utf-8")
    print(f"[오분류] {len(wrong)}건 → {out}/errors.jsonl")

if __name__ == "__main__":
    main()
