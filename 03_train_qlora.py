# -*- coding: utf-8 -*-
"""
03_train_qlora.py : LLM QLoRA 파인튜닝 (분류)
- Qwen2.5-7B-Instruct 4bit(NF4) + LoRA, 상담 전사 → 클래스명 1단어 생성 SFT
- 평가: 제약 생성 후 클래스명 매칭, macro-F1 / 혼동행렬
- A100-40GB 1장 기준 동작 (7B, max_len 4096)
사용:
  pip install "transformers>=4.41" peft bitsandbytes accelerate scikit-learn
  python 03_train_qlora.py --data data/processed --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse, json, re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          get_linear_schedule_with_warmup)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

CLASSES = ["우울증", "불안장애", "중독", "일반군"]

SYSTEM = ("당신은 심리상담 전사 분석 전문가입니다. 주어진 상담 대화 전사를 읽고 "
          "내담자의 상태를 다음 중 정확히 하나로 분류하세요: 우울증, 불안장애, 중독, 일반군. "
          "다른 말 없이 분류명만 출력하세요.")

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def truncate(text, tok, budget):
    """전사 앞 70% + 뒤 30% 보존 (도입부 증상 서술 + 종결부 정리 반영)"""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return text
    head = int(budget * 0.7); tail = budget - head
    return (tok.decode(ids[:head]) + "\n...(중략)...\n" + tok.decode(ids[-tail:]))

def build_prompt(tok, text, answer=None):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"[상담 전사]\n{text}\n\n분류:"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt + (answer + tok.eos_token if answer is not None else "")

class SFTDS(Dataset):
    def __init__(self, recs, tok, max_len=4096, field="text"):
        self.recs, self.tok, self.max_len, self.field = recs, tok, max_len, field
        self.budget = max_len - 300  # 프롬프트/답변 여유분

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        text = truncate(r[self.field], self.tok, self.budget)
        full = build_prompt(self.tok, text, answer=r["cls"])
        prompt_only = build_prompt(self.tok, text)
        f = self.tok(full, truncation=True, max_length=self.max_len)["input_ids"]
        p_len = len(self.tok(prompt_only)["input_ids"])
        labels = [-100] * min(p_len, len(f)) + f[p_len:]
        labels = labels[:len(f)]
        return {"input_ids": f, "labels": labels}

def collate(batch, pad_id):
    mx = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), mx), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), mx), -100, dtype=torch.long)
    att = torch.zeros((len(batch), mx), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        lab[i, :n] = torch.tensor(b["labels"])
        att[i, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}

def parse_pred(text):
    for c in CLASSES:
        if c in text:
            return CLASSES.index(c)
    return CLASSES.index("일반군")  # 파싱 실패 시 기본값 (빈도 기록됨)

@torch.no_grad()
def evaluate(model, tok, recs, max_len, device, field="text", log_every=50):
    model.eval()
    ys, ps, fails = [], [], 0
    for i, r in enumerate(recs):
        text = truncate(r[field], tok, max_len - 300)
        prompt = build_prompt(tok, text)
        ids = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=max_len).to(device)
        out = model.generate(**ids, max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        if not any(c in gen for c in CLASSES):
            fails += 1
        ys.append(r["label"]); ps.append(parse_pred(gen))
        if i % log_every == 0:
            print(f"  eval {i}/{len(recs)}  gen='{gen.strip()[:20]}'")
    if fails:
        print(f"[안내] 클래스명 파싱 실패 {fails}건 (기본값 대체)")
    return np.array(ys), np.array(ps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="runs/qlora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--field", default="client_text",
                    choices=["text", "client_text"],
                    help="client_text=내담자 발화만(권장, 라벨 누수 차단), text=전체 대화")
    ap.add_argument("--zero_shot_only", action="store_true",
                    help="파인튜닝 없이 zero-shot 평가만 수행")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    data = Path(args.data)
    train, val, test = (load_jsonl(data / f"{k}.jsonl") for k in ("train", "val", "test"))
    print(f"[데이터] train={len(train)} val={len(val)} test={len(test)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa")

    if args.zero_shot_only:
        y, p = evaluate(model, tok, test, args.max_len, device, field=args.field)
        tag = "zeroshot"
    else:
        model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2,
                          lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        ds = SFTDS(train, tok, args.max_len, field=args.field)
        dl = DataLoader(ds, batch_size=args.bs, shuffle=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))
        opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad],
                                lr=args.lr)
        total = (len(dl) // args.grad_accum) * args.epochs
        sch = get_linear_schedule_with_warmup(opt, int(total * 0.03), total)

        step_g = 0
        for ep in range(1, args.epochs + 1):
            model.train()
            for step, b in enumerate(dl):
                out_ = model(input_ids=b["input_ids"].to(device),
                             attention_mask=b["attention_mask"].to(device),
                             labels=b["labels"].to(device))
                (out_.loss / args.grad_accum).backward()
                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
                    opt.step(); sch.step(); opt.zero_grad(); step_g += 1
                    if step_g % 10 == 0:
                        print(f"ep{ep} gstep{step_g}/{total} loss={out_.loss.item():.4f}")
            # 에폭별 val 일부(최대 100건)로 중간 점검
            y, p = evaluate(model, tok, val[:100], args.max_len, device, field=args.field)
            print(f"[ep{ep}] val(≤100) macro-F1={f1_score(y, p, average='macro'):.4f}")
        model.save_pretrained(out / "adapter")
        tok.save_pretrained(out / "adapter")
        y, p = evaluate(model, tok, test, args.max_len, device, field=args.field)
        tag = "qlora"

    rep = (f"model={args.model} ({tag})  field={args.field}  "
           f"max_len={args.max_len} epochs={args.epochs} lora_r={args.lora_r}\n"
           f"[TEST] acc={accuracy_score(y,p):.4f}  macro-F1={f1_score(y,p,average='macro'):.4f}\n\n"
           + classification_report(y, p, target_names=CLASSES, digits=4))
    print(rep)
    (out / f"report_{tag}.txt").write_text(rep, encoding="utf-8")
    np.savetxt(out / f"confusion_{tag}.csv", confusion_matrix(y, p), fmt="%d", delimiter=",")
    wrong = [{"id": test[i]["id"], "true": CLASSES[int(y[i])], "pred": CLASSES[int(p[i])]}
             for i in range(len(y)) if y[i] != p[i]]
    (out / f"errors_{tag}.jsonl").write_text(
        "\n".join(json.dumps(w, ensure_ascii=False) for w in wrong), encoding="utf-8")
    print(f"[오분류] {len(wrong)}건 → {out}/errors_{tag}.jsonl")
    print(f"[저장 완료] {out}/")

if __name__ == "__main__":
    main()
