# -*- coding: utf-8 -*-
"""
05_train_qlora_ddp.py : QLoRA 분산 학습·평가 (1개 작업을 GPU 3장이 분담)
- 학습: DDP — 데이터를 GPU 수만큼 분할해 병렬 학습 (유효 배치 = bs x GPU수 x grad_accum)
- 평가: test 세션을 GPU 수만큼 분할해 병렬 생성 → rank0이 취합
- 추가 기능: 소수 클래스 오버샘플링(--oversample), 긴 세션 청크 앙상블(--chunks),
  피험자 수준 집계, 부트스트랩 95% CI (모두 rank0 리포트에 포함)

사용 (3장 전부 사용):
  # 학습 + 평가
  torchrun --nproc_per_node=3 05_train_qlora_ddp.py --data data/processed --out runs/ddp_7b
  # 기존 어댑터로 평가만 (청크 앙상블 + 피험자 집계 + CI)
  torchrun --nproc_per_node=3 05_train_qlora_ddp.py --eval_only \
      --adapter runs/qwen_qlora_client/adapter --chunks 3 --out runs/eval_agg
"""
import argparse, json, os, random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
# Compatibility shim: newer EXAONE remote code imports symbols absent in transformers 4.44.2.
try:
    from transformers.modeling_rope_utils import RopeParameters  # noqa: F401
except ImportError:
    import transformers.modeling_rope_utils as _mru
    _mru.RopeParameters = dict

from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          get_linear_schedule_with_warmup)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

CLASSES = ["우울증", "불안장애", "중독", "일반군"]
SYSTEM = ("당신은 심리상담 전사 분석 전문가입니다. 주어진 상담 대화 전사를 읽고 "
          "내담자의 상태를 다음 중 정확히 하나로 분류하세요: 우울증, 불안장애, 중독, 일반군. "
          "다른 말 없이 분류명만 출력하세요.")

def log0(rank, *a):
    if rank == 0:
        print(*a, flush=True)

def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def truncate_ids(ids, tok, budget):
    if len(ids) <= budget:
        return tok.decode(ids)
    head = int(budget * 0.7); tail = budget - head
    return tok.decode(ids[:head]) + "\n...(중략)...\n" + tok.decode(ids[-tail:])

def build_prompt(tok, text, answer=None):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"[상담 전사]\n{text}\n\n분류:"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt + (answer + tok.eos_token if answer is not None else "")

class SFTDS(Dataset):
    def __init__(self, recs, tok, max_len, field):
        self.recs, self.tok, self.max_len, self.field = recs, tok, max_len, field
        self.budget = max_len - 300

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        ids = self.tok(r[self.field], add_special_tokens=False)["input_ids"]
        text = truncate_ids(ids, self.tok, self.budget)
        full = build_prompt(self.tok, text, answer=r["cls"])
        prompt_only = build_prompt(self.tok, text)
        f = self.tok(full, truncation=True, max_length=self.max_len)["input_ids"]
        p_len = len(self.tok(prompt_only)["input_ids"])
        labels = [-100] * min(p_len, len(f)) + f[p_len:]
        return {"input_ids": f, "labels": labels[:len(f)]}

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
    return CLASSES.index("일반군")

def split_chunks(ids, n):
    """토큰 리스트를 n등분 (겹침 없음, 마지막이 나머지 흡수)"""
    if n <= 1 or len(ids) == 0:
        return [ids]
    step = max(1, len(ids) // n)
    out = [ids[i*step:(i+1)*step] for i in range(n - 1)]
    out.append(ids[(n-1)*step:])
    return [c for c in out if c]

@torch.no_grad()
def eval_shard(model, tok, recs, max_len, device, field, chunks, rank):
    """이 rank에 할당된 세션들을 생성 평가. 반환: 레코드별 dict 목록"""
    gen_model = model.module if isinstance(model, DDP) else model
    gen_model.eval()
    out = []
    for i, r in enumerate(recs):
        ids = tok(r[field], add_special_tokens=False)["input_ids"]
        parts = split_chunks(ids, chunks) if chunks > 1 else [ids]
        chunk_preds = []
        for part in parts:
            text = truncate_ids(part, tok, max_len - 300)
            prompt = build_prompt(tok, text)
            enc = tok(prompt, return_tensors="pt", truncation=True,
                      max_length=max_len).to(device)
            gen = gen_model.generate(**enc, max_new_tokens=8, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            txt = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            chunk_preds.append(parse_pred(txt))
        # 청크 다수결 (동률 시 첫 청크 우선)
        cnt = Counter(chunk_preds)
        top = cnt.most_common()
        pred = top[0][0]
        if len(top) > 1 and top[0][1] == top[1][1]:
            pred = chunk_preds[0]
        out.append({"id": r["id"], "subject": r["subject_id"],
                    "true": r["label"], "pred": int(pred),
                    "chunk_preds": chunk_preds})
        if rank == 0 and i % 10 == 0:
            print(f"  [rank0] eval {i}/{len(recs)}", flush=True)
    return out

def bootstrap_ci(y, p, n_boot=2000, seed=42, subjects=None):
    """macro-F1 95% CI. subjects 주어지면 피험자 단위 리샘플링(권장)"""
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

def subject_level(results):
    """피험자 단위 집계: 세션 예측 다수결(동률 시 세션 확률 평균 대신 첫 다수 클래스)"""
    by_s = defaultdict(list)
    true_s = {}
    for r in results:
        by_s[r["subject"]].append(r["pred"])
        true_s[r["subject"]] = r["true"]
    ys, ps, subs = [], [], []
    for s, preds in by_s.items():
        cnt = Counter(preds).most_common()
        pred = cnt[0][0]
        ys.append(true_s[s]); ps.append(pred); subs.append(s)
    return np.array(ys), np.array(ps), subs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="runs/ddp_qlora")
    ap.add_argument("--field", default="client_text", choices=["text", "client_text"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=3,
                    help="유효 배치 = bs x GPU수 x grad_accum (3GPU면 기본 9)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--oversample", default="",
                    help="예: '일반군:3' — 해당 클래스 train 샘플을 3배 복제")
    ap.add_argument("--chunks", type=int, default=1,
                    help=">1이면 평가 시 세션을 n청크로 나눠 다수결 앙상블")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--adapter", default="", help="eval_only 시 로드할 LoRA 어댑터 경로")
    ap.add_argument("--revision", default=None, help="pin HF model revision")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ── DDP 초기화 ──────────────────────────────────────────
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    data = Path(args.data)
    train, val, test = (load_jsonl(data / f"{k}.jsonl") for k in ("train", "val", "test"))
    log0(rank, f"[데이터] train={len(train)} val={len(val)} test={len(test)}  world={world}")

    # 오버샘플링 (분할 전에 리스트 복제 → DistributedSampler가 자동 분배)
    if args.oversample and not args.eval_only:
        cls_name, factor = args.oversample.split(":")
        factor = int(factor)
        extra = [r for r in train if r["cls"] == cls_name] * (factor - 1)
        train = train + extra
        random.Random(args.seed).shuffle(train)
        log0(rank, f"[오버샘플] {cls_name} x{factor} → train={len(train)} "
                   f"{dict(Counter(r['cls'] for r in train))}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, revision=args.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map={"": local}, attn_implementation="sdpa", trust_remote_code=True, revision=args.revision)

    if args.eval_only:
        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter)
            log0(rank, f"[어댑터 로드] {args.adapter}")
        else:
            log0(rank, "[zero-shot] 어댑터 없이 베이스 모델로 평가")
    else:
        model = prepare_model_for_kbit_training(model)
        tmods = (["q_proj","k_proj","v_proj","out_proj","c_fc_0","c_fc_1","c_proj"]
                 if "exaone" in args.model.lower() else
                 ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
        lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2,
                          lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                          target_modules=tmods)
        model = get_peft_model(model, lora)
        if rank == 0:
            model.print_trainable_parameters()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model = DDP(model, device_ids=[local], find_unused_parameters=False)

        ds = SFTDS(train, tok, args.max_len, args.field)
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                     shuffle=True, seed=args.seed)
        dl = DataLoader(ds, batch_size=args.bs, sampler=sampler,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr)
        steps_per_ep = len(dl) // args.grad_accum
        total = steps_per_ep * args.epochs
        sch = get_linear_schedule_with_warmup(opt, max(1, int(total * 0.03)), total)
        log0(rank, f"[학습] {steps_per_ep} steps/ep x {args.epochs}ep  "
                   f"(유효배치 {args.bs * world * args.grad_accum})")

        gstep = 0
        for ep in range(1, args.epochs + 1):
            sampler.set_epoch(ep)
            model.train()
            for step, b in enumerate(dl):
                out_ = model(input_ids=b["input_ids"].to(device),
                             attention_mask=b["attention_mask"].to(device),
                             labels=b["labels"].to(device))
                (out_.loss / args.grad_accum).backward()
                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step(); sch.step(); opt.zero_grad(); gstep += 1
                    if rank == 0 and gstep % 10 == 0:
                        print(f"ep{ep} gstep{gstep}/{total} loss={out_.loss.item():.4f}",
                              flush=True)
        if rank == 0:
            model.module.save_pretrained(out / "adapter")
            tok.save_pretrained(out / "adapter")
            print(f"[저장] {out}/adapter", flush=True)
        dist.barrier()

    # ── 분산 평가: test를 rank별로 분할 ─────────────────────
    shard = test[rank::world]
    log0(rank, f"[평가] chunks={args.chunks}  rank당 {len(shard)}세션 내외")
    my_results = eval_shard(model, tok, shard, args.max_len, device,
                            args.field, args.chunks, rank)
    gathered = [None] * world
    dist.all_gather_object(gathered, my_results)

    if rank == 0:
        results = [r for g in gathered for r in g]
        (out / "predictions.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in results),
            encoding="utf-8")
        y = np.array([r["true"] for r in results])
        p = np.array([r["pred"] for r in results])
        subs = [r["subject"] for r in results]

        lines = [f"model={args.model} field={args.field} max_len={args.max_len} "
                 f"epochs={args.epochs} lora_r={args.lora_r} chunks={args.chunks} "
                 f"oversample={args.oversample or '없음'} world={world}", ""]
        # 세션 수준
        f1 = f1_score(y, p, average="macro", zero_division=0)
        lo, hi = bootstrap_ci(y, p, subjects=np.array(subs))
        lines.append(f"[세션 수준] n={len(y)}  acc={accuracy_score(y,p):.4f}  "
                     f"macro-F1={f1:.4f}  (95% CI {lo:.3f}–{hi:.3f}, 피험자 리샘플링)")
        lines.append(classification_report(y, p, target_names=CLASSES,
                                           digits=4, zero_division=0))
        np.savetxt(out / "confusion_session.csv", confusion_matrix(y, p),
                   fmt="%d", delimiter=",")
        # 피험자 수준
        ys, ps, s_ids = subject_level(results)
        f1s = f1_score(ys, ps, average="macro", zero_division=0)
        lo2, hi2 = bootstrap_ci(ys, ps)
        lines.append(f"[피험자 수준] n={len(ys)}명  acc={accuracy_score(ys,ps):.4f}  "
                     f"macro-F1={f1s:.4f}  (95% CI {lo2:.3f}–{hi2:.3f})")
        lines.append(classification_report(ys, ps, target_names=CLASSES,
                                           digits=4, zero_division=0))
        np.savetxt(out / "confusion_subject.csv", confusion_matrix(ys, ps),
                   fmt="%d", delimiter=",")
        rep = "\n".join(lines)
        print(rep, flush=True)
        (out / "report.txt").write_text(rep, encoding="utf-8")
        print(f"[저장 완료] {out}/", flush=True)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
