#!/bin/bash
# run_ddp.sh : 성능 개선 실험 — 각 단계가 GPU 3장을 전부 사용, 단계는 순차 실행
# 사용: tmux 안에서  bash run_ddp.sh
set -e
DATA=data/processed
mkdir -p logs runs
T="torchrun --nproc_per_node=3"

echo "== [A단계] 기존 7B 어댑터 재평가: 청크 앙상블 + 피험자 집계 + CI (약 30분) =="
$T 05_train_qlora_ddp.py --eval_only \
  --adapter runs/qwen_qlora_client/adapter \
  --data $DATA --chunks 3 \
  --out runs/A_eval_chunks 2>&1 | tee logs/ddp_A.log

echo "== [B단계] 7B 재학습: 일반군 3배 오버샘플 + 청크 평가 (약 1.5~2시간) =="
$T 05_train_qlora_ddp.py \
  --data $DATA --epochs 2 --max_len 4096 --lora_r 16 \
  --oversample "일반군:3" --chunks 3 \
  --out runs/B_7b_oversample 2>&1 | tee logs/ddp_B.log

echo "== [C단계] 14B 스케일업: 오버샘플 + 청크 평가 (약 3~4시간) =="
$T 05_train_qlora_ddp.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --data $DATA --epochs 2 --max_len 4096 --lora_r 16 \
  --oversample "일반군:3" --chunks 3 \
  --out runs/C_14b_oversample 2>&1 | tee logs/ddp_C.log

echo "== 전체 완료 — 결과 비교 =="
grep -H "수준" runs/A_eval_chunks/report.txt runs/B_7b_oversample/report.txt runs/C_14b_oversample/report.txt | grep macro
