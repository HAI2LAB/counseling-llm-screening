#!/bin/bash
# run_improve.sh : QLoRA 개선 실험 (A100 3장 병렬)
# 기준선: 2ep / 4096 / r16 → client macro-F1 0.656
set -e
DATA=data/processed
mkdir -p logs runs

# GPU0: 컨텍스트 확장 (4096 → 8192), 에폭 3
CUDA_VISIBLE_DEVICES=0 nohup python 03_train_qlora.py \
  --data $DATA --field client_text \
  --epochs 3 --max_len 8192 --lora_r 16 --grad_accum 8 \
  --out runs/qlora_c8k_e3 > logs/imp_gpu0_c8k_e3.log 2>&1 &

# GPU1: 에폭 확장 (4에폭), 컨텍스트는 4096 유지 → 에폭 효과 단독 측정
CUDA_VISIBLE_DEVICES=1 nohup python 03_train_qlora.py \
  --data $DATA --field client_text \
  --epochs 4 --max_len 4096 --lora_r 16 --grad_accum 8 \
  --out runs/qlora_c4k_e4 > logs/imp_gpu1_c4k_e4.log 2>&1 &

# GPU2: 용량 확장 (LoRA r=32) + 8192 + 3에폭
CUDA_VISIBLE_DEVICES=2 nohup python 03_train_qlora.py \
  --data $DATA --field client_text \
  --epochs 3 --max_len 8192 --lora_r 32 --lr 7e-5 --grad_accum 8 \
  --out runs/qlora_c8k_r32 > logs/imp_gpu2_c8k_r32.log 2>&1 &

echo "3개 실험 시작:"
echo "  GPU0  8192 / 3ep / r16   → runs/qlora_c8k_e3"
echo "  GPU1  4096 / 4ep / r16   → runs/qlora_c4k_e4"
echo "  GPU2  8192 / 3ep / r32   → runs/qlora_c8k_r32"
echo "모니터링: tail -f logs/imp_gpu0_c8k_e3.log"
wait
echo "완료. 비교:"
grep -H "macro-F1" runs/qlora_*/report_qlora.txt
