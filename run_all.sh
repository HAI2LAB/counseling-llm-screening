#!/bin/bash
# run_all.sh : A100 3장 병렬 활용 — GPU마다 다른 실험 동시 실행
# 사용: bash run_all.sh   (tmux 세션 안에서 실행 권장)
set -e
DATA=data/processed

# counsel 가상환경 자동 활성화
if [ -f ~/venvs/counsel/bin/activate ]; then
  source ~/venvs/counsel/bin/activate
  echo "[env] counsel 활성화됨: $(which python)"
else
  echo "[경고] ~/venvs/counsel 없음 — 먼저 bash setup_env.sh 실행"; exit 1
fi

mkdir -p logs

# ── 1단계: 데이터셋 구축 (1회만) ─────────────────────────────
if [ ! -f "$DATA/train.jsonl" ]; then
  python 01_build_dataset.py --root /home/jovyan/HJW_counseling --out $DATA
fi

# ── 2단계: GPU별 병렬 실행 ──────────────────────────────────
# GPU0: 인코더 2종 순차 (가볍고 빠름)
CUDA_VISIBLE_DEVICES=0 nohup bash -c "
  python 02_train_encoder.py --data $DATA --model klue/roberta-base \
    --field client_text --out runs/roberta_client &&
  python 02_train_encoder.py --data $DATA --model monologg/koelectra-base-v3-discriminator \
    --field client_text --out runs/koelectra_client
" > logs/gpu0_encoder.log 2>&1 &

# GPU1: LLM zero-shot → QLoRA (내담자 발화만, 본 실험)
CUDA_VISIBLE_DEVICES=1 nohup bash -c "
  python 03_train_qlora.py --data $DATA --zero_shot_only \
    --field client_text --out runs/qwen_zs_client &&
  python 03_train_qlora.py --data $DATA \
    --field client_text --out runs/qwen_qlora_client
" > logs/gpu1_qlora.log 2>&1 &

# GPU2: 누수 대조 실험 (전체 대화 사용 시 성능 — '누수 효과' 정량화용)
CUDA_VISIBLE_DEVICES=2 nohup bash -c "
  python 02_train_encoder.py --data $DATA --model klue/roberta-base \
    --field text --out runs/roberta_full &&
  python 03_train_qlora.py --data $DATA \
    --field text --out runs/qwen_qlora_full
" > logs/gpu2_full.log 2>&1 &

echo '3개 GPU 작업 시작. 모니터링:'
echo '  tail -f logs/gpu0_encoder.log'
echo '  tail -f logs/gpu1_qlora.log'
echo '  tail -f logs/gpu2_full.log'
echo '  watch -n5 nvidia-smi'
wait
echo '전체 완료. 결과: runs/*/report*.txt'
