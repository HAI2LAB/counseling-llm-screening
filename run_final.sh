#!/bin/bash
# run_final.sh : 무인 일괄 실행 (A1 → B1 → C 14B), 총 4~5시간
# 한 단계가 실패해도 다음 단계는 계속 진행. 마지막에 결과 요약을 summary.txt로 저장.
# 사용: tmux 안에서  bash run_final.sh
DATA=data/processed
mkdir -p logs runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T="torchrun --nproc_per_node=3"
START=$(date '+%F %T')

run_stage () {  # $1=이름  $2...=명령
  local name=$1; shift
  echo "=============================================="
  echo "[$(date '+%T')] $name 시작"
  echo "=============================================="
  if "$@"; then
    echo "[$(date '+%T')] $name 완료"
  else
    echo "[$(date '+%T')] $name 실패 — 다음 단계로 진행 (로그 확인 필요)"
  fi
}

# ── A1: 기존 최고 7B 모델 평가 (청크 없음, 피험자 집계 + CI) ──
run_stage "A1 (기존 7B 재평가)" \
  $T 05_train_qlora_ddp.py --eval_only \
    --adapter runs/qwen_qlora_client/adapter --data $DATA --chunks 1 \
    --out runs/A1_nochunk 2>&1 | tee logs/ddp_A1.log

# ── B1: 오버샘플 7B 모델 평가 (동일 조건) ──
run_stage "B1 (오버샘플 7B 재평가)" \
  $T 05_train_qlora_ddp.py --eval_only \
    --adapter runs/B_7b_oversample/adapter --data $DATA --chunks 1 \
    --out runs/B1_nochunk 2>&1 | tee logs/ddp_B1.log

# ── C: 14B 학습 + 평가 (오버샘플, 4096) ──
run_stage "C (14B 학습)" \
  $T 05_train_qlora_ddp.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --data $DATA --epochs 2 --max_len 4096 --lora_r 16 \
    --oversample "일반군:3" --chunks 1 \
    --out runs/C_14b_v2 2>&1 | tee logs/ddp_C2.log

# C가 OOM으로 실패했으면 경량 설정으로 1회 자동 재시도
if [ ! -f runs/C_14b_v2/report.txt ] && grep -q "OutOfMemoryError" logs/ddp_C2.log 2>/dev/null; then
  echo "[$(date '+%T')] C단계 OOM 감지 → 3072/accum6으로 자동 재시도"
  run_stage "C-재시도 (14B, 3072)" \
    $T 05_train_qlora_ddp.py \
      --model Qwen/Qwen2.5-14B-Instruct \
      --data $DATA --epochs 2 --max_len 3072 --lora_r 16 --grad_accum 6 \
      --oversample "일반군:3" --chunks 1 \
      --out runs/C_14b_v2 2>&1 | tee logs/ddp_C3.log
fi

# ── 결과 요약 ──
{
  echo "실행: $START ~ $(date '+%F %T')"
  echo
  for d in A1_nochunk B1_nochunk C_14b_v2; do
    echo "===== runs/$d ====="
    if [ -f "runs/$d/report.txt" ]; then
      grep -E "^\[세션 수준\]|^\[피험자 수준\]" "runs/$d/report.txt"
      grep -A6 "피험자 수준" "runs/$d/report.txt" | grep -E "우울증|불안장애|중독|일반군"
    else
      echo "(리포트 없음 — 실패, logs 확인)"
    fi
    echo
  done
} | tee summary.txt

echo "전체 종료. 요약: summary.txt / 상세: runs/*/report.txt"
