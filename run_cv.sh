#!/bin/bash
# run_cv.sh : 14B 5-fold 교차검증 무인 실행 (fold당 약 4시간, 총 ~20시간)
# - 이미 완료된 fold(report.txt 존재)는 건너뛰므로 중단 후 재실행하면 이어서 진행
# - 전부 끝나면 07_merge_cv.py로 통합 리포트 생성
# 사용: tmux 안에서  bash run_cv.sh
DATA_FOLDS=data/folds
MODEL="Qwen/Qwen2.5-14B-Instruct"
mkdir -p logs runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
START=$(date '+%F %T')

# 0) fold 분할이 없으면 생성
if [ ! -f "$DATA_FOLDS/fold0/train.jsonl" ]; then
  echo "[$(date '+%T')] fold 분할 생성"
  python 06_make_folds.py --data data/processed --out $DATA_FOLDS || exit 1
fi

# 1) fold 0~4 순차 학습·평가
for k in 0 1 2 3 4; do
  OUT=runs/cv14b_fold$k
  if [ -f "$OUT/report.txt" ]; then
    echo "[$(date '+%T')] fold$k 이미 완료 — 건너뜀"
    continue
  fi
  echo "=============================================="
  echo "[$(date '+%T')] fold$k 시작 (test=fold$k 피험자)"
  echo "=============================================="
  torchrun --nproc_per_node=3 05_train_qlora_ddp.py \
    --model $MODEL \
    --data $DATA_FOLDS/fold$k \
    --epochs 2 --max_len 4096 --lora_r 16 \
    --oversample "일반군:3" --chunks 1 \
    --out $OUT 2>&1 | tee logs/cv14b_fold$k.log
  if [ ! -f "$OUT/report.txt" ]; then
    echo "[$(date '+%T')] fold$k 실패 — 로그 확인: logs/cv14b_fold$k.log"
    # OOM이면 경량 설정 1회 재시도
    if grep -q "OutOfMemoryError" logs/cv14b_fold$k.log; then
      echo "[$(date '+%T')] fold$k OOM → 3072/accum6 재시도"
      torchrun --nproc_per_node=3 05_train_qlora_ddp.py \
        --model $MODEL \
        --data $DATA_FOLDS/fold$k \
        --epochs 2 --max_len 3072 --lora_r 16 --grad_accum 6 \
        --oversample "일반군:3" --chunks 1 \
        --out $OUT 2>&1 | tee logs/cv14b_fold${k}_retry.log
    fi
  fi
done

# 2) 통합 리포트
echo "[$(date '+%T')] 5-fold 통합 리포트 생성"
python 07_merge_cv.py --pattern "runs/cv14b_fold{k}" --out runs/cv14b_final \
  2>&1 | tee logs/cv_merge.log

echo "실행: $START ~ $(date '+%F %T')"
echo "최종 결과: runs/cv14b_final/final_report.txt"
