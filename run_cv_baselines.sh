#!/bin/bash
# run_cv_baselines.sh : 논문 결과표 빈칸 3종 무인 실행 (총 ~24시간)
#  1) zero-shot 7B — fold별 평가만 (~2시간)
#  2) 인코더 RoBERTa CV — fold를 GPU 3장에 병렬 배분 (~1.5시간)
#  3) 전체대화(text) 14B CV — 누수 대조의 교차검증판 (~20시간)
# 완료된 fold는 건너뛰므로 중단 후 재실행하면 이어서 진행.
# 사용: tmux 안에서  bash run_cv_baselines.sh
DATA_FOLDS=data/folds
mkdir -p logs runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
START=$(date '+%F %T')

[ -f "$DATA_FOLDS/fold0/train.jsonl" ] || { echo "fold 분할 없음 — 06_make_folds.py 먼저"; exit 1; }

# ── 1) zero-shot 7B (fold별 test 평가만) ─────────────────────
for k in 0 1 2 3 4; do
  OUT=runs/cvzs_fold$k
  [ -f "$OUT/report.txt" ] && { echo "[zs fold$k] 완료 — 건너뜀"; continue; }
  echo "[$(date '+%T')] zero-shot fold$k"
  torchrun --nproc_per_node=3 05_train_qlora_ddp.py --eval_only \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data $DATA_FOLDS/fold$k --chunks 1 \
    --out $OUT 2>&1 | tee logs/cvzs_fold$k.log
done
python 07_merge_cv.py --pattern "runs/cvzs_fold{k}" --out runs/cvzs_final \
  2>&1 | tee logs/cvzs_merge.log || true

# ── 2) 인코더 RoBERTa CV (fold를 GPU에 병렬 배분: 0,1,2 → 3,4) ──
run_enc () {  # $1=fold  $2=gpu
  local OUT=runs/cvrob_fold$1
  [ -f "$OUT/report.txt" ] && { echo "[rob fold$1] 완료 — 건너뜀"; return; }
  CUDA_VISIBLE_DEVICES=$2 python 02_train_encoder.py \
    --data $DATA_FOLDS/fold$1 --model klue/roberta-base \
    --field client_text --out $OUT > logs/cvrob_fold$1.log 2>&1
}
echo "[$(date '+%T')] 인코더 CV: fold 0,1,2 병렬"
run_enc 0 0 & run_enc 1 1 & run_enc 2 2 & wait
echo "[$(date '+%T')] 인코더 CV: fold 3,4 병렬"
run_enc 3 0 & run_enc 4 1 & wait
{ echo "[인코더 RoBERTa fold별 test macro-F1]"
  for k in 0 1 2 3 4; do
    v=$(grep -oP 'macro-F1=\K[0-9.]+' runs/cvrob_fold$k/report.txt 2>/dev/null | tail -1)
    echo "  fold$k: ${v:-실패}"
  done
} | tee runs/cvrob_summary.txt

# ── 3) 전체대화(text) 14B CV — 누수 대조 ─────────────────────
for k in 0 1 2 3 4; do
  OUT=runs/cv14b_full_fold$k
  [ -f "$OUT/report.txt" ] && { echo "[full fold$k] 완료 — 건너뜀"; continue; }
  echo "=============================================="
  echo "[$(date '+%T')] 전체대화 14B fold$k 시작"
  torchrun --nproc_per_node=3 05_train_qlora_ddp.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --data $DATA_FOLDS/fold$k --field text \
    --epochs 2 --max_len 4096 --lora_r 16 \
    --oversample "일반군:3" --chunks 1 \
    --out $OUT 2>&1 | tee logs/cv14b_full_fold$k.log
  if [ ! -f "$OUT/report.txt" ] && grep -q "OutOfMemoryError" logs/cv14b_full_fold$k.log; then
    echo "[fold$k] OOM → 3072/accum6 재시도"
    torchrun --nproc_per_node=3 05_train_qlora_ddp.py \
      --model Qwen/Qwen2.5-14B-Instruct \
      --data $DATA_FOLDS/fold$k --field text \
      --epochs 2 --max_len 3072 --lora_r 16 --grad_accum 6 \
      --oversample "일반군:3" --chunks 1 \
      --out $OUT 2>&1 | tee logs/cv14b_full_fold${k}_retry.log
  fi
done
python 07_merge_cv.py --pattern "runs/cv14b_full_fold{k}" --out runs/cv14b_full_final \
  2>&1 | tee logs/cvfull_merge.log || true

# ── 최종 요약 ─────────────────────────────────────────────
{
  echo "실행: $START ~ $(date '+%F %T')"
  echo
  echo "===== 논문 결과표 (5-fold CV, n=1500세션/209명) ====="
  echo "[zero-shot 7B]"
  grep -E "^\[세션 수준·통합\]|^\[피험자 수준·통합\]" runs/cvzs_final/final_report.txt 2>/dev/null || echo "  없음"
  echo
  echo "[인코더 RoBERTa (fold별)]"
  cat runs/cvrob_summary.txt 2>/dev/null | tail -6
  echo
  echo "[14B QLoRA — 내담자 발화 (주 결과)]"
  grep -E "^\[세션 수준·통합\]|^\[피험자 수준·통합\]" runs/cv14b_final/final_report.txt 2>/dev/null || echo "  없음"
  echo
  echo "[14B QLoRA — 전체 대화 (누수 대조)]"
  grep -E "^\[세션 수준·통합\]|^\[피험자 수준·통합\]" runs/cv14b_full_final/final_report.txt 2>/dev/null || echo "  없음"
} | tee summary_baselines.txt

echo "전체 종료 — summary_baselines.txt 확인"
