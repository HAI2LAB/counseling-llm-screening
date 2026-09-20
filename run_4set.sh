#!/bin/bash
# 4종 실험 일괄 실행.  1) bash run_4set.sh dryrun   ← CPU 단계만 + 탐지 출력 검증
#                      2) bash run_4set.sh gpu      ← 14B 랜덤분할 CV → 마스킹 CV (tmux 안에서!)
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs runs
M=Qwen/Qwen2.5-14B-Instruct

phase_cpu () {
  echo "=== [A] 무작위 세션분할 fold 생성 ==="; python3 09_make_random_session_folds.py || return 1
  echo "=== [B] 마스킹 fold 생성 ==="; python3 10_build_masked_folds.py || return 1
  echo "=== [C] k회기 누적 곡선 (기존 14B 예측 재활용) ==="
  python3 11_k_session_curve.py runs/cv14b_final/all_predictions.jsonl | tee runs/k_session_curve.txt
  echo "=== [D] TF-IDF 기준선: 원본 ==="; python3 12_tfidf_baseline.py data/folds | tee runs/tfidf_original.txt
  echo "=== [E] TF-IDF 기준선: 마스킹 ==="; python3 12_tfidf_baseline.py data/folds_masked | tee runs/tfidf_masked.txt
}

train_cv () {  # $1=fold_root  $2=out_prefix
  local pids=()
  for k in 0 1 2 3 4; do
    OUT=runs/${2}_fold$k
    if [ -f "$OUT/report.txt" ]; then echo "[$2 fold$k] 완료 — 건너뜀"; continue; fi
    CUDA_VISIBLE_DEVICES=$k torchrun --nproc_per_node=1 --master_port=$((29600+$k)) \
      05_train_qlora_ddp.py --model $M \
      --data $1/fold$k --epochs 2 --max_len 4096 --lora_r 16 --grad_accum 9 \
      --oversample "일반군:3" --chunks 1 \
      --out $OUT > logs/${2}_fold$k.log 2>&1 &
    pids+=($!)
    echo "[$2 fold$k] GPU$k 에서 시작 (pid $!)"
  done
  wait "${pids[@]}" 2>/dev/null
  python3 07_merge_cv.py --pattern "runs/${2}_fold{k}" --out runs/${2}_final
}

case "${1:-dryrun}" in
  dryrun) phase_cpu && echo -e "\n[다음] 탐지 출력에 이상 없으면: tmux 안에서  bash run_4set.sh gpu" ;;
  gpu)
    phase_cpu || { echo "CPU 단계 실패 — 중단"; exit 1; }
    echo "=== [F] 실험① 14B 무작위 세션분할 CV (5 GPU 병렬, ~8-10h) ==="
    train_cv data/folds_randsess cv14b_randsess
    echo "=== [G] 실험② 14B 마스킹 client-only CV (5 GPU 병렬, ~8-10h) ==="
    train_cv data/folds_masked cv14b_masked
    echo "=== 요약 ===" | tee runs/EXP4_SUMMARY.txt
    for f in runs/cv14b_randsess_final/final_report.txt runs/cv14b_masked_final/final_report.txt \
             runs/k_session_curve.txt runs/tfidf_original.txt runs/tfidf_masked.txt; do
      echo -e "\n##### $f" | tee -a runs/EXP4_SUMMARY.txt
      tail -30 "$f" 2>/dev/null | tee -a runs/EXP4_SUMMARY.txt
    done ;;
esac
