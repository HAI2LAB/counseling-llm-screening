#!/bin/bash
# run_symptom_cv.sh : 증상 태거 5-fold CV (fold를 GPU 3장에 병렬 배분, 총 ~2시간)
DATA_FOLDS=data/folds
mkdir -p logs runs
run_f () {
  local OUT=runs/cvsym_fold$1
  [ -f "$OUT/metrics.json" ] && { echo "[sym fold$1] 완료 — 건너뜀"; return; }
  CUDA_VISIBLE_DEVICES=$2 python 04_symptom_tagger.py \
    --data $DATA_FOLDS/fold$1 --label_group symptom --pw_cap 10 --epochs 3 \
    --out $OUT > logs/cvsym_fold$1.log 2>&1
}
echo "[$(date '+%T')] fold 0,1,2 병렬"; run_f 0 0 & run_f 1 1 & run_f 2 2 & wait
echo "[$(date '+%T')] fold 3,4 병렬";  run_f 3 0 & run_f 4 1 & wait
python 08_merge_symptom_cv.py | tee runs/cvsym_summary.txt
echo "완료 — runs/cvsym_summary.txt"
