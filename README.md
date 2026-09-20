# Counseling-LLM-Screening

Code for **"Label Leakage and Subject-Independent Evaluation in
LLM-Based Screening of Depression, Anxiety, and Addiction from Korean
Clinical Counseling Transcripts"** (submitted to *JMIR Mental Health*).

We fine-tune open-weight LLMs (QLoRA, on-premises) to screen four
groups (depression / anxiety disorder / addiction / control) from the
**client's utterances only** in real CBT counseling transcripts, under
**participant-independent 5-fold cross-validation**, and quantify how
much **counselor-utterance label leakage** inflates apparent
performance. An utterance-level symptom tagger (41 expert codes)
provides evidence highlighting.

## Key results (participant level, n=209)

| Model | Input | Session macro-F1 | Participant macro-F1 | Acc. |
|---|---|---|---|---|
| Qwen2.5-7B zero-shot | client-only | 0.466 | 0.562 | 60.8% |
| Qwen2.5-7B QLoRA | client-only | 0.660 | 0.759 | 78.5% |
| EXAONE-3.5-7.8B QLoRA | client-only | 0.691 | 0.798 | 81.8% |
| **Qwen2.5-14B QLoRA** | client-only | 0.673 | **0.820** | **83.7%** |
| Qwen2.5-14B QLoRA | full dialogue (leakage) | 0.816 | 0.895 | 91.9% |

## Data access (not included in this repository)

The corpus is the **"Psychological Counseling Data"** dataset from the
Open AI Dataset Project (AI-Hub, S. Korea), constructed by NIA:
<https://www.aihub.or.kr>. Access requires AI-Hub registration and
agreement to its terms of use. Per those terms, **transcripts must not
be redistributed or transferred abroad**; this repository therefore
contains **code only** — no data and no fine-tuned adapter weights
(withheld to preclude training-data memorization leakage). All
experiments were run on-premises (3x NVIDIA A100-40GB for training).

## Setup

```bash
pip install "numpy<2" "transformers==4.44.2" "peft==0.12.0" \
            "bitsandbytes==0.43.3" "accelerate==0.33.0" "triton==2.3.0" \
            scikit-learn matplotlib sentencepiece protobuf
```
Tested with Python 3.11, PyTorch 2.3.0, CUDA 12.1.

## Pipeline

| Step | Script | Purpose |
|---|---|---|
| 0 | `00_explore.py` | Corpus statistics; label-file integrity checks |
| 1 | `01_build_dataset.py` | Session records; speaker separation (client-only vs full); leakage keyword audit |
| 2 | `06_make_folds.py` | Participant-level stratified 5-fold splits |
| 3 | `02_train_encoder.py` | KLUE-RoBERTa encoder baseline |
| 4 | `05_train_qlora_ddp.py` | QLoRA fine-tuning + evaluation (DDP; zero-shot mode; oversampling; chunking; participant aggregation; cluster bootstrap) |
| 5 | `04_symptom_tagger.py` | Utterance-level 41-code symptom tagger |
| 6 | `07_merge_cv.py`, `08_merge_symptom_cv.py` | Merge fold results; pooled metrics + 95% CIs; McNemar tests |
| 7 | `09_make_random_session_folds.py` | Audit 1: naive random session-level folds |
| 8 | `10_build_masked_folds.py` | Audit 2: diagnostic/substance cue masking (dictionary + boundary rules) |
| 9 | `11_k_session_curve.py` | Audit 3: first-k-session aggregation curve (no retraining) |
| 10 | `12_tfidf_baseline.py` | Lexical TF-IDF + logistic regression baseline |
| - | `run_4set.sh` | Orchestrates all four audits (dryrun for CPU checks, gpu for training) |

Example (one fold, primary model):

```bash
torchrun --nproc_per_node=3 05_train_qlora_ddp.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --data data/folds/fold0 --epochs 2 --max_len 4096 --lora_r 16 \
  --oversample "일반군:3" --chunks 1 --out runs/cv14b_fold0
```

For EXAONE-3.5, pin the repository revision for compatibility with
transformers 4.44.2:

```bash
--model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
--revision 496aef060b296b34c6b0035149f5af9e2b8c168c
```

`run_*.sh` scripts reproduce every condition reported in the paper
(single-GPU x gradient-accumulation-9 runs are optimizer-equivalent to
3-GPU x accumulation-3; see paper Methods).

## Ethics

Secondary analysis of publicly released, de-identified transcripts; no
participant contact; IRB exemption determination by GIST (No. TBD).
Do not attempt re-identification.

## Citation

TBD upon publication.

## License

MIT (applies to the code in this repository only; the dataset is
governed by AI-Hub terms of use).
