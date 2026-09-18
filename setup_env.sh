#!/bin/bash
# setup_env.sh : counsel 가상환경 생성 (기존 conda 환경의 torch/CUDA 재사용)
# 사용: bash setup_env.sh
#   이후 활성화: source ~/venvs/counsel/bin/activate
set -e
ENV_DIR=~/venvs/counsel

echo "== 1) 기존 환경의 torch 확인 =="
python - << 'EOF'
import torch, sys
print(f"python {sys.version.split()[0]} / torch {torch.__version__} / cuda {torch.version.cuda} / GPU {torch.cuda.device_count()}장")
EOF

echo "== 2) venv 생성 (--system-site-packages: torch 재사용) =="
mkdir -p ~/venvs
python -m venv --system-site-packages "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip -q

echo "== 3) 패키지 설치 =="
pip install -r requirements.txt

echo "== 4) 검증 =="
python - << 'EOF'
import torch, transformers, peft, bitsandbytes, accelerate, sklearn, numpy
print(f"torch        {torch.__version__} (cuda {torch.version.cuda}, GPU {torch.cuda.device_count()}장)")
print(f"transformers {transformers.__version__}")
print(f"peft         {peft.__version__}")
print(f"bitsandbytes {bitsandbytes.__version__}")
print(f"accelerate   {accelerate.__version__}")
print(f"numpy        {numpy.__version__}  (2.x면 안 됨)")
# bitsandbytes 4bit 동작 스모크 테스트
from transformers import BitsAndBytesConfig
print("BitsAndBytesConfig import OK")
assert torch.cuda.is_available(), "CUDA 미인식!"
print("\n[완료] counsel 환경 사용 준비 끝")
print("활성화:  source ~/venvs/counsel/bin/activate")
EOF
