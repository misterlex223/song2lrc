#!/bin/bash
# LRC 工作流 — 環境安裝
#
# 請在你自己的 Terminal 執行（agent 的沙箱會擋 pip 暫存目錄）：
#   cd ~/WorkBuddy/AIGC-Music/lrc-workflow
#   chmod +x setup.sh && ./setup.sh
#
# 預設裝 fw 路線（不需要 torch，Intel Mac 一定裝得起來）
# 想要「真強制對齊」精度就跑： ./setup.sh qwen

set -euo pipefail
cd "$(dirname "$0")"
ROUTE="${1:-fw}"

if [ "$ROUTE" = "fw" ]; then

  echo "→ [1/3] 建立虛擬環境"
  PY="${PYTHON:-$(command -v python3.12 || command -v python3)}"
  echo "   使用 $PY"
  # 一定要先砍掉舊的：venv 重建時不會重設既有的 bin/python 捷徑，
  # 會留下「pyvenv.cfg 是 3.12、但 bin/python 指到別的版本」這種鬼狀態。
  rm -rf .venv
  "$PY" -m venv .venv

  echo "→ [2/3] 安裝 faster-whisper + soundfile"
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install faster-whisper soundfile

  echo "→ [3/3] 驗證"
  # 若 bin/python 跟 site-packages 版本不一致，import 會失敗
  .venv/bin/python - <<'CHK'
import sys, sysconfig, os
want = os.path.basename(sysconfig.get_paths()["purelib"])
have = "python%d.%d" % sys.version_info[:2]
assert want.endswith(have), f"版本不一致：site-packages={want} 直譯器={have}"
import faster_whisper, soundfile, ctranslate2
print(f"✓ 環境 OK — Python {have.split('python')[1]} / faster_whisper {faster_whisper.__version__}")
CHK
  echo ""
  echo "下一步：把音樂檔與 lyrics.txt 放進 input/，然後跑"
  echo "  .venv/bin/python align_local.py input/你的歌.mp3 input/lyrics.txt -o words.json --model small"
  echo "  python3 json2lrc.py words.json -o song.netease.lrc --netease"
  echo "  python3 json2lrc.py words.json -o song.kugou.lrc --ti '歌名' --ar '歌手'"

else

  echo "→ Qwen 強制對齊路線（torch 2.2.2 + Python 3.12）"
  echo "   注意：PyTorch 從 2.3 起不再提供 Intel Mac 輪子，只能鎖 2.2.2"
  if [ ! -x /usr/local/bin/python3.12 ]; then
    echo "✗ 找不到 python3.12，請先： brew install python@3.12"; exit 1
  fi
  /usr/local/bin/python3.12 -m venv .venv312
  .venv312/bin/pip install --upgrade pip
  .venv312/bin/pip install torch==2.2.2 torchaudio==2.2.2
  .venv312/bin/pip install --no-deps transformers==4.57.6
  .venv312/bin/pip install "tokenizers<1" huggingface-hub safetensors regex pyyaml packaging requests tqdm numpy jinja2
  .venv312/bin/pip install --no-deps qwen-asr soundfile

  echo ""
  .venv312/bin/python -c "import torch, transformers, qwen_asr; print('✓ 環境 OK, torch', torch.__version__)"
  echo ""
  echo "下一步："
  echo "  .venv312/bin/python align_qwen.py input/你的歌.wav input/lyrics.txt -o words.json"

fi
