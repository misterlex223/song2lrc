#!/bin/bash
# 一鍵跑完整條流程（在你自己的 Terminal 執行）
#
#   ./run.sh input/你的歌.mp3 input/lyrics.txt "歌名" "歌手" [語言]
#   語言預設 zh；英文歌請傳 en，例如：
#   ./run.sh input/song.mp3 input/lyrics.txt "Draw the Line" "LMS2" en
#
# 環境變數：
#   MODEL=small          模型大小，tiny / base / small / medium
#   FORCE=1              已對齊過也重跑（預設會跳過，直接沿用 words.json）
#
# 產出全部集中在 output/<歌名>/：
#   <歌名>.words.json        字級時間戳（中介檔，可進 git）
#   <歌名>.netease.lrc       網易雲版：標準逐行、無 metadata
#   <歌名>.kugou.lrc         酷狗/QQ 版：含 metadata
#   <歌名>.kugou.word.lrc    酷狗/QQ 逐字版（卡拉OK 高亮）
#   align.log               對齊過程紀錄

set -euo pipefail
cd "$(dirname "$0")"

AUDIO="${1:?用法: ./run.sh input/歌.mp3 input/lyrics.txt [歌名] [歌手]}"
LYRICS="${2:?缺少歌詞檔}"
TITLE="${3:-}"
ARTIST="${4:-}"
BASE="$(basename "${AUDIO%.*}")"

OUT="output/${BASE}"
mkdir -p "$OUT"

# 中國大陸連 HuggingFace 很慢，預設走鏡像；要用官方就 HF_ENDPOINT=https://huggingface.co
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
MODEL="${MODEL:-small}"                  # tiny / base / small / medium
LANG_CODE="${5:-${LANG_CODE:-zh}}"       # zh / en / yue，也可用第 5 個參數指定

# 注意：bash 3.2 在 UTF-8 環境下會把緊跟在 $VAR 後面的全形字元
# 當成變數名稱的一部分，所以變數一律用 ${...} 包起來。

cp -f "$LYRICS" "$OUT/${BASE}.lyrics.txt"   # 留存這版歌詞，對齊結果才對得回來

if [[ -f "$OUT/${BASE}.words.json" && -z "${FORCE:-}" ]]; then
  echo "→ [1/5] 已有對齊結果，跳過對齊（要重跑請加 FORCE=1）"
else
  echo "→ [1/5] 對齊（faster-whisper ${MODEL} / ${LANG_CODE}），首次會下載模型"
  echo "       350 秒的歌在 CPU 上約需 17 分鐘，去泡杯咖啡"
  .venv/bin/python align_local.py "$AUDIO" "$LYRICS" \
    -o "$OUT/${BASE}.words.json" --model "$MODEL" --language "$LANG_CODE" \
    2>&1 | tee "$OUT/align.log"
fi

echo "→ [2/5] 網易雲版"
python3 json2lrc.py "$OUT/${BASE}.words.json" -o "$OUT/${BASE}.netease.lrc" --netease

echo "→ [3/5] 酷狗 / QQ 版"
python3 json2lrc.py "$OUT/${BASE}.words.json" -o "$OUT/${BASE}.kugou.lrc" \
  ${TITLE:+--ti "$TITLE"} ${ARTIST:+--ar "$ARTIST"}

echo "→ [4/5] 酷狗 / QQ 逐字版"
python3 json2lrc.py "$OUT/${BASE}.words.json" -o "$OUT/${BASE}.kugou.word.lrc" \
  --enhanced ${TITLE:+--ti "$TITLE"} ${ARTIST:+--ar "$ARTIST"}

echo "→ [5/5] 品檢"
python3 qc.py "$OUT/${BASE}.words.json"

echo ""
echo "✓ 完成，成品在 $(pwd)/$OUT/"
ls -la "$OUT"
