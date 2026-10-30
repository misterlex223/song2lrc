#!/bin/bash
# 批次處理 input/ 裡所有還沒跑過的歌
#
#   ./batch.sh                      跑所有還沒有成品的歌
#   FORCE=1 ./batch.sh              全部重跑（含已經跑過的）
#   ARTIST="LMS2" ./batch.sh        指定歌手名（預設 LMS2）
#   LANG_CODE=en ./batch.sh         指定語言（預設 en）
#
# 歌名預設由檔名推導（My-Own-Gravity.mp3 → "My Own Gravity"）。
# 歌詞檔會自動配對：input/<歌名>_lyrics.txt 或 input/<歌名>-lyrics.txt。
# 單首失敗不會中斷整批，最後會列出失敗清單。

set -uo pipefail
cd "$(dirname "$0")"
shopt -s nullglob

ARTIST="${ARTIST:-LMS2}"
LANG_CODE="${LANG_CODE:-en}"
FAILED=()
DONE=()
SKIPPED=()

for f in input/*.mp3 input/*.wav input/*.m4a input/*.flac; do
  [[ -e "$f" ]] || continue
  base="$(basename "${f%.*}")"
  out="output/${base}"

  if [[ -f "$out/${base}.words.json" && -z "${FORCE:-}" ]]; then
    SKIPPED+=("$base")
    continue
  fi

  lyrics=""
  for cand in "input/${base}_lyrics.txt" "input/${base}-lyrics.txt" "input/${base}.txt"; do
    if [[ -f "$cand" ]]; then lyrics="$cand"; break; fi
  done
  if [[ -z "$lyrics" ]]; then
    echo "⚠ 找不到 ${base} 的歌詞檔，跳過"
    FAILED+=("$base（缺歌詞）")
    continue
  fi

  title="$(echo "$base" | tr '-' ' ')"
  echo ""
  echo "════════════════════════════════════════"
  echo "  ${title}"
  echo "════════════════════════════════════════"
  if ./run.sh "$f" "$lyrics" "$title" "$ARTIST" "$LANG_CODE"; then
    DONE+=("$base")
  else
    FAILED+=("$base")
  fi
done

echo ""
echo "════════════ 批次結束 ════════════"
echo "本次完成 ${#DONE[@]} 首：${DONE[*]:-（無）}"
echo "已跳過   ${#SKIPPED[@]} 首：${SKIPPED[*]:-（無）}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "失敗     ${#FAILED[@]} 首：${FAILED[*]}"
  exit 1
fi
echo "全部成功。"
