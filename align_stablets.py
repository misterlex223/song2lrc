#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_stablets.py — 本機 CPU 強制對齊：音樂檔 + 歌詞 → words.json

用 faster-whisper（CTranslate2 核心）+ stable-ts 的 align()。
不需要 torch，Intel Mac 也能跑。

它不做聽寫：你把歌詞餵進去，它只回傳每個字在音軌上的起訖時間。

用法：
  .venv/bin/python align_stablets.py input/song.mp3 input/lyrics.txt -o words.json
  .venv/bin/python align_stablets.py input/vocals.wav input/lyrics.txt -o words.json --model small

--model 選擇（CPU 上越往下越慢越準）：
  tiny / base / small / medium / large-v3
中文建議 small 起步；覺得不準再上 medium。
"""

import argparse
import json
import os
import re
import sys

PUNCT = re.compile(r"[\s　,.\-—…!?;:'\"“”‘’()\[\]（）【】《》~～]+")


def read_lyrics(path: str):
    """讀歌詞，回傳 (原文行, 去掉標點的行)"""
    raw_lines, clean_lines = [], []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            c = PUNCT.sub("", s)
            if not c:
                continue
            raw_lines.append(s)
            clean_lines.append(c)
    return raw_lines, clean_lines


def duration_of(path: str):
    try:
        import soundfile as sf
        return float(sf.info(path).duration)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("lyrics")
    p.add_argument("-o", "--output", default="words.json")
    p.add_argument("--model", default="small",
                   help="tiny / base / small / medium / large-v3")
    p.add_argument("--language", default="zh")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    raw_lines, clean_lines = read_lyrics(args.lyrics)
    if not clean_lines:
        raise SystemExit("歌詞是空的（或去掉標點後全空）。")

    dur = duration_of(args.audio)
    print(f"[音訊] {os.path.basename(args.audio)}"
          + (f"  {dur:.1f}s" if dur else ""))
    print(f"[歌詞] {len(clean_lines)} 行 / {sum(len(c) for c in clean_lines)} 字")

    import stable_whisper
    print(f"[載入] faster-whisper '{args.model}'（首次會下載模型，請稍候）")
    model = stable_whisper.load_faster_whisper(args.model, device=args.device)

    # 用換行當行界，stable-ts 會依斷行切成 segments
    text = "\n".join(clean_lines)
    print(f"[對齊] language={args.language}，正在計算字級時間戳…")
    result = model.align(args.audio, text, language=args.language)

    segs = list(result.segments)
    print(f"[對齊] 取得 {len(segs)} 段")

    lines = []
    for idx, seg in enumerate(segs):
        orig = raw_lines[idx] if idx < len(raw_lines) else str(seg.text).strip()
        words = []
        for w in (getattr(seg, "words", None) or []):
            ws = str(getattr(w, "word", "")).strip()
            if not ws:
                continue
            words.append({
                "w": ws,
                "start": round(float(w.start), 3),
                "end": round(float(w.end) if w.end is not None else float(w.start), 3),
            })
        start = float(seg.start) if seg.start is not None else (words[0]["start"] if words else 0.0)
        end = float(seg.end) if seg.end is not None else (words[-1]["end"] if words else start)
        lines.append({"text": orig, "start": round(start, 3),
                      "end": round(end, 3), "words": words})

    if len(lines) != len(clean_lines):
        print(f"[警告] 歌詞 {len(clean_lines)} 行，對齊回傳 {len(segs)} 段，"
              f"行數對不上，請人工複核。", file=sys.stderr)

    payload = {
        "meta": {"audio": os.path.basename(args.audio), "language": args.language,
                 "engine": f"stable-ts/{args.model}", "duration": dur},
        "lines": lines,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[輸出] {args.output}（{len(lines)} 行）")


if __name__ == "__main__":
    main()
