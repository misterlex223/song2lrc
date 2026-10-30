#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_qwen.py — 路線 B：本機純 CPU，音樂檔 + 歌詞 → words.json

用 Qwen3-ForcedAligner-0.6B（阿里開源，Apache-2.0）。中文逐字精度目前最好，
0.6B 在 8GB RAM 的 Intel CPU 上跑得動，一首 4 分鐘歌約 8–15 分鐘。

它不做語音辨識，只做強制對齊：你給的歌詞就是標準答案，它只回答「哪個字在第幾秒」。

安裝：
  pip install torch torchaudio transformers soundfile numpy
  pip install qwen-asr

用法：
  python align_qwen.py song.wav lyrics.txt -o words.json
  python align_qwen.py vocals.wav lyrics.txt -o words.json --device mps   # Apple Silicon

限制：單次最多 ~5 分鐘音訊。超過會直接報錯，請改用 align_whisperx.py（GPU 路線）。
"""

import argparse
import json
import os
import re
import sys

PUNCT = re.compile(r"[\s　,.\-—…!?;:'\"“”‘’()\[\]（）【】《》~～]+")
MAX_SECONDS = 290


def read_lyrics(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def duration_of(path: str) -> float:
    import soundfile as sf
    info = sf.info(path)
    return info.duration


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("lyrics")
    p.add_argument("-o", "--output", default="words.json")
    p.add_argument("--device", default=None, help="cuda / mps / cpu，預設自動")
    p.add_argument("--language", default="Chinese")
    args = p.parse_args()

    lyrics = read_lyrics(args.lyrics)
    if not lyrics:
        raise SystemExit("歌詞是空的。")

    dur = duration_of(args.audio)
    print(f"[音訊] {os.path.basename(args.audio)}  {dur:.1f}s")
    if dur > MAX_SECONDS:
        raise SystemExit(
            f"音訊 {dur:.0f}s 超過 Qwen aligner 單次 {MAX_SECONDS}s 上限。\n"
            "請改用 align_whisperx.py（Colab GPU），或先分段再合併。"
        )

    # 標點不發音，對齊前先拿掉；行文字仍保留原文
    stripped = [PUNCT.sub("", l) for l in lyrics if PUNCT.sub("", l)]
    if not stripped:
        raise SystemExit("去掉標點後歌詞是空的。")
    full = "".join(stripped)
    print(f"[歌詞] {len(stripped)} 行 / {len(full)} 字")

    import torch
    from qwen_asr import Qwen3ForcedAligner

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[環境] device={device}")

    model = Qwen3ForcedAligner.from_pretrained(
        "Qwen/Qwen3-ForcedAligner-0.6B",
        dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map=device,
    )

    results = model.align(audio=args.audio, text=full, language=args.language)
    items = results[0]
    print(f"[對齊] 取得 {len(items)} 個字級時間戳")

    if len(items) != len(full):
        print(f"[警告] 字數 {len(full)} 與對齊數 {len(items)} 不符，"
              f"將按比例切回各行，請人工複核。", file=sys.stderr)

    # 依每行字數把字級時間戳切回行
    lines, idx = [], 0
    for text, stext in zip(lyrics, stripped):
        n = len(stext)
        chunk = items[idx:idx + n]
        idx += n
        if not chunk:
            continue
        start = float(chunk[0].start_time)
        end = float(chunk[-1].end_time)
        words = [{"w": c.text, "start": round(float(c.start_time), 3),
                  "end": round(float(c.end_time), 3)} for c in chunk]
        lines.append({"text": text, "start": round(start, 3),
                      "end": round(end, 3), "words": words})

    payload = {
        "meta": {"audio": os.path.basename(args.audio), "language": args.language,
                 "engine": "qwen-forced-aligner", "duration": dur},
        "lines": lines,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[輸出] {args.output}（{len(lines)} 行）")


if __name__ == "__main__":
    main()
