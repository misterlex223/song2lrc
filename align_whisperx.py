#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_whisperx.py — 路線 A：GPU（Colab T4）對齊，音樂檔 + 歌詞 → words.json

做法：先讓 WhisperX 聽一遍拿到粗略的句子邊界，再用序列比對把這些時間
      貼回「你的歌詞」上。歌詞永遠以你給的為準，ASR 只貢獻時間。

Colab 用法：
  !pip install -q demucs whisperx
  !python align_whisperx.py song.mp3 lyrics.txt -o words.json --language zh

若已有乾淨人聲（Suno 分軌）：
  !python align_whisperx.py vocals.wav lyrics.txt -o words.json --no-demucs

輸出：words.json（交給本機的 json2lrc.py 轉 .lrc）
"""

import argparse
import difflib
import json
import os
import re
import sys
import tempfile

PUNCT = re.compile(r"[\s　,.\-—…!?;:'\"“”‘’()\[\]（）【】《》~～]+")


def norm(s: str) -> str:
    return PUNCT.sub("", s).lower()


def read_lyrics(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def separate_vocals(path: str, device: str) -> str:
    """用 Demucs 抽出人聲。沒有 GPU 時會很慢，建議直接用 Suno 分軌。"""
    import subprocess
    outdir = tempfile.mkdtemp(prefix="demucs_")
    cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals",
           "-n", "htdemucs", "-d", device, "-o", outdir, path]
    print("[demucs]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    wav = os.path.join(outdir, "htdemucs", stem, "vocals.wav")
    if not os.path.exists(wav):
        raise SystemExit(f"找不到 demucs 輸出：{wav}")
    return wav


def transcribe(audio: str, language: str, model_size: str, device: str, hint: str):
    import whisperx
    model = whisperx.load_model(model_size, device, compute_type="float16"
                                if device == "cuda" else "int8", language=language)
    result = model.transcribe(audio, language=language, initial_prompt=hint)
    segs = result.get("segments", [])

    # wav2vec2 做一次 CTC 對齊，把詞級時間戳補準
    try:
        model_a, meta = whisperx.load_align_model(language_code=language, device=device)
        aligned = whisperx.align(segs, model_a, meta, audio, device)
        segs = aligned.get("segments", segs)
    except Exception as e:
        print(f"[提示] 詞級對齊失敗，退回句級時間戳：{e}", file=sys.stderr)
    return segs


def attach(lyrics, segs):
    """
    用 LCS 把 ASR 句子對應到歌詞行。
    - 配對成功：沿用 ASR 的時間，文字用你的歌詞
    - ASR 幻覺（多出來的重複副歌）：丟掉
    - 你有的行但 ASR 漏了：在左右鄰居之間插值
    """
    a = [norm(x) for x in lyrics]
    b = [norm(s.get("text", "")) for s in segs if norm(s.get("text", ""))]
    keep = [s for s in segs if norm(s.get("text", ""))]

    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    mapping = {}          # lyric index -> (start, end, words)
    used_b = set()

    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            li, bj = i + k, j + k
            seg = keep[bj]
            mapping[li] = (float(seg["start"]), float(seg.get("end", seg["start"])),
                           seg.get("words") or [])
            used_b.add(bj)

    # 補漏網的行
    known = sorted(mapping.keys())
    for idx, line in enumerate(lyrics):
        if idx in mapping:
            continue
        prev_end = mapping[known[0]][1] if known else 0.0
        for k in known:
            if k < idx:
                prev_end = mapping[k][1]
        nxt = next((mapping[k][0] for k in known if k > idx), prev_end + 3.0)
        if nxt <= prev_end:
            nxt = prev_end + 3.0
        mapping[idx] = (prev_end, nxt, [])
        print(f"[警告] 第 {idx+1} 行「{line}」ASR 沒抓到，已插值。", file=sys.stderr)

    lines = []
    for idx, line in enumerate(lyrics):
        s, e, words = mapping[idx]
        lines.append({
            "text": line,
            "start": round(s, 3),
            "end": round(e, 3),
            "words": [{"w": w.get("word", "").strip(),
                       "start": round(float(w["start"]), 3),
                       "end": round(float(w.get("end", w["start"])), 3)}
                      for w in words if w.get("word", "").strip()],
        })
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("lyrics")
    p.add_argument("-o", "--output", default="words.json")
    p.add_argument("--language", default="zh")
    p.add_argument("--model", default="large-v3")
    p.add_argument("--device", default=None, help="cuda / cpu，預設自動")
    p.add_argument("--no-demucs", action="store_true", help="輸入已是乾淨人聲")
    args = p.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[環境] device={device}")

    lyrics = read_lyrics(args.lyrics)
    if not lyrics:
        raise SystemExit("歌詞是空的。")
    print(f"[歌詞] {len(lyrics)} 行")

    hint = " ".join(lyrics[:8])[:600]        # 餵給 Whisper 當提示，降低聽錯率

    audio = args.audio
    if not args.no_demucs:
        audio = separate_vocals(args.audio, device)

    segs = transcribe(audio, args.language, args.model, device, hint)
    print(f"[ASR] {len(segs)} 段")

    lines = attach(lyrics, segs)

    duration = None
    try:
        import soundfile as sf
        duration = len(sf.read(audio)[0]) / sf.read(audio)[1]
    except Exception:
        pass

    payload = {
        "meta": {"audio": os.path.basename(args.audio), "language": args.language,
                 "engine": "whisperx", "duration": duration},
        "lines": lines,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[輸出] {args.output}（{len(lines)} 行）")


if __name__ == "__main__":
    main()
