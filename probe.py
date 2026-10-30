#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe.py — 區間診斷工具

懷疑某段時間軸對不上時，單獨把那一段挖出來轉寫一次，
並印出每秒能量與 no_speech_prob，用來判斷：
  · 是 Whisper 漏聽（能量高、有唱）
  · 還是那段根本是間奏（能量低）

用法：
  python probe.py input/song.mp3 4 36 [--model small] [--language en]
  python probe.py input/song.mp3 4 36 --step 2      # 每 2 秒一片獨立轉寫
"""

import argparse
import sys

import numpy as np

from align_local import (TARGET_SR, load_audio_16k, resolve_model,
                         _transcribe_one, is_hallucination, norm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("start", type=float)
    p.add_argument("end", type=float)
    p.add_argument("--model", default="small")
    p.add_argument("--language", default="en")
    p.add_argument("--step", type=float, default=0.0,
                   help=">0 時改用這個長度（秒）的小切片逐片轉寫")
    p.add_argument("--nst", type=float, default=0.85)
    args = p.parse_args()

    audio = load_audio_16k(args.audio)
    total = len(audio) / TARGET_SR
    a = max(0.0, args.start)
    b = min(args.end if args.end > 0 else total, total)
    print(f"[音訊] {args.audio}  總長 {total:.1f}s  探查區間 {a:.1f}–{b:.1f}s")

    clip = audio[int(a * TARGET_SR):int(b * TARGET_SR)]

    # 每秒能量（dBFS），粗判有沒有聲音
    print("\n[能量] 每秒 RMS（dBFS，越接近 0 越大聲）")
    for t in range(int(a), int(b)):
        seg = audio[int(t * TARGET_SR):int((t + 1) * TARGET_SR)]
        if not len(seg):
            continue
        rms = float(np.sqrt(np.mean(seg ** 2)))
        db = 20 * np.log10(rms + 1e-9)
        bar = "#" * max(0, int((db + 60) / 2))
        print(f"  {t:5d}s {db:7.1f} dB {bar}")

    from faster_whisper import WhisperModel
    model = WhisperModel(resolve_model(args.model), device="cpu",
                         compute_type="int8")

    print(f"\n[轉寫] {args.model} / {args.language}")
    if args.step > 0:
        s = a
        while s < b - 0.5:
            e = min(s + args.step, b)
            c = audio[int(s * TARGET_SR):int(e * TARGET_SR)]
            for g in _transcribe_one(model, c, args.language, "", args.nst):
                junk = is_hallucination(g)
                flag = f"  ← 幻覺（{junk}）" if junk else ""
                print(f"  {s + g['start']:6.2f}→{s + g['end']:6.2f} "
                      f"{g['text'][:70]!r}{flag}")
            s = e
    else:
        segs, info = model.transcribe(
            clip, language=args.language, word_timestamps=True,
            vad_filter=False, condition_on_previous_text=False,
            no_speech_threshold=args.nst)
        segs = list(segs)
        print(f"  no_speech_prob / avg_logprob / compression_ratio")
        for g in segs:
            junk = is_hallucination({"text": g.text, "start": g.start,
                                     "end": g.end})
            flag = f"  ← 幻覺（{junk}）" if junk else ""
            print(f"  {a + g.start:6.2f}→{a + g.end:6.2f} "
                  f"nsp={g.no_speech_prob:.2f} lp={g.avg_logprob:.2f} "
                  f"cr={g.compression_ratio:.2f} {g.text[:60]!r}{flag}")
        if not segs:
            print("  （沒有任何輸出）")


if __name__ == "__main__":
    main()
