#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qc.py — words.json 品質檢查

跑完對齊後先過這關，看看時間軸能不能直接上平台，還是得進歌詞編輯器手修。

用法：
  python qc.py Draw-the-Line.words.json
  python qc.py Draw-the-Line.words.json -n 15      # 多列幾筆可疑行
"""

import argparse
import json
import sys


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = data.get("lines", data) if isinstance(data, dict) else data
    return lines, (data.get("meta", {}) if isinstance(data, dict) else {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("-n", "--show", type=int, default=10,
                    help="最多列出幾筆可疑行")
    args = ap.parse_args()

    lines, meta = load(args.json)
    if not lines:
        raise SystemExit("沒有資料。")

    dur = meta.get("duration")
    n = len(lines)
    print(f"檔案     {args.json}")
    print(f"引擎     {meta.get('engine', '?')}   語言 {meta.get('language', '?')}")
    print(f"音訊     {dur:.1f}s" if dur else "音訊     (未知)")
    print(f"行數     {n}")

    interp = [i for i, l in enumerate(lines) if l.get("_note")]
    print(f"插值行   {len(interp)}（{len(interp)/n*100:.0f}%）"
          f"{'  ← 比例偏高，對齊可能失敗' if len(interp) > n * 0.3 else ''}")

    # 1) 時間軸單調性
    bad_order = []
    for i in range(1, n):
        if lines[i]["start"] < lines[i - 1]["start"]:
            bad_order.append(i)
    print(f"順序顛倒 {len(bad_order)} 處")

    # 2) 異常長度
    short = [i for i, l in enumerate(lines) if l["end"] - l["start"] < 0.4]
    long_ = [i for i, l in enumerate(lines) if l["end"] - l["start"] > 15]
    print(f"過短(<0.4s) {len(short)} 行   過長(>15s) {len(long_)} 行")

    # 3) 行間空檔過大（可能是漏抓）
    gaps = []
    for i in range(1, n):
        g = lines[i]["start"] - lines[i - 1]["end"]
        if g > 6:
            gaps.append((i, g))
    print(f"行間空檔>6s {len(gaps)} 處")

    # 4) 超出音訊長度
    over = [i for i, l in enumerate(lines) if dur and l["end"] > dur + 0.5]
    print(f"超出音訊   {len(over)} 行")

    print(f"時間範圍   {lines[0]['start']:.2f}s → {lines[-1]['end']:.2f}s")
    if dur:
        print(f"覆蓋率     末行結束 / 音訊長度 = {lines[-1]['end']/dur*100:.0f}%"
              f"{'  ← 偏低，可能前面幾句沒對到' if lines[-1]['end']/dur < 0.5 else ''}")

    # 可疑行清單
    suspects = []
    for i in interp:
        suspects.append((i, "插值"))
    for i in bad_order:
        suspects.append((i, "時間倒退"))
    for i in short:
        suspects.append((i, "過短"))
    for i in long_:
        suspects.append((i, "過長"))
    for i, g in gaps:
        suspects.append((i, f"前一行結束後空檔 {g:.1f}s"))
    for i in over:
        suspects.append((i, "超出音訊尾端"))

    if suspects:
        seen, uniq = set(), []
        for i, why in sorted(suspects):
            if i not in seen:
                seen.add(i)
                uniq.append((i, why))
        print(f"\n可疑行（前 {min(args.show, len(uniq))} 筆，建議優先檢查）：")
        for i, why in uniq[:args.show]:
            l = lines[i]
            print(f"  第{i+1:3d}行 [{l['start']:6.2f}→{l['end']:6.2f}] {why}"
                  f"  {l['text'][:38]}")
        if len(uniq) > args.show:
            print(f"  …還有 {len(uniq)-args.show} 筆")
    else:
        print("\n沒有可疑行，時間軸看起來是乾淨的。")

    score = 100
    score -= min(60, len(interp) / n * 100)
    score -= min(20, len(bad_order) * 5)
    score -= min(20, len(short) * 2)
    print(f"\n健康度   {max(0, int(score))}/100"
          f"  {'可直接上平台' if score >= 85 else '建議進歌詞編輯器修一下'}")


if __name__ == "__main__":
    main()
