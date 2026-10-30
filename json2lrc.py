#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
json2lrc.py — 把字級時間戳 JSON 轉成音樂平台要的 .lrc

純標準函式庫，不需要 torch / ffmpeg，任何機器都跑得動。

支援輸入格式（自動判別）：
  words     本專案的統一中介格式 {"meta":..., "lines":[{"text","start","end","words":[...]}]}
  whisperx  WhisperX 原始輸出    {"segments":[{"text","start","end","words":[{"word","start","end"}]}]}
  qwen      Qwen3-ForcedAligner  [{"text","start_time","end_time"}, ...]

用法：
  python json2lrc.py words.json -o song.lrc
  python json2lrc.py words.json -o song.lrc --netease
  python json2lrc.py words.json -o song.lrc --enhanced
  python json2lrc.py words.json -o song.lrc --shift-ms -180
  python json2lrc.py words.json -o song.lrc --ti "歌名" --ar "歌手"
  python json2lrc.py words.json --lyrics lyrics.txt -o song.lrc   # 以 lyrics.txt 的文字為準
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

SEC_PER_MIN = 60


# ---------- 時間戳格式化 ----------

def fmt_lrc(t: float) -> str:
    """秒 → [mm:ss.xx]，毫秒固定兩位。負值一律當 0。"""
    if t < 0:
        t = 0.0
    cs = int(round(t * 100))          # 總厘秒
    mm = cs // 6000
    ss = (cs % 6000) // 100
    xx = cs % 100
    return f"[{mm:02d}:{ss:02d}.{xx:02d}]"


def fmt_word(t: float) -> str:
    """增強 LRC 的字級標籤 <mm:ss.xx>"""
    if t < 0:
        t = 0.0
    cs = int(round(t * 100))
    mm = cs // 6000
    ss = (cs % 6000) // 100
    xx = cs % 100
    return f"<{mm:02d}:{ss:02d}.{xx:02d}>"


# ---------- 讀入各種對齊引擎的輸出 ----------

def detect_format(data: Any) -> str:
    if isinstance(data, list):
        return "qwen"
    if isinstance(data, dict):
        if "lines" in data:
            return "words"
        if "segments" in data:
            return "whisperx"
    raise SystemExit("無法判別 JSON 格式：請用 words / whisperx / qwen 之一")


def load_qwen(data: List[Dict]) -> List[Dict]:
    """Qwen3-ForcedAligner 輸出：一串字（中文逐字）。"""
    items = []
    for d in data:
        items.append({
            "w": str(d.get("text", "")).strip(),
            "start": float(d.get("start_time", 0.0)),
            "end": float(d.get("end_time", 0.0)),
        })
    return [{"text": "".join(i["w"] for i in items),
             "start": items[0]["start"] if items else 0.0,
             "end": items[-1]["end"] if items else 0.0,
             "words": items}]


def load_whisperx(data: Dict) -> List[Dict]:
    lines = []
    for seg in data.get("segments", []):
        words = []
        for w in seg.get("words", []) or []:
            ws = str(w.get("word", "")).strip()
            if not ws:
                continue
            words.append({
                "w": ws,
                "start": float(w.get("start", seg.get("start", 0.0)) or seg.get("start", 0.0)),
                "end": float(w.get("end", seg.get("end", 0.0)) or seg.get("end", 0.0)),
            })
        text = str(seg.get("text", "")).strip()
        if not text and words:
            text = "".join(w["w"] for w in words)
        if not text:
            continue
        s = float(seg.get("start", words[0]["start"] if words else 0.0))
        e = float(seg.get("end", words[-1]["end"] if words else s))
        lines.append({"text": text, "start": s, "end": e, "words": words})
    return lines


def load_words(data: Dict) -> List[Dict]:
    return data.get("lines", [])


def load_alignment(path: str) -> tuple:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fmt = detect_format(data)
    if fmt == "qwen":
        lines, meta = load_qwen(data), {"engine": "qwen"}
    elif fmt == "whisperx":
        lines, meta = load_whisperx(data), {"engine": "whisperx"}
    else:
        lines, meta = load_words(data), data.get("meta", {"engine": "words"})
    return lines, meta, fmt


# ---------- 歌詞文本（可選，以它為準重排） ----------

def read_lyrics(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def realign_to_lyrics(lines: List[Dict], lyrics: List[str]) -> List[Dict]:
    """
    把對齊結果依照 lyrics.txt 的行切法重新分組。
    前提是總字數對得上；對不上就退回原樣並警告。
    """
    flat: List[Dict] = []
    for ln in lines:
        ws = ln.get("words") or []
        if ws:
            flat.extend(ws)
        else:
            flat.append({"w": ln["text"], "start": ln["start"], "end": ln["end"]})

    want = sum(len(l) for l in lyrics)
    got = len(flat)
    if want != got:
        print(f"[警告] 歌詞字數 {want} 與對齊字數 {got} 不符，"
              f"改用對齊端原本的斷行。請檢查 lyrics.txt 是否漏了副歌或含標點。",
              file=sys.stderr)
        return lines

    out, i = [], 0
    for text in lyrics:
        n = len(text)
        chunk = flat[i:i + n]
        i += n
        joined = "".join(c["w"] for c in chunk)
        out.append({
            "text": text,
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "words": [dict(c, w=c["w"]) for c in chunk],
        })
        if joined != text:
            print(f"[提示] 行「{text}」對齊到「{joined}」，文字略有出入（可能是標點）。",
                  file=sys.stderr)
    return out


# ---------- 清理與驗證 ----------

def clean(lines: List[Dict], shift_ms: float, min_gap: float, duration: Optional[float]) -> List[Dict]:
    shift = shift_ms / 1000.0
    out = []
    for ln in lines:
        s = float(ln["start"]) + shift
        e = float(ln["end"]) + shift
        if s < 0:
            s = 0.0
        if e < s:
            e = s
        if duration and s > duration:
            continue
        words = []
        for w in ln.get("words") or []:
            ws = float(w["start"]) + shift
            we = float(w["end"]) + shift
            if ws < 0:
                ws = 0.0
            if we < ws:
                we = ws
            words.append({"w": w["w"], "start": ws, "end": we})
        out.append({"text": ln["text"], "start": s, "end": e, "words": words})

    # 保證嚴格遞增：時間軸倒退會讓部分播放器直接不顯示
    prev = -1.0
    fixed = 0
    for ln in out:
        if ln["start"] <= prev:
            ln["start"] = prev + min_gap
            fixed += 1
        if ln["end"] < ln["start"]:
            ln["end"] = ln["start"]
        prev = ln["start"]
        pw = -1.0
        for w in ln["words"]:
            if w["start"] < ln["start"]:
                w["start"] = ln["start"]
            if w["start"] <= pw:
                w["start"] = pw + 0.01
            if w["end"] < w["start"]:
                w["end"] = w["start"] + 0.01
            pw = w["start"]
        if ln["words"]:
            ln["end"] = max(ln["end"], ln["words"][-1]["end"])
    if fixed:
        print(f"[修正] {fixed} 行時間軸倒退，已強制遞增。", file=sys.stderr)
    return out


# ---------- 輸出 ----------

def build(lines: List[Dict], args) -> str:
    buf: List[str] = []

    if not args.netease:
        if args.ti:
            buf.append(f"[ti:{args.ti}]")
        if args.ar:
            buf.append(f"[ar:{args.ar}]")
        if args.al:
            buf.append(f"[al:{args.al}]")
        if args.by:
            buf.append(f"[by:{args.by}]")
        if args.length:
            buf.append(f"[length:{args.length}]")
    if args.offset_tag is not None:
        buf.append(f"[offset:{int(args.offset_tag)}]")

    for ln in lines:
        head = fmt_lrc(ln["start"])
        if args.enhanced and ln["words"]:
            # 卡拉OK 式：每個字的高亮延伸到下一字開始
            body = []
            for idx, w in enumerate(ln["words"]):
                nxt = ln["words"][idx + 1]["start"] if idx + 1 < len(ln["words"]) else ln["end"]
                body.append(f"{fmt_word(w['start'])}{w['w']}{fmt_word(max(nxt, w['end']))}")
            buf.append(head + "".join(body))
        else:
            buf.append(head + ln["text"])

    return "\n".join(buf) + "\n"


def main():
    p = argparse.ArgumentParser(description="字級時間戳 JSON → .lrc")
    p.add_argument("json", help="對齊結果 JSON")
    p.add_argument("-o", "--output", help="輸出 .lrc 路徑（預設與輸入同名）")
    p.add_argument("--lyrics", help="以這份歌詞文本的行切法為準重新分組")
    p.add_argument("--enhanced", action="store_true", help="輸出增強 LRC（逐字）")
    p.add_argument("--netease", action="store_true",
                   help="網易雲模式：不寫 ti/ar/al/by metadata")
    p.add_argument("--shift-ms", type=float, default=0.0,
                   help="整體平移毫秒，負數=提早顯示（烘焙進時間戳）")
    p.add_argument("--offset-tag", type=int, default=None,
                   help="只寫入 [offset:N] 標籤，不改時間戳")
    p.add_argument("--min-gap", type=float, default=0.15, help="相鄰兩行最小間隔秒數")
    p.add_argument("--bom", action="store_true", help="寫入 UTF-8 BOM（預設不寫）")
    p.add_argument("--ti", help="歌名")
    p.add_argument("--ar", help="歌手")
    p.add_argument("--al", help="專輯")
    p.add_argument("--by", help="製作者")
    p.add_argument("--length", help="長度 mm:ss")
    args = p.parse_args()

    lines, meta, fmt = load_alignment(args.json)
    if not lines:
        raise SystemExit("對齊結果是空的，沒有可輸出的歌詞行。")

    print(f"[讀入] 格式={fmt} 引擎={meta.get('engine', '?')} 共 {len(lines)} 行")

    if args.lyrics:
        lines = realign_to_lyrics(lines, read_lyrics(args.lyrics))

    lines = clean(lines, args.shift_ms, args.min_gap, meta.get("duration"))

    text = build(lines, args)

    out = args.output or os.path.splitext(args.json)[0] + ".lrc"
    enc = "utf-8-sig" if args.bom else "utf-8"
    with open(out, "w", encoding=enc, newline="\n") as f:
        f.write(text)

    print(f"[輸出] {out}")
    print(f"       編碼={enc}  行數={len(lines)}  "
          f"首行={fmt_lrc(lines[0]['start'])}  末行={fmt_lrc(lines[-1]['start'])}")

    if meta.get("duration"):
        span = lines[-1]["end"]
        print(f"       音訊長度={meta['duration']:.1f}s  歌詞結束={span:.1f}s")
        if span > meta["duration"] + 1:
            print("       [警告] 歌詞結束時間超過音訊長度。", file=sys.stderr)


if __name__ == "__main__":
    main()
