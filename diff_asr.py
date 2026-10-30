#!/usr/bin/env python3
"""
比對「Whisper 實際聽到的」(.asr.txt) 跟「你寫的歌詞」(lyrics.txt)。
找出彼此對不上的地方，判斷是 Whisper 沒聽到，還是歌詞文本寫錯了。

用法：
  python3 diff_asr.py output/<歌名>/<歌名>.words.asr.txt input/<歌名>_lyrics.txt
"""
import argparse
import difflib
import re
import sys

PUNCT = re.compile(r"[\s　,.\-—…!?;:'\"“”‘’()\[\]（）【】《》~～]+")


def norm(s: str) -> str:
    """正規化：小寫、去標點空白。英文歌以「詞」為單位比對。"""
    return PUNCT.sub(" ", s).strip().lower()


def tokens(s: str):
    return [t for t in norm(s).split(" ") if t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("asr")
    ap.add_argument("lyrics")
    ap.add_argument("--context", type=int, default=3,
                    help="每個差異區塊前後要顯示幾行歌詞")
    args = ap.parse_args()

    asr_text = open(args.asr, encoding="utf-8").read()
    lyr_text = open(args.lyrics, encoding="utf-8").read()

    # asr.txt 每行是「[ 起 → 訖] 內容」，先把時間戳記前綴剝掉，
    # 否則數字會被當成詞彙污染比對
    TS = re.compile(r"^\s*\[[^\]]*\]\s*")
    asr_lines = [TS.sub("", l) for l in asr_text.splitlines()]
    asr_tok = tokens(" ".join(asr_lines))
    lyr_lines = [l.strip() for l in lyr_text.splitlines()]
    lyr_lines = [l for l in lyr_lines if l and not l.startswith("#")]
    lyr_tok = []
    owner = []                      # 每個詞屬於第幾行歌詞
    for i, line in enumerate(lyr_lines):
        for t in tokens(line):
            lyr_tok.append(t)
            owner.append(i)

    print(f"ASR 聽到 {len(asr_tok)} 個詞")
    print(f"你寫的歌詞 {len(lyr_lines)} 行 / {len(lyr_tok)} 個詞")
    print(f"詞數差 {len(lyr_tok) - len(asr_tok):+d}"
          f"（正=歌詞比 AI 聽到的多）")

    sm = difflib.SequenceMatcher(None, lyr_tok, asr_tok, autojunk=False)
    ratio = sm.ratio()
    print(f"\n整體相似度 {ratio:.1%}")

    # 收集差異區塊
    blocks = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        blocks.append((tag, i1, i2, j1, j2))

    if not blocks:
        print("\n✓ 完全一致，沒有差異。")
        return

    print(f"\n差異區塊 {len(blocks)} 處：\n")
    for tag, i1, i2, j1, j2 in blocks:
        lyr_seg = " ".join(lyr_tok[i1:i2])
        asr_seg = " ".join(asr_tok[j1:j2])
        label = {"replace": "對不上", "delete": "歌詞有、AI 沒聽到",
                 "insert": "AI 多聽到（歌詞沒寫）"}[tag]
        print(f"── {label} ──")
        print(f"   你寫的   : {lyr_seg or '（無）'}")
        print(f"   AI 聽到  : {asr_seg or '（無）'}")
        if i1 < len(owner):
            ln = owner[i1]
            lo = max(0, ln - args.context)
            hi = min(len(lyr_lines), ln + args.context + 1)
            print(f"   歌詞第 {ln+1} 行附近：")
            for k in range(lo, hi):
                mark = " >>" if k == ln else "   "
                print(f"   {mark} {k+1:3d}. {lyr_lines[k]}")
        print()

    # 統計：哪些歌詞行完全沒被 ASR 覆蓋
    covered = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                covered.add(owner[k])
    missing = [i for i in range(len(lyr_lines)) if i not in covered]
    if missing:
        print(f"⚠ 有 {len(missing)} 行歌詞在 ASR 裡完全找不到：")
        for i in missing[:30]:
            print(f"   第 {i+1:3d} 行：{lyr_lines[i]}")
        if len(missing) > 30:
            print(f"   …還有 {len(missing)-30} 行")


if __name__ == "__main__":
    main()
