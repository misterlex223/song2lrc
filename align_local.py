#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_local.py — 備用路線：faster-whisper（CTranslate2），不需要 torch

跟 align_qwen.py 的差別：
  Qwen 是「真強制對齊」——你的歌詞就是標準答案，它只算時間。
  這支是「AI 粗對齊 + 映射」——讓 Whisper 聽一遍拿時間戳，再用序列比對
  貼回你的歌詞。**文字永遠以你給的為準**，只有時間戳來自 AI。

精度稍遜於 Qwen，但快很多，且在任何 CPU 上都能跑。

安裝：
  pip install faster-whisper soundfile

用法：
  python align_local.py input/song.mp3 input/lyrics.txt -o words.json
  python align_local.py input/vocals.wav input/lyrics.txt -o words.json --model small
"""

import argparse
import difflib
import json
import os
import re
import sys

PUNCT = re.compile(r"[\s　,.\-—…!?;:'\"“”‘’()\[\]（）【】《》~～]+")
TARGET_SR = 16000


def norm(s: str) -> str:
    return PUNCT.sub("", s).lower()


def read_lyrics(path: str):
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


def _resample_np(x, sr_in, sr_out, taps=32, phase_steps=256):
    """
    帶限 sinc 重取樣（純 numpy，不依賴 scipy / soxr / ffmpeg）。
    比線性內插好很多，Whisper 的 mel 特徵才不會糊掉。
    """
    import numpy as np

    if sr_in == sr_out or len(x) == 0:
        return np.asarray(x, dtype=np.float32)

    ratio = sr_out / sr_in
    half = taps // 2
    fc = min(0.5, ratio / 2.0)              # 截止頻率（對輸入取樣率正規化）
    n = np.arange(-half + 1, half + 1, dtype=np.int64)

    # 每個相位一組權重：h(d) = 2fc·sinc(2fc·d)·kaiser(d)，最後正規化到 DC=1
    bank = np.empty((phase_steps, taps), dtype=np.float64)
    beta = 8.6
    for q in range(phase_steps):
        frac = q / phase_steps
        d = n - frac
        win = np.i0(beta * np.sqrt(np.maximum(1.0 - (d / half) ** 2, 0.0))) / np.i0(beta)
        h = 2 * fc * np.sinc(2 * fc * d) * win
        s = h.sum()
        bank[q] = h / s if abs(s) > 1e-12 else h

    n_out = int(round(len(x) * ratio))
    xpad = np.pad(np.asarray(x, dtype=np.float64), (half + 1, half + 1))
    out = np.empty(n_out, dtype=np.float32)
    block = 1 << 16
    for start in range(0, n_out, block):
        m = np.arange(start, min(start + block, n_out), dtype=np.float64)
        p = m / ratio                       # 輸出樣本在輸入取樣率下的位置
        i0 = np.floor(p).astype(np.int64)
        q = np.clip(((p - i0) * phase_steps).astype(np.int64), 0, phase_steps - 1)
        idx = i0[:, None] + n[None, :] + half + 1
        out[start:start + len(m)] = np.einsum("bk,bk->b", bank[q], xpad[idx])
    return out


def resolve_model(name: str) -> str:
    """
    優先使用本機快取裡的模型，直接把路徑丟給 WhisperModel。
    否則 faster-whisper 每次都會先連 HuggingFace 查 repo_info，
    在中國大陸很容易卡在代理 503——明明模型早就在硬碟裡了。
    """
    if os.path.isdir(name):
        return name
    hubs = [os.environ.get("HF_HUB_CACHE"),
            os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
            os.path.expanduser("~/.cache/huggingface/hub")]
    for hub in filter(None, hubs):
        snap = os.path.join(hub, f"models--Systran--faster-whisper-{name}", "snapshots")
        if not os.path.isdir(snap):
            continue
        for rev in sorted(os.listdir(snap), reverse=True):
            p = os.path.join(snap, rev)
            if os.path.exists(os.path.join(p, "model.bin")):
                return p
    return name                    # 沒快取就走原本的下載流程


def _transcribe_one(model, clip, language, hint, nst):
    """轉寫一段音訊，回傳正規化後的 segments"""
    segments, _info = model.transcribe(
        clip,
        language=language,
        word_timestamps=True,
        initial_prompt=hint,
        vad_filter=False,             # Silero 對音樂完全失效，見 README
        condition_on_previous_text=False,   # 音樂上容易讓幻覺一路連鎖下去
        no_speech_threshold=nst,
    )
    out = []
    for s in segments:
        ws = [{"w": str(w.word).strip(),
               "start": round(float(w.start), 3),
               "end": round(float(w.end if w.end is not None else w.start), 3)}
              for w in (s.words or []) if str(w.word).strip()]
        out.append({"text": str(s.text).strip(),
                    "start": round(float(s.start), 3),
                    "end": round(float(s.end), 3),
                    "words": ws})
    return out


def transcribe_chunked(model, audio, language, hint,
                       chunk=60.0, overlap=4.0, nst=0.85):
    """
    分段轉寫。

    為什麼要切：整首歌一次餵給 Whisper，長時間解碼會讓 logprob 劣化，
    一旦 no_speech_prob 又偏高（唱歌很常見，0.8~0.9），faster-whisper
    就會「fast-forward」整段跳掉——實測一首 333 秒的歌只轉出 16 段、
    後半 160 秒全空。切成 60 秒一段、每段獨立解碼就沒有這個問題。

    重疊區用來當上下文：只保留完整落在「本段結尾 - overlap」之前的段，
    跨過界的留給下一段從頭抓，因此不會重複也不會漏。
    """
    total = len(audio) / TARGET_SR
    if total <= chunk * 1.5:
        return _transcribe_one(model, audio, language, hint, nst)

    segs, s0 = [], 0.0
    while s0 < total:
        s1 = min(s0 + chunk, total)
        clip = audio[int(s0 * TARGET_SR):int(s1 * TARGET_SR)]
        keep_until = total if s1 >= total else s1 - overlap
        got = _transcribe_one(model, clip, language, hint, nst)
        kept = 0
        for g in got:
            a, b = g["start"] + s0, g["end"] + s0
            if a >= keep_until:            # 整段落在重疊區 → 交給下一段
                continue
            g2 = dict(g)
            g2["start"] = round(a, 3)
            g2["end"] = round(min(b, keep_until), 3)
            g2["words"] = [{"w": w["w"],
                            "start": round(min(w["start"] + s0, keep_until), 3),
                            "end": round(min(w["end"] + s0, keep_until), 3)}
                           for w in g["words"]]
            segs.append(g2)
            kept += 1
        print(f"  · {s0:6.1f}–{s1:6.1f}s → 抓到 {len(got)} 段，保留 {kept} 段")
        if s1 >= total:
            break
        s0 = s1 - overlap

    segs.sort(key=lambda x: x["start"])
    return segs


def _to_float32(arr):
    """把 int16/int32/uint8 PCM 正規化到 [-1, 1]，浮點原樣轉型"""
    import numpy as np
    if arr.dtype.kind == "f":
        return arr.astype(np.float32)
    info = np.iinfo(arr.dtype)
    a = arr.astype(np.float32)
    if arr.dtype.kind == "u":               # uint8：中點是 128
        return (a - (int(info.max) + 1) / 2.0) / ((int(info.max) + 1) / 2.0)
    return a / float(-int(info.min))         # int16 → /32768，int32 → /2^31


def _decode_ffmpeg(path, target=TARGET_SR):
    """備援：用系統的 ffmpeg 解成 raw PCM"""
    import subprocess
    import numpy as np
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
           "-f", "s16le", "-ac", "1", "-ar", str(target), "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def load_audio_16k(path, target=TARGET_SR):
    """
    把任何音訊讀成 16kHz 單聲道 float32。
    刻意不走 faster-whisper 內建的 av 解碼：PyAV ≥13 拿掉了 metadata_errors
    參數，會讓 faster-whisper 1.2.x 直接 TypeError。自己餵 ndarray 就能繞開。
    """
    import numpy as np

    data, sr, last_err = None, None, None

    # 1) PyAV（內含完整 ffmpeg，對 mp3/m4a 的長度最準，自己開就不會踩
    #    faster-whisper 那個 metadata_errors 的坑）
    try:
        import av
        with av.open(str(path)) as c:
            sr = int(c.streams.audio[0].rate)
            chunks = []
            for frame in c.decode(audio=0):
                arr = frame.to_ndarray()
                arr = _to_float32(arr)                  # int16/int32 PCM 要正規化
                chunks.append(arr.mean(axis=0) if arr.ndim > 1 else arr)
            data = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    except Exception as e:                                  # noqa: BLE001
        data, last_err = None, e

    # 2) soundfile / libsndfile（wav/flac 最快最準，mp3 也讀得動）
    if data is None:
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            if data.ndim > 1:
                data = data.mean(axis=1)
        except Exception as e:                              # noqa: BLE001
            data, last_err = None, e

    # 3) 系統的 ffmpeg
    if data is None:
        try:
            return _decode_ffmpeg(path, target)
        except Exception as e:                              # noqa: BLE001
            raise SystemExit(f"無法解碼音訊 {path}：{e}（先前嘗試：{last_err}）")

    if data.ndim > 1:                       # 混成單聲道
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    if sr != target:
        data = _resample_np(data, int(sr), target)
    return np.ascontiguousarray(data, dtype=np.float32)


def build_char_map(lyric_chars: str, asr_chars: str):
    """
    把歌詞的每個字元對應到 ASR 字串的字元索引。
    回傳 (amap, strict)：strict[i] 表示該字元是否真的在 LCS 配對區段裡。
    配對區段之間的字元會補一個大致的位置，但 strict=False，
    讓上層能判斷「這行其實沒抓到」而不是硬套一個錯的時間。
    """
    amap = [None] * len(lyric_chars)
    strict = [False] * len(lyric_chars)
    sm = difflib.SequenceMatcher(None, lyric_chars, asr_chars, autojunk=False)
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            amap[i + k] = j + k
            strict[i + k] = True

    last = None
    for i, v in enumerate(amap):
        if v is None:
            amap[i] = None if last is None else min(last + 1, len(asr_chars) - 1)
        else:
            last = v
    first = next((v for v in amap if v is not None), None)
    if first is not None:
        for i, v in enumerate(amap):
            if v is None:
                amap[i] = max(first - (i + 1), 0)
            else:
                break
    return amap, strict


def attach(raw_lines, lyrics_clean, segs):
    """
    字級映射：先把所有 ASR 詞攤平成字串，跟整份歌詞做序列比對，
    再依行的邊界切回去。這樣即使 Whisper 把好幾行併成一段也能對得上。
    若某行真正配對到的字元太少（<50%），判定為「沒抓到」，改用前後插值。
    """
    words = [w for s in segs for w in s["words"]]
    asr_char_owner, asr_parts = [], []
    for idx, w in enumerate(words):
        n = norm(w["w"])
        asr_parts.append(n)
        asr_char_owner.extend([idx] * len(n))
    asr_chars = "".join(asr_parts)

    lyric_chars = "".join(lyrics_clean)
    if asr_chars:
        amap, strict = build_char_map(lyric_chars, asr_chars)
    else:
        amap, strict = [None] * len(lyric_chars), [False] * len(lyric_chars)

    out, pos = [], 0
    for raw, clean in zip(raw_lines, lyrics_clean):
        lo, hi = pos, pos + len(clean)
        pos = hi
        hit = sum(1 for k in range(lo, hi) if strict[k]) if hi > lo else 0
        ratio = hit / len(clean) if clean else 0.0

        if hi > lo and asr_chars and ratio >= 0.5 and amap[lo] is not None:
            wi_lo = asr_char_owner[amap[lo]]
            wi_hi = asr_char_owner[min(amap[hi - 1], len(asr_char_owner) - 1)]
            s = words[wi_lo]["start"]
            e = max(words[wi_hi]["end"], s + 0.2)
            out.append({"text": raw, "start": s, "end": e,
                        "words": _spread(raw, s, e, words[wi_lo:wi_hi + 1]),
                        "_note": None})
        else:
            out.append({"text": raw, "start": None, "end": None,
                        "words": [], "_note": "Whisper 沒抓到，需插值"})

    _fill_gaps(out, segs)
    return out


def _spread(line_text, s, e, span):
    """把行的時間區間按字數比例分給歌詞裡的每個詞"""
    toks = line_text.split()
    if not toks:
        return []
    # ASR 詞數剛好對上就直接沿用它的時間，最準
    if len(span) == len(toks):
        return [{"w": t, "start": span[i]["start"], "end": span[i]["end"]}
                for i, t in enumerate(toks)]
    total = sum(len(t) for t in toks) or 1
    dur = max(e - s, 0.1)
    out, cur = [], s
    for t in toks:
        w = dur * len(t) / total
        out.append({"w": t, "start": round(cur, 3), "end": round(cur + w, 3)})
        cur += w
    return out


def _fill_gaps(out, segs):
    """把沒對上的行，用前後有抓到的行插值補時間"""
    known = [i for i, m in enumerate(out) if m["start"] is not None]
    if not known:
        tail = segs[-1]["end"] if segs else 0.0
        for i, m in enumerate(out):
            m["start"] = m["end"] = float(i * 3.0)
        return
    for i, m in enumerate(out):
        if m["start"] is not None:
            continue
        prev_end = next((out[k]["end"] for k in reversed(known) if k < i), 0.0)
        nxt_start = next((out[k]["start"] for k in known if k > i), prev_end + 3.0)
        if nxt_start <= prev_end:
            nxt_start = prev_end + 3.0
        m["start"], m["end"] = prev_end, nxt_start
        m["_note"] = "插值"
    for i, m in enumerate(out):
        if m["_note"]:
            print(f"[警告] 第 {i+1} 行「{m['text'][:24]}」{m['_note']}，建議手修。",
                  file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("lyrics")
    p.add_argument("-o", "--output", default="words.json")
    p.add_argument("--model", default="small",
                   help="tiny / base / small / medium / large-v3")
    p.add_argument("--language", default="zh")
    p.add_argument("--vad", action="store_true",
                   help="開啟 VAD（對音樂極不可靠：Silero 是語音模型，"
                        "實測會把整首歌判成靜音而砍光，故預設關閉）")
    p.add_argument("--chunk", type=float, default=60.0,
                   help="分段長度（秒）。整首一次解碼容易中途崩塌，故預設 60")
    p.add_argument("--overlap", type=float, default=4.0,
                   help="分段之間的重疊秒數，純供上下文，不影響輸出")
    p.add_argument("--no-speech-threshold", type=float, default=0.85,
                   help="放寬到 0.85：唱歌常被判成「非語音」而被整段跳掉"
                        "（原始預設 0.6 太嚴）")
    args = p.parse_args()

    raw_lines, clean_lines = read_lyrics(args.lyrics)
    if not clean_lines:
        raise SystemExit("歌詞是空的。")
    print(f"[音訊] {os.path.basename(args.audio)}")
    print(f"[歌詞] {len(clean_lines)} 行 / {sum(len(c) for c in clean_lines)} 字")

    from faster_whisper import WhisperModel
    mpath = resolve_model(args.model)
    cached = os.path.isdir(mpath)
    print(f"[載入] faster-whisper '{args.model}'"
          + ("（使用本機快取，不連網）" if cached else "（首次會下載模型）"))
    model = WhisperModel(mpath, device="cpu", compute_type="int8")

    audio = load_audio_16k(args.audio)
    dur = len(audio) / TARGET_SR      # 用實際樣本數，不信 sf.info 的估計值
    print(f"[解碼] {dur:.1f}s @ {TARGET_SR}Hz（已繞開 faster-whisper 的 PyAV 呼叫）")

    hint = " ".join(clean_lines[:8])[:600]
    print(f"[轉寫] language={args.language}，每 {args.chunk:.0f}s 一段、重疊 {args.overlap:.0f}s")
    segs = transcribe_chunked(model, audio, args.language, hint,
                              chunk=args.chunk, overlap=args.overlap,
                              nst=args.no_speech_threshold)
    print(f"[轉寫] 共取得 {len(segs)} 段，覆蓋 {segs[0]['start']:.1f}s → {segs[-1]['end']:.1f}s"
          if segs else "[轉寫] 沒有取得任何段落！")

    # 存一份 ASR 逐字稿，事後要查「Whisper 到底聽到什麼」才不用重跑
    asr_path = os.path.splitext(args.output)[0] + ".asr.txt"
    try:
        with open(asr_path, "w", encoding="utf-8") as f:
            for s in segs:
                f.write(f"[{s['start']:7.2f} → {s['end']:7.2f}] {s['text']}\n")
    except OSError:
        pass

    mapped = attach(raw_lines, clean_lines, segs)

    lines = [{
        "text": m["text"],
        "start": round(m["start"], 3),
        "end": round(m["end"], 3),
        "words": [{"w": w["w"], "start": w["start"], "end": w["end"]} for w in m["words"]],
        "_note": m["_note"],
    } for m in mapped]

    payload = {
        "meta": {"audio": os.path.basename(args.audio), "language": args.language,
                 "engine": f"faster-whisper/{args.model}", "duration": round(dur, 3),
                 "interpolated": sum(1 for m in mapped if m["_note"])},
        "lines": lines,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    miss = sum(1 for m in mapped if m["_note"])
    print(f"[輸出] {args.output}（{len(lines)} 行）")
    if miss:
        print(f"[注意] {miss} 行是插值出來的，建議用歌詞編輯器手修這幾句。")


if __name__ == "__main__":
    main()
