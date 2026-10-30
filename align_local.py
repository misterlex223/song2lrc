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
import time

PUNCT = re.compile(r"[\s　,.\-—…!?;:'\"“”‘’()\[\]（）【】《》~～]+")
TARGET_SR = 16000

# ---------------------------------------------------------------------------
# Whisper 幻覺黑名單
#
# Whisper 是在大量 YouTube 字幕 / 影片音軌上訓練的。餵它一段沒聽懂的音樂，
# 它不會回傳空白，而是很自信地吐出訓練語料裡最常見的句子：
# "Thanks for watching." / "I hope you enjoyed the video." / "www.xxx.com"。
# 這些假段落會搶走真實歌詞的時間軸——實測 Wild Little Detours 就被
# 一句 "Thanks for watching." 佔掉 8.4~26.0s，害主歌一整段全變插值。
#
# 兩道關卡：
#   1) 片語黑名單（下面這條 regex）
#   2) 稀疏檢查：很長的段落卻只有幾個字，正常唱歌不會這樣
# 命中就整段丟掉，並把那段时间拿去用更小的切片重轉一次，嘗試撈回真歌詞。
# ---------------------------------------------------------------------------
HALLUCINATION = re.compile(
    r"thanks?(?:\s*you)?\s*(?:for\s*)?(?:watching|listening"
    r"|your\s*(?:time|support))"
    r"|(?:谢谢|感謝|感谢)\s*(?:观看|收看|收聽|聆听)"
    r"|i\s*hope\s*you\s*(?:enjoyed|enjoy|liked|like)"
    r"|(?:see|talk\s*to)\s*you\s*(?:in|on|next|soon|later)"
    r"|please\s*(?:subscribe|like|share|comment)"
    r"|(?:like\s*and\s*)?subscribe(?:\s*to\s*(?:my|our|the)\s*channel)?"
    r"|don'?t\s*forget\s*to\s*(?:subscribe|like)"
    r"|www\s*\.|https?\s*://|\.(?:com|net|org)\b"
    r"|subtitles?\s*(?:by|provided)"
    r"|(?:字幕|字幕\s*由|字幕\s*提供|字幕菌|字幕组|字幕組)"
    r"|(?:请不吝点赞|订阅|轉發|打赏|明けまして)"
    r"|(?:translated|transcription)\s*by"
    r"|amara\s*\.?\s*org"
    r"|music\s*(?:by|from)\s*copyright"
    r"|all\s*rights\s*reserved"
    , re.I)


def is_hallucination(seg):
    """回傳 None（正常）或一個說明字串（該丟掉）"""
    t = (seg.get("text") or "").strip()
    if not t:
        return "空白"
    if HALLUCINATION.search(t):
        return "片語黑名單"
    # 注意：不能用「文字出現在 initial_prompt 裡」來判斷回音。
    # initial_prompt 就是歌詞本身，凡是轉寫正確的句子都必然是它的子字串，
    # 一開就連開場對白 "Ready? / You know the way?" 都被誤殺（健康度反而下降）。
    # 回音只能靠片語黑名單（例如 "Youknowthewayofcourse.com"）去擋。
    dur = float(seg["end"]) - float(seg["start"])
    n = len(norm(t))
    if dur >= 12.0 and n / dur < 1.0:
        return f"稀疏（{dur:.1f}s 只有 {n} 字）"
    return None


def norm(s: str) -> str:
    return PUNCT.sub("", s).lower()


def read_lyrics_sec(path: str):
    """
    讀歌詞，額外回傳每一行屬於第幾個「段」。

    段 = 空行分隔的區塊（主歌／副歌／橋段）。這是歌詞檔本来就有的結構，
    不用另外偵測，而且是粗對位現成的錨點——見 coarse_bounds()。
    """
    raw_lines, clean_lines, sec_ids = [], [], []
    k, prev_blank = -1, True
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                prev_blank = True
                continue
            if s.startswith("#"):
                continue
            if prev_blank:
                k += 1
                prev_blank = False
            c = PUNCT.sub("", s)
            if not c:
                continue
            raw_lines.append(s)
            clean_lines.append(c)
            sec_ids.append(k)
    return raw_lines, clean_lines, sec_ids


def read_lyrics(path: str):
    raw, clean, _ = read_lyrics_sec(path)
    return raw, clean


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
        initial_prompt=(hint or None),
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


def _shift(seg, off, cap=None):
    """把段落（含字級）時間軸平移 off 秒，並可裁到 cap 為止"""
    g = dict(seg)
    g["start"] = round(seg["start"] + off, 3)
    g["end"] = round(seg["end"] + off, 3)
    if cap is not None:
        g["start"] = min(g["start"], cap)
        g["end"] = min(g["end"], cap)
    g["words"] = [{"w": w["w"],
                   "start": round(w["start"] + off, 3),
                   "end": round(w["end"] + off, 3)}
                  for w in seg["words"]]
    if cap is not None:
        g["words"] = [w for w in g["words"] if w["start"] < cap]
        for w in g["words"]:
            w["end"] = min(w["end"], cap)
    return g


def _merge_no_overlap(base, extra):
    """合併兩批段落，重疊時保留詞數較多（資訊量較大）的那一段"""
    out = []
    for s in sorted(base + extra, key=lambda x: (x["start"], x["end"])):
        if out and s["start"] < out[-1]["end"] - 0.25:
            if len(s["words"]) > len(out[-1]["words"]):
                out[-1] = s
            continue
        out.append(s)
    return out


def _gap_ranges(segs, min_gap, total=None):
    """
    找出沒有被任何段落覆蓋到的時間洞：段落之間、第一段之前、最後一段之後。
    這比「整段都沒輸出」更能抓到真正的漏聽：實測 Wild Little Detours
    的主歌一就是被夾在 6.2s 與 34.3s 兩個段落之間、長達 28 秒的洞，
    整段 Whisper 靜默，但切成 4 秒一片就聽得出 "Morning sun and open road"。
    """
    if min_gap <= 0 or not segs:
        return []
    out = []
    if total is not None and segs[0]["start"] >= min_gap:
        out.append([0.0, segs[0]["start"]])
    reach = segs[0]["end"]
    for s in segs[1:]:
        if s["start"] - reach >= min_gap:
            out.append([reach, s["start"]])
        reach = max(reach, s["end"])
    if total is not None and total - reach >= min_gap:
        out.append([reach, total])
    return out


def recover_range(model, audio, a, b, language, hint, nst, sub=8.0):
    """
    幻覺通常長在「長而含糊」的音訊上。切成 8 秒的小片各自獨立解碼，
    解碼狀態短、不容易再觸發同一句幻覺，常常就能把真歌詞撈回來。
    """
    total = len(audio) / TARGET_SR
    b = min(b, total)
    out, s = [], a
    while s < b - 0.5:
        e = min(s + sub, b)
        clip = audio[int(s * TARGET_SR):int(e * TARGET_SR)]
        if len(clip) < int(TARGET_SR * 0.5):
            break
        for g in _transcribe_one(model, clip, language, hint, nst):
            if is_hallucination(g):
                continue
            out.append(_shift(g, s, cap=b))
        s = e
    return out


def transcribe_chunked(model, audio, language, hint,
                       chunk=60.0, overlap=4.0, nst=0.85,
                       retry=8.0, gap=8.0, filter_junk=True):
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
        got = _transcribe_one(model, audio, language, hint, nst)
        if not filter_junk:
            return got
        good = [g for g in got if not is_hallucination(g)]
        junk = [(g["start"], g["end"]) for g in got if is_hallucination(g)]
        return _retry_ranges(model, audio, good, junk, language, hint,
                             nst, retry, gap, total)

    segs, s0, junk = [], 0.0, []
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
            if filter_junk and (reason := is_hallucination(g)):
                # 記下絕對時間，稍後用小切片重試
                junk.append((a, min(b, keep_until)))
                print(f"  · 丟棄幻覺 {a:6.1f}–{b:6.1f}s（{reason}）：{g['text'][:40]!r}")
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
    return _retry_ranges(model, audio, segs, junk, language, hint,
                         nst, retry, gap, total)


def _retry_ranges(model, audio, segs, junk, language, hint,
                  nst, retry, gap, total):
    """
    兩類可疑區間都用小切片重轉一次：
      1) 命中幻覺被丟掉的段落
      2) 段落之間的覆蓋空檔（>= gap 秒）——真正的漏聽最常躲在這裡
    """
    if retry <= 0:
        return segs
    todo = [(a, min(b, total)) for a, b in junk if b > a]
    todo += [(a, b) for a, b in _gap_ranges(segs, gap, total) if b > a]
    if not todo:
        return segs

    todo.sort()
    merged = []
    for a, b in todo:                       # 相鄰/重疊的合併成一個區間
        if merged and a <= merged[-1][1] + 1.0:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    for a, b in merged:
        print(f"  · 重試 {a:6.1f}–{b:6.1f}s（每 {retry:.0f}s 一片）")
        got = recover_range(model, audio, a, b, language, hint, nst,
                            sub=retry)
        print(f"      撈回 {len(got)} 段"
              + (f"：{got[0]['text'][:40]!r}" if got else ""))
        segs = _merge_no_overlap(segs, got)
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


def _word_char_index(asr_parts):
    """每個 ASR 詞在 asr_chars 裡的起始位置"""
    idx, cur = [], 0
    for p in asr_parts:
        idx.append(cur)
        cur += len(p)
    return idx


def _match_line(clean, asr_chars, wpos, words, p_word, deadline,
                lookahead, n_words, min_cover=0.0, earliest=False):
    """
    在 ASR 串裡、從第 p_word 個詞開始往後，找這行歌詞最可能的位置。
    回傳 (覆蓋率, asr_chars 起, asr_chars 訖) 或 None。

    兩個限制都很關鍵：
      · 只回傳「最長連續共同區塊」的覆蓋率。若用全域 LCS 的相似度，
        副歌重複三次時很容易把某一行配到很後面的那一次去。
      · 搜尋不能超過 deadline（上一個對上的行 + max-gap 秒）。否則像
        「Here we go again!」這種跟後面「Here we go tonight」只差幾個字的行，
        會一口氣往前跳 43 秒，把中間十幾行全部鎖死在門外。
    """
    best, need = None, len(clean)
    for k in range(p_word, min(p_word + lookahead, n_words)):
        if words[k]["start"] > deadline:
            break
        a0 = wpos[k]
        cand = asr_chars[a0:a0 + need * 2 + 12]
        if not cand:
            break
        sm = difflib.SequenceMatcher(None, clean, cand, autojunk=False)
        m = sm.find_longest_match(0, need, 0, len(cand))
        cover = m.size / need
        if cover < min_cover:
            continue
        if earliest:                    # 夠好就用，不貪心找更後面更完美的
            return (cover, a0 + m.b, a0 + m.b + m.size)
        if best is None or cover > best[0]:
            best = (cover, a0 + m.b, a0 + m.b + m.size)
        if cover >= 0.999:              # 完美命中，不必再往後找
            break
    return best


NEG = -1e9


def sec_score(sec_toks, asr_toks, j, win_mul=2.0, pad=8):
    """
    段放在 ASR 第 j 個詞開始位置的匹配分數（詞級 F1，失敗則退化字元級）。

    用詞級而不是純字元級：段短的時候，字元級光靠字母重合就能湊到 30~40%，
    完全沒有鑑別力（實測一堆段因此全擠在同一個位置）。
    用 F1 而不是 recall：視窗比段長，recall 會被「視窗裡剛好有這幾個詞」
    灌水，F1 同時看精確度。
    詞級完全對不上時退化到字元級：Whisper 常把 "Lets go" 併成一個詞
    "letsgo"，詞級就一個也對不上（實測 Wild Little Detours 段 13 因此
    整段無解，連帶整個 DP 走不通）。
    """
    n = len(sec_toks)
    win = asr_toks[j:j + int(n * win_mul) + pad]
    if len(win) < n * 0.6:
        return NEG
    sm = difflib.SequenceMatcher(None, sec_toks, win, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    if matched > 1:
        return 2.0 * matched / (n + len(win))
    sc, sw = "".join(sec_toks), "".join(win)
    if not sc or not sw:
        return NEG
    m2 = sum(b.size for b in difflib.SequenceMatcher(
        None, sc, sw, autojunk=False).get_matching_blocks())
    if not m2:
        return NEG
    return 0.8 * 2.0 * m2 / (len(sc) + len(sw))   # 打折：字元級證據較弱


def coarse_anchors(sec_toks_list, asr_toks, win_mul=2.0, pad=8,
                   gap_ratio=None, slack=1.0):
    """
    粗對位：全域 DP 把 ASR 詞序列切成 N 段，第 i 段放第 i 個歌詞段。

    為什麼粗對位要跟行級用不同演算法：
      行級 = 逐行貪婪推進（earliest + 時間上限），便宜，但一行配錯後面
             全部鎖死，只能用 --max-gap 這種補丁去擋，本質上是在猜。
      段級 = 全域 DP 求單調最優分割，貴（O(段數 × 詞數)），但不會級聯，
             而且段長、獨特，重複副歌誘不走。

    段把時間窗定下來之後，行級只在窗內跑，級聯就被關進籠子裡——
    最壞情況是一個段內錯，不會擴散到全曲。

    回傳 list：每個段在 ASR 裡的起始詞 index；無解回傳 []。
    """
    n_sec, W = len(sec_toks_list), len(asr_toks)
    if not n_sec or not W:
        return []
    # slack=1.0 最準（實測偏移中位 3~4 詞），但那剛好是把 ASR 詞數用滿，
    # 遇到 ASR 比歌詞短的情況就塞不下。無解時逐步放寬重試。
    for s in (slack, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
        got = _coarse_dp(sec_toks_list, asr_toks, win_mul, pad, gap_ratio, s)
        if got:
            return got
    return []


def _coarse_dp(sec_toks_list, asr_toks, win_mul, pad, gap_ratio, slack):
    n_sec, W = len(sec_toks_list), len(asr_toks)
    score = []
    for toks in sec_toks_list:
        row = [sec_score(toks, asr_toks, j, win_mul, pad) for j in range(W)]
        if max(row) <= NEG / 2:
            # 整段都找不到證據（例如 ASR 完全沒聽到這段）：設成 0 分，
            # 讓它不貢獻也不阻斷 DP，位置交給前後段的單調約束去決定。
            row = [0.0] * W
        score.append(row)

    # 段至少要佔這麼多詞，才不會兩段擠在一起。
    #
    # 這個下界非常關鍵，而且不能拍腦袋設成常數：Whisper 吐的詞通常比歌詞
    # 多（會加字、重複、把 "Lets go" 併成 "letsgo" 或拆成兩個），所以
    # 「歌詞 N 個詞」在 ASR 裡實際佔的位置是 N × (ASR總詞數/歌詞總詞數)。
    # 實測用固定的 0.6 倍會讓每段都被往前擠，誤差沿段累積到第 10 段已經
    # 偏了 38 個詞（約 20 秒）。slack 保留一點彈性給間奏和漏聽。
    total = sum(len(t) for t in sec_toks_list) or 1
    ratio = gap_ratio if gap_ratio else (W / total)
    est = [max(int(len(t) * ratio * slack), 1) for t in sec_toks_list]
    dp = [[NEG] * W for _ in range(n_sec)]
    back = [[-1] * W for _ in range(n_sec)]
    dp[0] = score[0][:]                      # 第一段可以從任何地方開始
    for i in range(1, n_sec):
        # best[j] = max(dp[i-1][j'] for j' <= j - est[i-1])，用 prefix max
        best, bestj, run, runj = NEG, -1, NEG, -1
        for j in range(W):
            src = j - est[i - 1]
            if src >= 0 and dp[i - 1][src] > run:
                run, runj = dp[i - 1][src], src
            if run > best:
                best, bestj = run, runj
            if score[i][j] > NEG / 2:
                dp[i][j] = best + score[i][j]
                back[i][j] = bestj

    end = max(range(W), key=lambda j: dp[n_sec - 1][j])
    if dp[n_sec - 1][end] <= NEG / 2:
        return []
    out = [0] * n_sec
    j = end
    for i in range(n_sec - 1, 0, -1):
        out[i] = j
        j = back[i][j]
        if j < 0:
            return []
    out[0] = j
    return out


def coarse_bounds(sec_ids, raw_lines, asr_toks, margin=3, min_block=0,
                  verbose=False, words=None, **kw):
    """
    粗對位的產出：每一行的搜尋詞範圍 [(lo, hi), ...]。

    margin 是給塊邊界的寬容（詞數）——DP 的塊長度是估計的，
    切得太死會把正確的配對擋在門外。

    min_block>0 時會先把連續的小段併成「塊」再定位。這通常比較好：
    歌詞的空行常常切得太細（很多段只有兩行），短段的錨點不穩，
    窗也因此窄到會誤傷；併成幾十個詞的大塊之後，錨點更準、窗更寬，
    卻一樣能擋住跨塊的級聯。
    """
    n_sec = max(sec_ids) + 1 if sec_ids else 0
    if n_sec <= 1:
        return None
    sec_toks = [[] for _ in range(n_sec)]
    for i, s in enumerate(sec_ids):
        for t in raw_lines[i].split():
            if any(ch.isalnum() for ch in t):
                sec_toks[s].append(norm(t))

    if min_block > 0:                       # 併段：記錄塊 → 包含哪些段
        groups, cur = [], []
        for si in range(n_sec):
            cur.append(si)
            if sum(len(sec_toks[x]) for x in cur) >= min_block:
                groups.append(cur)
                cur = []
        if cur:
            if groups and len(sec_toks) and \
                    sum(len(sec_toks[x]) for x in cur) < min_block // 2:
                groups[-1].extend(cur)      # 剩下的太少就併進上一塊
            else:
                groups.append(cur)
        blk_toks = []
        for g in groups:
            t = []
            for si in g:
                t.extend(sec_toks[si])
            blk_toks.append(t)
        sec_of_blk = {si: bi for bi, g in enumerate(groups) for si in g}
    else:
        blk_toks = sec_toks
        sec_of_blk = {si: si for si in range(n_sec)}

    anchors = coarse_anchors(blk_toks, asr_toks, **kw)
    if len(anchors) != len(blk_toks):
        return None
    if verbose:
        def tt(i):
            return f"{words[i]['start']:6.1f}s" if words and i < len(words) \
                else f"詞{i}"
        print("       塊錨點：" + "  ".join(
            f"{tt(a)}" for a in anchors[:12])
            + ("  …" if len(anchors) > 12 else ""))
    W = len(asr_toks)
    bounds = []
    for si in sec_ids:
        bi = sec_of_blk[si]
        lo = max(anchors[bi] - margin, 0)
        hi = min((anchors[bi + 1] if bi + 1 < len(anchors) else W) + margin, W)
        bounds.append((lo, max(hi, lo + 1)))
    return bounds


def attach(raw_lines, lyrics_clean, segs, min_cover=0.4, lookahead=120,
           max_gap=30.0, earliest=True, bounds=None):
    """
    依序錨定映射：把 ASR 詞攤平成一個字串，然後**依照歌詞行的順序**
    往後找每一行最可能的位置，找到就把游標推進到該處之後。

    為什麼不用整份歌詞對整份 ASR 做一次全域比對：副歌會重複好幾次，
    全域 LCS 一旦某一行配錯（實測 "Feels a little right with you" 配到
    第三段副歌），後面十幾行就統統沒有得配，全部掉進插值。
    強制單向推進 + 時間上限可以保證時間軸永遠往前、也一定先用最近的一次。
    """
    words = [w for s in segs for w in s["words"]]
    asr_parts = [norm(w["w"]) for w in words]
    asr_char_owner = [i for i, p in enumerate(asr_parts) for _ in p]
    asr_chars = "".join(asr_parts)
    wpos = _word_char_index(asr_parts)
    n_words = len(words)

    out, p, anchor = [], 0, 0.0
    for i, (raw, clean) in enumerate(zip(raw_lines, lyrics_clean)):
        hit = None
        if clean and asr_chars:
            # 粗對位給的窗：游標要「夾」在窗內，不是「至少」在窗內。
            #
            # 這裡的細節決定成敗。兩種失敗的寫法都實測過：
            #   · 純軟約束（窗內找不到就退回全文）→ 完全沒作用。24 組參數
            #     測試裡插值行數一模一樣，因為級聯發生時游標 p 早就跑出窗
            #     外了，退回全文只是讓級聯繼續。
            #   · 硬下界（p0 = max(p, lo)）→ 健康度 90 → 52。只要有一行
            #     配到窗外，p 就推進過頭，後面每行的下界都被抬高，全首崩。
            # 正確做法是把 p 夾回窗內：級聯因此被限制在單一段裡，
            # 進到下一段時游標會被拉回正確的範圍。窗內真的找不到才退回全文。
            if bounds:
                lo, hi = bounds[i]
                p0 = min(max(p, lo), max(hi - 1, lo))
                best = _match_line(clean, asr_chars, wpos, words, p0,
                                   anchor + max_gap, lookahead, hi,
                                   min_cover=min_cover, earliest=earliest)
                if best and best[0] >= min_cover:
                    hit = best
            if not hit:
                best = _match_line(clean, asr_chars, wpos, words, p,
                                   anchor + max_gap, lookahead, n_words,
                                   min_cover=min_cover, earliest=earliest)
                if best and best[0] >= min_cover:
                    hit = best
        if hit:
            _, lo, hi = hit
            wi_lo = asr_char_owner[lo]
            wi_hi = asr_char_owner[hi - 1]
            s = words[wi_lo]["start"]
            e = max(words[wi_hi]["end"], s + 0.2)
            out.append({"text": raw, "start": s, "end": e,
                        "words": _spread(raw, s, e, words[wi_lo:wi_hi + 1]),
                        "_note": None})
            p = wi_hi + 1
            anchor = max(anchor, e)
        else:
            out.append({"text": raw, "start": None, "end": None,
                        "words": [], "_note": "Whisper 沒抓到，需插值"})

    _fill_gaps(out, segs)
    return out


def _spread(line_text, s, e, span, fine=True):
    """
    細對位：把行的時間分給歌詞裡的每個詞（逐字版 LRC 用）。

    兩條路：
      · 詞級比對（fine=True）：拿歌詞的詞跟 ASR 的詞做序列比對，命中的
        直接用 Whisper 給的真實時間戳，只有沒命中的才用前後鄰居補。
        實測 74~88% 的詞能拿到真實時間戳，相較於純比例分配，詞級誤差
        90 分位 0.11~0.67s、最大 1.16s。
      · 字數比例（fallback）：詞級完全對不上時（例如整行只有一個 ASR 詞
        涵蓋）退回原本的按比例切。

    注意這層只影響逐字版的詞時間戳，行級的 start/end 不受影響——
    實測行級誤差已經是 0.00s，細對位在行級一點用都沒有。
    """
    toks = line_text.split()
    if not toks:
        return []
    # ASR 詞數剛好對上就直接沿用它的時間，最準
    if len(span) == len(toks):
        return [{"w": t, "start": span[i]["start"], "end": span[i]["end"]}
                for i, t in enumerate(toks)]

    got = {}
    if fine and len(span) >= 2:
        nt = [norm(t) for t in toks]
        ns = [norm(w["w"]) for w in span]
        sm = difflib.SequenceMatcher(None, nt, ns, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    got[i1 + k] = (span[j1 + k]["start"], span[j1 + k]["end"])

    if not got:                             # 詞級沒對上 → 退回字數比例
        total = sum(len(t) for t in toks) or 1
        dur = max(e - s, 0.1)
        out, cur = [], s
        for t in toks:
            w = dur * len(t) / total
            out.append({"w": t, "start": round(cur, 3),
                        "end": round(cur + w, 3)})
            cur += w
        return out

    out = []
    for i, t in enumerate(toks):
        if i in got:
            a, b = got[i]
        else:                               # 沒命中的詞：夾在鄰居之間
            prev = next((o for o in reversed(out[:i])
                         if o["start"] is not None), None)
            a = prev["end"] if prev else s
            nxt = next((got[j][0] for j in range(i + 1, len(toks))
                        if j in got), None)
            b = nxt if nxt else e
            if b <= a:
                b = a + 0.05
        out.append({"w": t, "start": round(a, 3), "end": round(b, 3)})

    # 保證單調遞增，否則逐字版會出現時間倒流
    for i in range(1, len(out)):
        if out[i]["start"] < out[i - 1]["end"]:
            out[i]["start"] = out[i - 1]["end"]
        if out[i]["end"] <= out[i]["start"]:
            out[i]["end"] = out[i]["start"] + 0.05
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
    p.add_argument("--retry-chunk", type=float, default=8.0,
                   help="命中幻覺的時間段，改用這個長度（秒）的小切片重轉；"
                        "傳 0 可關閉重試")
    p.add_argument("--gap", type=float, default=8.0,
                   help="段落之間超過這個秒數的空檔，就用小切片重轉補洞；"
                        "0 = 不補")
    p.add_argument("--hint", default="",
                   help="initial_prompt。預設留空：實測把歌詞當提示詞餵進去，"
                        "Whisper 遇到聽不懂的音樂會直接把提示詞當回音吐出來"
                        "（'Youknowthewayofcourse'、'Morningsunanopenroad'），"
                        "污染序列比對。要自動帶入歌詞前幾行請傳 auto")
    p.add_argument("--allow-junk", action="store_true",
                   help="關閉幻覺過濾（除錯用：想看 Whisper 到底吐了什麼）")
    p.add_argument("--segs-in", default="",
                   help="讀取已存好的 <歌名>.segs.json 直接做映射，不重跑轉寫")
    p.add_argument("--segs-out", default="",
                   help="原始 ASR 段落存檔路徑（預設 <輸出>.segs.json）")
    p.add_argument("--min-cover", type=float, default=0.4,
                   help="一行歌詞要有多少比例的字在 ASR 裡連續命中才算對上。"
                        "0.4 實測最穩：門檻拉高會讓轉寫跟歌詞落差大的整首崩掉")
    p.add_argument("--prefer-best", action="store_true",
                   help="改用「覆蓋率最高的候選」而非預設的「最早的可接受候選」。"
                        "後者比較符合唱歌的順序性，實測插值行更少")
    p.add_argument("--lookahead", type=int, default=120,
                   help="每一行最多往後找幾個 ASR 詞（防止配到很遠的重複副歌）")
    p.add_argument("--max-gap", type=float, default=30.0,
                   help="每一行最多只能配到「上一行結束 + 這個秒數」之前。"
                        "太大會被相似歌詞誘走，太小會穿不過長間奏")
    p.add_argument("--coarse", default="off", choices=["off", "section"],
                   help="粗對位：先用「段」做全域 DP 錨定，再讓行級在窗內跑。"
                        "段長又獨特，抗錯能力遠高於單行，而且能級聯錯誤關在"
                        "單一段內。off = 只用行級（舊行為）")
    p.add_argument("--coarse-margin", type=int, default=3,
                   help="段邊界的前後寬容（詞數）。DP 的段長度是估計的，"
                        "切太死會把正確配對擋在門外")
    p.add_argument("--coarse-block", type=int, default=0,
                   help="粗對位的粒度：先把連續的小段併成至少這麼多詞的"
                        "「塊」再定位（0 = 不併，直接用歌詞的空行分段）。"
                        "併大塊通常比較穩，見 coarse_bounds() 的說明")
    args = p.parse_args()

    raw_lines, clean_lines, sec_ids = read_lyrics_sec(args.lyrics)
    if not clean_lines:
        raise SystemExit("歌詞是空的。")
    n_sec = (max(sec_ids) + 1) if sec_ids else 0
    print(f"[音訊] {os.path.basename(args.audio)}")
    print(f"[歌詞] {len(clean_lines)} 行 / {sum(len(c) for c in clean_lines)} 字"
          f" / {n_sec} 段（空行分隔）")

    segs_path = args.segs_out or os.path.splitext(args.output)[0] + ".segs.json"

    if args.segs_in:
        # 已經轉寫過，直接沿用：調 attach 的參數時不必再花十幾分鐘重聽一次
        with open(args.segs_in, "r", encoding="utf-8") as f:
            saved = json.load(f)
        segs = saved["segments"]
        dur = saved.get("duration") or (segs[-1]["end"] if segs else 0.0)
        print(f"[沿用] {args.segs_in}（{len(segs)} 段），不做轉寫")
    else:
        from faster_whisper import WhisperModel
        mpath = resolve_model(args.model)
        cached = os.path.isdir(mpath)
        print(f"[載入] faster-whisper '{args.model}'"
              + ("（使用本機快取，不連網）" if cached else "（首次會下載模型）"))
        model = WhisperModel(mpath, device="cpu", compute_type="int8")

        audio = load_audio_16k(args.audio)
        dur = len(audio) / TARGET_SR      # 用實際樣本數，不信 sf.info 的估計值
        print(f"[解碼] {dur:.1f}s @ {TARGET_SR}Hz（已繞開 faster-whisper 的 PyAV 呼叫）")

        hint = (" ".join(clean_lines[:8])[:600] if args.hint == "auto"
                else args.hint)
        print(f"[轉寫] language={args.language}，每 {args.chunk:.0f}s 一段、"
              f"重疊 {args.overlap:.0f}s")
        segs = transcribe_chunked(model, audio, args.language, hint,
                                  chunk=args.chunk, overlap=args.overlap,
                                  nst=args.no_speech_threshold,
                                  retry=args.retry_chunk, gap=args.gap,
                                  filter_junk=not args.allow_junk)
        print(f"[轉寫] 共取得 {len(segs)} 段，"
              f"覆蓋 {segs[0]['start']:.1f}s → {segs[-1]['end']:.1f}s"
              if segs else "[轉寫] 沒有取得任何段落！")

        try:                            # 存原始段落，方便離線重跑 attach
            with open(segs_path, "w", encoding="utf-8") as f:
                json.dump({"audio": os.path.basename(args.audio),
                           "duration": round(dur, 3), "segments": segs},
                          f, ensure_ascii=False)
        except OSError:
            pass

    # 存一份 ASR 逐字稿，事後要查「Whisper 到底聽到什麼」才不用重跑
    asr_path = os.path.splitext(args.output)[0] + ".asr.txt"
    try:
        with open(asr_path, "w", encoding="utf-8") as f:
            for s in segs:
                f.write(f"[{s['start']:7.2f} → {s['end']:7.2f}] {s['text']}\n")
    except OSError:
        pass

    bounds = None
    if args.coarse == "section":
        asr_toks = [norm(w["w"]) for s in segs for w in (s.get("words") or [])]
        t0 = time.time()
        bounds = coarse_bounds(sec_ids, raw_lines, asr_toks,
                               margin=args.coarse_margin,
                               min_block=args.coarse_block, verbose=True,
                               words=[w for s in segs for w in (s.get("words") or [])])
        if bounds:
            print(f"[粗對位] {n_sec} 段全域 DP 完成（{time.time() - t0:.1f}s），"
                  f"行級搜尋已限制在各段的時間窗內")
        else:
            print(f"[粗對位] 失敗（段數不足或 ASR 太短），退回純行級")

    mapped = attach(raw_lines, clean_lines, segs,
                    min_cover=args.min_cover, lookahead=args.lookahead,
                    max_gap=args.max_gap, earliest=not args.prefer_best,
                    bounds=bounds)

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
