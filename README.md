# song2lrc

**把「已經寫好的歌詞 ＋ 最終混音檔」變成音樂平台要的動態歌詞 `.lrc`。**

> 適用情境：歌詞已經寫好，只有最終混音檔（WAV/MP3），需要產出音樂平台要的動態歌詞 .lrc。
> 典型使用者：用 Suno / Udio 等 AIGC 工具做歌的人——歌詞是你寫的，只差時間軸。

## 實測狀態

| 路線 | 腳本 | 狀態 |
|---|---|---|
| 本機 CPU（不需 GPU／不需 torch） | `align_local.py` | ✅ 已實測，8 首英文歌全部跑完 |
| Colab GPU + WhisperX | `align_whisperx.py` | ⚠️ 替代方案，本機未驗證 |
| 本機 Qwen3-ForcedAligner | `align_qwen.py` | ⚠️ 需 torch，Intel Mac 裝不起來 |
| 本機 stable-ts | `align_stablets.py` | ⚠️ 同上，Apple Silicon 才可行 |

`json2lrc.py` / `qc.py` / `run.sh` / `batch.sh` 皆已實測。

## 目錄怎麼擺

```
lrc-workflow/
├── input/                      放進來的原始素材（音訊 + lyrics.txt）
├── output/
│   └── Draw-the-Line/          一首歌一個資料夾
│       ├── Draw-the-Line.words.json      字級時間戳（中介檔，可進 git）
│       ├── Draw-the-Line.netease.lrc     網易雲版
│       ├── Draw-the-Line.kugou.lrc       酷狗/QQ 版
│       ├── Draw-the-Line.kugou.word.lrc  酷狗/QQ 逐字版
│       ├── Draw-the-Line.lyrics.txt      當初對齊用的那版歌詞（留存，方便對照）
│       ├── Draw-the-Line.words.segs.json 原始 ASR 段落（調參時可沿用，不用重聽）
│       ├── Draw-the-Line.words.asr.txt   Whisper 實際聽到的逐字稿（查錯用）
│       └── align.log                     對齊過程紀錄
├── align_local.py              對齊（本機 CPU 路線，不需 torch）
├── json2lrc.py                 words.json → .lrc
├── qc.py                       品檢
├── probe.py                    單一區間診斷（那段到底有沒有唱？）
├── diff_asr.py                 ASR 逐字稿 vs 歌詞，分辨「歌詞版本不對」還是「漏聽」
├── run.sh                      一首歌，一鍵跑完
├── batch.sh                    整張專輯批次跑
└── setup.sh                    建立環境（只需跑一次）
```

---

## 一、核心觀念：不要用聽寫，要用強製對齊

| | 語音辨識聽寫（ASR） | 強製對齊（Forced Alignment） |
|---|---|---|
| 輸入 | 只有音訊 | 音訊 **+ 你的正確歌詞** |
| 做什麼 | 猜測唱的是哪些字 | 只算每個字在幾分幾秒發聲 |
| 歌詞會不會錯 | 會。AIGC 人聲含糊時錯得更凶 | 不會，一個字都不會多或少 |
| 適合誰 | 沒有歌詞、要反推字幕 | **你（已有歌詞）** |

你已經有歌詞，**一律走強製對齊**。這是一切的基礎。

---

## 二、三條路線，按你的硬體選

你的機器：Intel i5-1030NG7 @1.1GHz / 8GB RAM / 無 GPU / macOS。

| 路線 | 工具 | 一首 4 分鐘歌耗時 | 中文精度 | 建議 |
|---|---|---|---|---|
| **A. 雲端 GPU（推荐）** | Colab + WhisperX | 2–4 分鐘 | 高 | ⭐ 主力 |
| **B. 本機純 CPU** | Qwen3-ForcedAligner-0.6B | 8–15 分鐘 | 最高（逐字） | 備用／不想上傳 |
| **C. 全手動** | LRC Maker 網頁版 | 10 分鐘人工 | 看你手感 | 只修幾句時 |

### 路線 A：Colab + WhisperX（推荐）

1. 開一個 Colab，執行階段類型選 **T4 GPU**（免費）。
2. 上傳音樂檔（若用 Suno，直接從 Suno 下載 **人聲分軌 WAV**，可跳過 Demucs，更快更準）。
3. 貼上 `align_whisperx.py` 的內容執行，下載產出的 `words.json`。
4. 本機跑 `json2lrc.py` 轉成 .lrc。

### 路線 B：本機 CPU + Qwen3-ForcedAligner

0.6B 模型，8GB RAM 跑得動，中文逐字精度目前最好。限製：單次最多 5 分鐘音訊，超過要分段。

```bash
pip install torch torchaudio transformers soundfile
python align_qwen.py song.wav lyrics.txt -o words.json
```

### 路線 C：手動精修

AI 自動對齊大概有九成到位，剩下的一成（長拖尾、气口、rap）用手修比較快。
推荐 **https://lrc-maker.github.io/**（灯裡的歌詞滾動姬，網頁版免安裝）。

---

## 三、統一中介格式：`words.json`

不論走哪一條路線，對齊端都輸出同一份 `words.json`，轉換端只認這個格式。
好處是換引擎不用改下遊，也可以把 JSON 存進 git 做版本管理。

```json
{
  "meta": { "audio": "song.wav", "duration": 214.5, "language": "zh", "engine": "whisperx" },
  "lines": [
    {
      "text": "穿過人海茫茫",
      "start": 12.34,
      "end": 16.80,
      "words": [
        { "w": "穿", "start": 12.34, "end": 12.61 },
        { "w": "過", "start": 12.61, "end": 12.95 }
      ]
    }
  ]
}
```

`json2lrc.py` 也能直接吃 WhisperX 原始輸出和 Qwen aligner 的輸出，會自動判別格式。

---

## 四、歌詞文本準備（`lyrics.txt`）

對齊精度高度依賴「文本跟音訊唱的完全一致」。請遵守：

- 一行一句，**斷句要跟實際演唱一致**，不要按書面語斷。
- 刪掉 `[Chorus]`、`(間奏)` 這類結構標記（可以留 `#` 開頭當註釋，腳本會跳過）。
- 副歌重複幾次就寫幾次，不要寫「（副歌重複）」。
- 和聲襯詞（`啊～`、`哦～`）若不影響主唱對齊，建議先刪；要對齊就必須照實寫進去。
- 標點符號建議全部移除——有些對齊引擎會把標點當成一個發音單位。
- 存成 **UTF-8、LF 換行**。

---

## 五、輸出 .lrc

日常直接用一行解决（成品會進 `output/<歌名>/`）：

```bash
# 一首歌
./run.sh input/Draw-the-Line.mp3 input/Draw-the-Line-lyrics.txt "Draw the Line" "LMS2" en

# 整張專輯：input/ 裡還沒跑過的全部跑一輪，歌名由檔名推导
./batch.sh
```

`batch.sh` 會自動配對 `input/<歌名>_lyrics.txt`，跳過已有成品的，單首失敗不中斷，
結束時列出完成／跳過／失敗清單。全專輯丟著跑即可（一首約 17 分鐘）。

`words.json` 已存在時會**自動跳過對齊**，只重轉 .lrc——改 metadata、調平移時不用再等 17 分鐘。
要真的重跑對齊就加 `FORCE=1`。

要單獨調的話：

```bash
# 標準逐行（網易雲模式：不寫 metadata、UTF-8 無 BOM）
python3 json2lrc.py output/Draw-the-Line/Draw-the-Line.words.json -o song.lrc --netease

# 增強逐字（酷狗/QQ/AIMP 卡拉OK 高亮）
python3 json2lrc.py output/Draw-the-Line/Draw-the-Line.words.json -o song.lrc --enhanced

# 整體偏早/偏晚時平移（負數=提早顯示）
python3 json2lrc.py output/Draw-the-Line/Draw-the-Line.words.json -o song.lrc --shift-ms -180

# 帶 metadata
python3 json2lrc.py output/Draw-the-Line/Draw-the-Line.words.json -o song.lrc --ti "歌名" --ar "歌手" --al "專輯"

# 品檢
python3 qc.py output/Draw-the-Line/Draw-the-Line.words.json
```

### 平台格式地雷（很重要）

| 項目 | 要求 |
|---|---|
| 編碼 | **UTF-8 無 BOM**。帶 BOM 有播放器會整段不顯示 |
| 時間軸 | `[mm:ss.xx]`，毫秒固定**兩位** |
| 括號 | 中括號、冒號、小數點**全部英文半形**，全形直接失效 |
| 順序 | 時間軸必須嚴格遞增，不能顛倒 |
| 網易雲 | 官方明講：**不要**寫 `[ti:][ar:][al:]`，**不要**寫製作者 `[by:]`。用 `--netease` |
| QQ／酷狗 | 逐字要 QRC，得用酷狗歌詞製作專家；LRC 只吃逐行 |
| 換行 | LF 或 CRLF 都可，不要 CR（Mac OS 9 古早格式 |

---

## 六、建議的日常節奏

```
Suno 產歌 → 下載人聲分軌 WAV
           → Colab 跑對齊（2 分鐘）
           → words.json 丟進 output/<歌名>/
           → 本機 json2lrc.py 產 .lrc（成品也放同一個資料夾）
           → 丟進歌詞滾動姬聽一遍，只修跑掉的那幾句
           → 上傳平台
```

一首歌從頭到尾約 10 分鐘，其中真正要動手的大概 3 分鐘。

---

## 六之一、實測踩坑清單（Intel Mac 上跑出來的，都是真的）

這幾條是實際跑 `Draw the Line`（350s 英文歌）時踩到的，每一條都會讓你白跑一輪：

1. **PyTorch 不再支援 Intel Mac。** 2.3 起只剩 Apple Silicon 的輪子，唯一還有 x86_64 的是 torch 2.2.2（且只到 Python 3.12）。所以 `stable-ts`、`Qwen3-ForcedAligner` 在 Intel Mac 上基本走不通，改用 faster-whisper（CTranslate2，不需要 torch）。
2. **faster-whisper 1.2.x + PyAV ≥13 會直接 TypeError。** 它內部調用 `av.open(..., metadata_errors=...)`，但新版 PyAV 把這參數拿掉了。`transcribe()` 只在輸入不是 ndarray 時才走這條路，所以**自己把音訊解成 16kHz float32 陣列丟進去就能繞開**（`align_local.py` 已這樣做）。
3. **千萬別開 VAD。** Silero VAD 是語音模型，遇到有伴奏的歌會把整首判成靜音——實測保留 0 秒 / 350 秒，轉寫直接回傳 0 段。關掉之後 75 段全部命中。`align_local.py` 預設關閉，要開請加 `--vad`。
4. **別相信 `sf.info()` 的長度。** 它對這個 mp3 回報 385.3s，實際是 349.96s（afinfo 與 PyAV 數出來的樣本數都這麼說）。差 35 秒會讓尾奏整段歪掉，所以長度一律用「實際解出來的樣本數」算。
5. **PyAV 對整數 PCM 回傳的是原始 int16**（不是 [-1,1] 浮點），必須自己除以 32768 正規化，否則整段破音。
6. **bash 3.2 會吃掉 `$VAR` 後面的全形字元。** macOS 內建 bash 還是 3.2，寫 `$LANG_CODE）` 會被當成變數名的一部分 → `unbound variable`。一律寫 `${LANG_CODE}`。
7. **整首歌一次餵給 Whisper 會中途崩塌。** faster-whisper 在 `no_speech_prob > 門檻` 且 logprob 偏低時會 **fast-forward 直接跳掉整段**；唱歌的 `no_speech_prob` 常常 0.8~0.9（它的語音檢測頭是為說話訓練的），一旦長時間解碼讓 logprob 劣化，後半首就整片蒸發。實測一首 333 秒的歌只轉出 16 段、後面 160 秒全空，但把同一段音訊單獨切出來轉寫卻內容豐富。**解法：切成 60 秒一段獨立解碼**（`--chunk`），並關掉 `condition_on_previous_text`、把 `no_speech_threshold` 放寬到 0.85。
8. **faster-whisper 每次都會先連 HuggingFace 查 repo_info，即使模型早已在硬碟裡。** 在中國大陸很容易撞到代理 503。`align_local.py` 會先找本機快取、直接丟路徑給它，第二次起完全不連網。
9. **判斷對齊好壞的唯一指標是「插值行比例」。** 每次跑完會存一份 `<歌名>.asr.txt`（Whisper 實際聽到的逐字稿），對齊崩掉時開來看一眼就知道是 Whisper 沒聽到，還是歌詞文本跟實際唱的不一樣。
10. **Whisper 會幻覺出 YouTube 片尾字幕。** 它是在大量影片音軌上訓練的，遇到聽不懂的音樂不會留白，而是很自信地吐出 `"Thanks for watching."` / `"I hope you enjoyed the video."` / `"www.github.com"`。這些假段落會**搶走真歌詞的時間軸**——實測 `Wild Little Detours` 被一句幻覺佔掉 8.4~26.0s，害主歌一整段全變插值。`align_local.py` 內建片語黑名單 + 稀疏檢查（很長的段落卻只有幾個字）自動丟掉，並把那段拿去用小切片重轉。要關掉請加 `--allow-junk`。
11. **別把歌詞當 `initial_prompt` 餵進去。** 直覺上「先給它看歌詞應該更準」，實測反而是災難：遇到沒聽懂的段落，Whisper 直接把提示詞當回音吐出來（`'Youknowthewayofcourse'`、`'Morningsunanopenroad'`），污染後面的序列比對。`align_local.py` 預設**不帶提示詞**；想試請加 `--hint auto`。
12. **不要用整份歌詞對整份 ASR 做一次全域比對。** 副歌會重複好幾次，全域 LCS 一旦某一行配錯（實測 `"Feels a little right with you"` 配到第三段副歌去），後面十幾行就統統沒得配、全部掉進插值。改成**依序錨定**：照歌詞行的順序往後找，找到就把游標推進，時間軸永遠往前、也一定先用最近的那一次。
13. **光是「依序」還不夠，搜尋還要設時間上限。** 只做單向推進的話，`"Here we go again!"` 會因為跟後面 `"Here we go tonight"` 只差幾個字，一口氣往前跳 43 秒，把中間十幾行全部鎖死在門外。`--max-gap`（預設 30 秒）限制每行只能配到「上一行結束 + N 秒」之前。實測 `Wild Little Detours` 的健康度：無上限 36 → 設 30 秒 84。
14. **候選要挑「最早能用的」，不要挑「配得最完美的」。** 唱歌是有順序的，後面那次副歌配得再漂亮也不是這一行。改成「找到第一個覆蓋率達標的候選就採用」之後，`Wild Little Detours` 的插值行從 6 降到 **2**，健康度 84 → **90**。要換回舊行為請加 `--prefer-best`。
15. **很長的段落卻轉不出東西，不代表那段沒唱。** 實測 `Wild Little Detours` 主歌一在 30 秒分段裡完全靜默，但切成 4 秒一片就聽得出 `"Morning sun and open road"`。`--gap` 會自動偵測覆蓋不到的時間洞（含開頭與結尾）並用 `--retry-chunk` 的小切片補，預設開著。

### 速度參考（i5-1030NG7 @1.1GHz / 8GB / CPU int8）

| 步骤 | 350 秒的歌 |
|---|---|
| 音訊解碼 + 重取樣至 16kHz | 約 33 秒 |
| faster-whisper small 轉寫（關 VAD） | 約 16 分鐘 |

整輪無人值守約 17 分鐘。若只是要驗證流程通不通，先用 `MODEL=tiny`（模型較小、速度快很多）。

### 產完 .lrc 先過一次品檢

```bash
python3 qc.py output/Draw-the-Line/Draw-the-Line.words.json
```

會告訴你：幾行是插值補的（這個比例是精度最重要的單一指標）、有沒有時間倒退、
有沒有過短／過長的行、末句結束與音訊長度的覆蓋率，最後給一個健康度分數。
插值行 >30% 通常代表歌詞文本跟實際唱的對不上，回去修 `lyrics.txt` 比手修時間軸快。

### 對齊崩掉時：用 `diff_asr.py` 定位原因

插值率偏高有兩種完全不同的原因，修法也不一樣，先用這支腳本區分：

```bash
python3 diff_asr.py output/<歌名>/<歌名>.words.asr.txt input/<歌名>_lyrics.txt
```

它把「Whisper 實際聽到的」跟「你寫的歌詞」逐詞對齊，告訴你：

- **整體相似度**——低於 60% 幾乎可以確定是**歌詞版本不對**（Suno 沒照你寫的唱，
  或 mp3 是另一個 take）。去 Suno 後台複製它實際用的歌詞，覆蓋 `lyrics.txt` 重跑。
- **「歌詞有、AI 沒聽到」**——可能是間奏，也可能是 Whisper 漏聽。
  把那段音訊單獨切出來轉寫一次：短片段聽得到、整首卻沒有，就是長解碼崩塌，
  縮小 `--chunk`（預設 60，可試 30）再跑一次通常能救回來。
- **「對不上」的錯字**（例如 detours↔details）——那只是 Whisper 聽錯，
  不影響歌詞，因為我們永遠輸出你給的文字。

### 只想查某一段：用 `probe.py`

懷疑某個時間區間對不上時，不必重跑整首。把那段挖出來單獨轉寫，
順便印出每秒能量，就能分辨「Whisper 漏聽」跟「那段本來就沒唱」：

```bash
.venv/bin/python probe.py input/Wild-Little-Detours.mp3 4 36 --language en
.venv/bin/python probe.py input/Wild-Little-Detours.mp3 4 36 --language en --step 4
```

直接轉 32 秒通常只吐得出零碎兩三段；加 `--step 4` 切成 4 秒一片，
常常整段歌詞就浮出來了——這正是 `--gap` / `--retry-chunk` 自動在做的事。

### 調參不想重聽：`--segs-in`

轉寫是最慢的一步（一首 3.5 分鐘的歌約 10 分鐘）。每次跑完會把原始 ASR 段落
存成 `<歌名>.segs.json`，之後要調映射參數直接沿用，幾秒就有結果：

```bash
.venv/bin/python align_local.py input/歌.mp3 input/歌詞.txt \
  -o output/歌/歌.words.json --segs-in output/歌/歌.segs.json \
  --min-cover 0.3 --max-gap 20
```

參數敏感性（兩首歌交叉實測，插值行數）：

| `--min-cover` | 0.4 | 0.45 | 0.5 | 0.55 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| Wild Little Detours | **6** | 6 | 7 | 8 | 5 | 5 |
| My Own Gravity | **2** | 8 | 12 | 16 | 18 | 22 |

| `--max-gap` | 15 | 20 | 25 | **30** | 35 | 40 | 60 |
|---|---|---|---|---|---|---|---|
| Wild Little Detours | 6 | 6 | 6 | **6** | 6 | 31 | 38 |
| My Own Gravity | 2 | 2 | 2 | **2** | 2 | 2 | 17 |

兩個結論：

- `--max-gap` 有一段穩定的高原（15~35 秒），預設取中間 30；一旦放太寬，
  相似歌詞就會互相誘走，健康度直接腰斬。
- `--min-cover` **寧可低不要高**。門檻拉到 0.6 對 `Wild Little Detours` 只多救回
  1 行，對 `My Own Gravity` 卻從 2 行插值暴增到 18 行（健康度 90 → 65）。
  轉寫跟歌詞的落差本來就大，設太嚴只會整首崩掉。

---

## 七、進階：如果要做逐字卡拉OK影片

`--enhanced` 產出的增強 LRC 可以直接餵給：
- AIMP、foobar2000（裝 ESLyric）
- 剪映／Premiere 做逐字字幕（先轉 SRT）

若要轉 SRT：每行結束時間 = 下一行開始時間，最後一行 +2 秒。
