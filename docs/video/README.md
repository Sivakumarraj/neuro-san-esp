# The ESP lens: a 10-minute film of neuro-san-esp

A narrated film of 9 minutes 58 seconds, rendered at 7680 × 4320 (8K) or 3840 × 2160, 30 fps, that
walks through this repository the way an author of ESP would review it: five questions, 36 scenes,
every figure taken from the repository and its committed logs. The scenes are three.js, the words on
screen are HTML laid over them, the narration is a synthetic voice made on the machine that renders,
and the music is synthesised, so the whole film is rebuilt from the files in this folder.

The rendered video is not committed. Render it with the steps below.

## What it shows

| Starts | Chapter | Scenes | Where the figures come from |
| --- | --- | --- | --- |
| 0:00 | Opening | title, the lens, the gap, which ESP this is | [CITATION.cff](../../CITATION.cff), [README](../../README.md), [FINDINGS](../FINDINGS.md#prior-art) |
| 1:01 | 01 Context, action, outcome | the objective, the genome, the test world, the question sets | [README](../../README.md#how-it-works), [DESIGN](../DESIGN.md), [SECURITY](../../SECURITY.md) |
| 2:13 | 02 The surrogate | the search loop, the seven operators, the Predictor, the per-outcome split, the Pareto front | [DESIGN](../DESIGN.md), [FINDINGS](../FINDINGS.md#what-the-predictor-is-exactly) |
| 3:35 | 03 Held out | what did not hold up, the held-out bank, the paid stages, the search cap, where the model goes, the head-to-head, the replication | [FINDINGS](../FINDINGS.md#the-paid-runs), [RESULTS](../RESULTS.md), [README](../../README.md#headline-result) |
| 5:40 | 04 Cost | dollars against tokens, routing as ESP with a context, agreement of two runs, the error autopsy | [README](../../README.md), [RESULTS](../RESULTS.md), [FINDINGS](../FINDINGS.md#inside-a-run-the-penalty-questions) |
| 6:48 | 05 Production | the defects found by running it, the audit, the service, providers, CI, serving | [FINDINGS](../FINDINGS.md#what-measurement-changed), [SERVING](../../SERVING.md), [GUIDE](../GUIDE.md), `.github/workflows/` |
| 8:31 | Close | the commit timeline, where this is not ESP, prior art, what comes next, the numbers, credits | `git log`, [README](../../README.md#limitations), [FINDINGS](../FINDINGS.md#prior-art) |

The counts in the closing numbers (46 commits, 18,457 lines of Python, 784 test functions in 12,054
lines of tests) and the commit timeline are a snapshot of commit `7af6b94`, written in by hand. They do
not update themselves.

The questions are framed the way an ESP author would review the work. Nothing in the film is said or
endorsed by Cognizant AI Lab or by the ESP paper's authors, and the end card says so. It also says the
code was written with Claude Code, and that the narrator is a synthetic voice.

## Render it

You need Node 18 or newer, ffmpeg, and Python 3 with NumPy. The narration also needs Kokoro.

```bash
cd docs/video
npm ci
npx playwright install chromium
```

**1. The narration and the timeline.** Every scene lasts as long as its spoken lines need, so the
narration comes first. Download `kokoro-v1.0.onnx` and `voices-v1.0.bin` from the
[kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases) (model-files-v1.0)
into `models/`, then:

```bash
pip install kokoro-onnx soundfile
python3 narration.py out/narration.wav timeline.js
```

It writes the voice track and `timeline.js`, which says when each scene starts and when each line
begins. `timeline.js` is committed, so this step can be skipped unless the words change; the voice
track is not, so it is needed once before assembling.

**2. The picture, and the assembled film.**

```bash
node render.js all                                    # 3840x2160
node render.js all --scale 2 --gl llvmpipe --workers 2   # 7680x4320
```

`out/neuro-san-esp-4k.mp4` or `out/neuro-san-esp-8k.mp4` is the result, with a 1080p copy beside it.
Chromium draws WebGL in software, so no GPU is needed and every machine renders the same frames.
`--gl llvmpipe` uses Mesa's renderer (`apt install libegl-mesa0 libegl1`), several times faster than
the bundled SwiftShader at 8K. Frames are written in 5-second chunks and encoded by one encoder, so an
interrupted run resumes where it stopped and memory stays within about 10 GB at 8K with two workers.
At 8K each frame costs about 1.3 seconds of a four-core machine, so the film's 17,952 frames take
six to eight hours there.

To look at single frames first:

```bash
node render.js stills 100 300 --gl llvmpipe
```

## How it is built

| File | What it does |
| --- | --- |
| `film.html` | The words on screen for all 36 scenes, the progress bar and the clock. |
| `film.js` | The 3D scenes. `window.renderAt(t)` draws the frame at `t` seconds from `t` alone, with seeded randomness, so frames can be drawn in any order and by several workers at once. |
| `timeline.js` | When each scene starts and when each spoken line begins; written by `narration.py`. |
| `narration.py` | The narration, line by line, synthesised with Kokoro (voice `af_heart`), and the timeline cut to it. |
| `music.py` | The music: a synthesised pad, bass, arpeggio and bells, with a hit at each chapter. Seeded, so every run writes the same file. |
| `render.js` | Serves the folder, drives headless Chromium, writes frames, encodes chunks, and mixes the music under the voice. |
| `package.json` | Pins three.js and Playwright. |
