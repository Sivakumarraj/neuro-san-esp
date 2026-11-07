# The ESP lens: a 4K film of neuro-san-esp

A film of 3 minutes 37 seconds, 3840 × 2160 at 30 fps, that walks through this repository the way an author of
ESP would review it: five questions, each answered from the repository's own files and committed logs. The
scenes are three.js, the text is HTML laid over them, and the soundtrack is synthesised, so the whole film is
rebuilt from the files in this folder.

The rendered video is not committed. Render it with the steps below.

## What it shows

| Starts | Scene | Where its figures come from |
| --- | --- | --- |
| 0:00 | Title | |
| 0:10 | The lens: five questions an ESP author would ask | [CITATION.cff](../../CITATION.cff) for the ESP paper |
| 0:21 | The gap: the designer builds one network and never measures it | [README](../../README.md), opening |
| 0:33 | 01 Context, action, outcome, and the fitness formula | [README](../../README.md#how-it-works) |
| 0:47 | 01 The test world: 24 depots, 40 contracts, 60 incidents, 124 documents | [README](../../README.md#how-it-works) |
| 0:58 | 02 The search: phases A to D and the seven operators | [DESIGN.md](../DESIGN.md), `esp/genome/mutations.py` |
| 1:18 | 02 The surrogate: one Predictor per outcome; accuracy +0.63, token cost −0.53 | [DESIGN.md](../DESIGN.md), [FINDINGS.md](../FINDINGS.md) |
| 1:33 | 03 Held out: 94.5% against 84.0%, 17 questions to 2, p = 0.0007 | [README](../../README.md#headline-result), `results/headtohead/paid-2026-10/` |
| 1:51 | 03 Replication with a model control: 90.0%, 77.5%, 75.0% | [README](../../README.md), `results/replication/fresh-2026-10/` |
| 2:04 | 04 Cost: the router, the one-line rule, agreement of two runs, $3.73 | [README](../../README.md), [RESULTS.md](../RESULTS.md) |
| 2:24 | 05 Production: the optimiser as an event-invoked service | [SERVING.md](../../SERVING.md), `compose.yaml` |
| 2:46 | 05 CI, the paid experiment's resumable legs, and serving the winner | `.github/workflows/`, `apps/web/serve.py` |
| 3:01 | Where this is not ESP | [README](../../README.md#limitations) |
| 3:15 | By the numbers | `git log`, `wc -l`, `results/history.json` |
| 3:27 | Credits | |

The counts in "By the numbers" (46 commits, 18,457 lines of Python, 784 test functions in 12,054 lines of
tests) are a snapshot of commit `7af6b94`, written into `film.html` by hand. They do not update themselves.

The questions are framed the way an ESP author would review the work. Nothing in the film is said or endorsed
by Cognizant AI Lab or by the ESP paper's authors, and the end card says so. It also says the code was written
with Claude Code.

## Render it

You need Node 18 or newer, ffmpeg, and Python 3 with NumPy for the soundtrack.

```bash
cd docs/video
npm ci
npx playwright install chromium
node render.js all 3
```

`out/neuro-san-esp-4k.mp4` (about 215 MB: H.264 at 8 Mbit/s, AAC normalised to −16 LUFS) and
`out/neuro-san-esp-1080p.mp4` (about 54 MB) are the results. Chromium draws WebGL in software (SwiftShader), so no
GPU is needed and every machine renders the same frames, slowly: about 2.2 seconds a frame for each worker. The
film's 6,510 frames took 76 minutes with three workers on four cores.

To look at single frames first:

```bash
node render.js stills 40 100 150
```

writes `out/still-40.png` and the others, at full resolution.

## How it is built

| File | What it does |
| --- | --- |
| `film.html` | The film. `window.renderAt(t)` draws the frame at `t` seconds from `t` alone, with seeded randomness, so frames can be rendered in any order and by several workers at once. |
| `render.js` | Serves the folder, drives headless Chromium, screenshots each frame and pipes it to ffmpeg; joins the parts and adds the soundtrack. |
| `music.py` | The soundtrack: a synthesised pad, bass, arpeggio and bells, with a swell and a low hit at each scene cut. Seeded, so every run writes the same file. |
| `package.json` | Pins three.js and Playwright. |
