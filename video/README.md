# neuro-san-esp, explained (video)

A 12-minute 3D explainer of this project for a viewer with no background, made with
[Remotion](https://www.remotion.dev/) and `@remotion/three`. Twenty scenes, narrated,
captioned, every figure taken from the README and `docs/DESIGN.md`.

| Path | What it is |
| --- | --- |
| `tools/narration.py` | The script: what is said, and what the caption shows |
| `tools/make_audio.py` | Speaks it with Kokoro, writes `public/audio/` and `src/timing.json` |
| `src/scenes/` | The twenty scenes, cut to the narration's timings |
| `src/three/` | Shared 3D pieces: agents, links, documents, labels, the stage |
| `tools/render_all.sh` | Renders each scene, joins them, adds the music bed |

## Render

```bash
npm install
npx remotion studio src/index.ts      # preview in the browser
tools/render_all.sh                   # out/neuro-san-esp-explained.mp4
```

Without a GPU, renders use software WebGL (`--gl=swangle`); about 3 frames a second on
four cores, so the whole video takes roughly two hours. `tools/render_all.sh` skips
scenes already rendered, so a stopped run resumes. `node tools/stills.mjs s06_hops:500`
renders single frames for checking.

## Changing the words

Edit `tools/narration.py`, then regenerate the voice and timings (Python 3.11+):

```bash
pip install kokoro-onnx soundfile
# kokoro-v1.0.onnx and voices-v1.0.bin from the kokoro-onnx GitHub releases
python tools/make_audio.py /path/to/model-dir
```

Scenes are timed from `src/timing.json`, so they follow the new narration.
