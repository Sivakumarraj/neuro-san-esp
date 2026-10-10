"""Speak the narration with Kokoro and write the timings the video is cut to.

    python make_audio.py MODEL_DIR

MODEL_DIR holds kokoro-v1.0.onnx and voices-v1.0.bin from the kokoro-onnx
releases. Writes public/audio/<scene>.mp3, public/audio/music.mp3 and
src/timing.json, which gives every scene its length and every caption line its
start and end frame.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

from narration import SCENES

FPS = 30
VOICE = "af_heart"
SPEED = 0.96
LEAD_IN = 0.7    # seconds of quiet before a scene's first line
GAP = 0.38       # between lines
TAIL = 1.0       # after the last line, while the scene settles and fades

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "public" / "audio"


def to_mp3(wav: Path, mp3: Path, bitrate: str = "128k") -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                    "-codec:a", "libmp3lame", "-b:a", bitrate, str(mp3)], check=True)
    wav.unlink()


def ambient(seconds: float, rate: int = 44100) -> np.ndarray:
    """A quiet, slowly moving pad: four detuned chords crossfading."""
    t = np.arange(int(seconds * rate)) / rate
    chords = [(110.0, 164.81, 220.0, 277.18), (98.0, 146.83, 196.0, 246.94),
              (87.31, 130.81, 174.61, 220.0), (98.0, 146.83, 196.0, 293.66)]
    position = (t / 8.0) % len(chords)   # each chord holds for eight seconds
    out = np.zeros_like(t)
    for i, chord in enumerate(chords):
        distance = np.abs(position - i)
        distance = np.minimum(distance, len(chords) - distance)
        env = np.where(distance < 1, np.cos(np.pi / 2 * distance) ** 2, 0.0)
        tone = sum(np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 1.003 * t)
                   for f in chord)
        out += env * tone
    out *= 0.5 + 0.5 * np.sin(2 * np.pi * t / 23.0) ** 2
    fade = np.minimum(1, np.minimum(t / 4, (seconds - t) / 6))
    out *= fade
    out /= np.max(np.abs(out)) + 1e-9
    return (out * 0.9).astype(np.float32)


def main(model_dir: str) -> None:
    kokoro = Kokoro(f"{model_dir}/kokoro-v1.0.onnx", f"{model_dir}/voices-v1.0.bin")
    AUDIO.mkdir(parents=True, exist_ok=True)
    timing = {"fps": FPS, "scenes": []}
    total = 0
    for scene in SCENES:
        rate = 24000
        parts = [np.zeros(int(LEAD_IN * rate), dtype=np.float32)]
        cursor = LEAD_IN
        lines = []
        for line in scene["lines"]:
            samples, rate = kokoro.create(line["say"], voice=VOICE, speed=SPEED, lang="en-us")
            length = len(samples) / rate
            lines.append({
                "text": line.get("show", line["say"]),
                "start": round(cursor * FPS),
                "end": round((cursor + length) * FPS),
            })
            parts.append(samples.astype(np.float32))
            parts.append(np.zeros(int(GAP * rate), dtype=np.float32))
            cursor += length + GAP
        parts.append(np.zeros(int((TAIL - GAP) * rate), dtype=np.float32))
        audio = np.concatenate(parts)
        wav = AUDIO / f"{scene['id']}.wav"
        sf.write(wav, audio, rate)
        to_mp3(wav, AUDIO / f"{scene['id']}.mp3")
        frames = int(np.ceil(len(audio) / rate * FPS))
        total += frames
        timing["scenes"].append({
            "id": scene["id"], "chapter": scene["chapter"], "title": scene["title"],
            "frames": frames, "lines": lines,
        })
        print(f"{scene['id']:<16} {frames / FPS:6.1f}s", flush=True)
    timing["frames"] = total
    (ROOT / "src" / "timing.json").write_text(json.dumps(timing, indent=1))

    music = AUDIO / "music.wav"
    sf.write(music, ambient(total / FPS + 2), 44100)
    to_mp3(music, AUDIO / "music.mp3", "64k")
    print(f"total {total / FPS / 60:.1f} min, {total} frames")


if __name__ == "__main__":
    main(sys.argv[1])
