"""The film's soundtrack, synthesised: no samples, no downloads, the same file every run.

    python3 music.py OUT.wav [SECONDS] [CUT,CUT,...]

A slow pad over a four-chord loop (Dm9, Bbmaj7, Fadd9, C6), a sub bass, a plucked
arpeggio that thickens while the results are on screen, a bell every other chord,
and a soft swell and low hit at each scene cut. render.js passes the film's length
and its scene starts, read from film.html, so the cuts land on the picture.
"""
import sys
import wave

import numpy as np

SR = 48000
FILM_SECONDS = 217.0
FILM_CUTS = [10, 21, 33, 47, 58, 78, 93, 111, 124, 144, 166, 181, 195, 207]


def hz(midi):
    return 440 * 2 ** ((midi - 69) / 12)


def envelope(n, attack, release):
    e = np.ones(n)
    a, r = int(attack * SR), int(release * SR)
    if a:
        e[:a] = np.linspace(0, 1, a) ** 1.5
    if r:
        e[-r:] *= np.linspace(1, 0, r) ** 1.5
    return e


def energy(t):
    """How busy the music is at second t: quiet titles, fullest over the results."""
    xs, ys = zip(*[(0, .35), (10, .5), (33, .6), (58, .75), (93, .95), (124, .9),
                   (144, .85), (181, .6), (195, .75), (207, .55), (217, .3)], strict=True)
    return float(np.interp(t, xs, ys))


def compose(seconds, cuts):
    n_total = int(SR * seconds)
    left, right = np.zeros(n_total), np.zeros(n_total)
    rng = np.random.default_rng(20261107)

    def add(sig, t0, pan=0.0, gain=1.0):
        i = int(t0 * SR)
        j = min(n_total, i + len(sig))
        if j <= i:
            return
        s = sig[:j - i] * gain
        left[i:j] += s * np.cos((pan + 1) * np.pi / 4)
        right[i:j] += s * np.sin((pan + 1) * np.pi / 4)

    chords = [[50, 53, 57, 60, 64], [46, 50, 53, 57, 60],
              [41, 48, 53, 57, 67], [48, 52, 55, 57, 62]]
    roots = [38, 34, 41, 36]
    chord_seconds = 6.857          # eight beats at 70 bpm
    beat = chord_seconds / 8
    t, k = 0.0, 0
    while t < seconds:
        chord, e = chords[k % 4], energy(t)
        n = int((chord_seconds + 2.5) * SR)
        tt = np.arange(n) / SR
        pad_env = envelope(n, 1.6, 2.6)
        for midi in chord:
            for detune, pan in ((-0.07, -.5), (0.0, 0.0), (0.07, .5)):
                f = hz(midi + detune)
                sig = np.zeros(n)
                for h in range(1, 7):
                    sig += np.sin(2 * np.pi * f * h * tt + rng.random() * 6.28) / h ** 1.6
                add(sig * pad_env, t, pan, 0.018 * (0.7 + 0.5 * e))
        f = hz(roots[k % 4])
        sub = np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(2 * np.pi * 2 * f * tt)
        add(sub * envelope(n, 1.0, 2.4), t, 0, 0.06 * (0.6 + 0.6 * e))
        if t >= 9:
            tones = [*sorted(chord), chord[1] + 12, chord[2] + 12]
            pattern = [0, 2, 4, 6, 5, 3, 1, 4]
            for b in range(16):
                tb = t + b * beat / 2
                if tb >= seconds - 3:
                    break
                if e < .5 and b % 2:
                    continue
                f = hz(tones[pattern[b % 8] % len(tones)] + 12)
                x = np.arange(int(1.1 * SR)) / SR
                tone = (np.sin(2 * np.pi * f * x) + 0.3 * np.sin(4 * np.pi * f * x)
                        + 0.08 * np.sin(6 * np.pi * f * x))
                pluck = tone * np.exp(-x * 5.5) * np.minimum(1, x * 400)
                add(pluck, tb, -0.6 if b % 2 else 0.6, 0.05 * e)
        if k % 2 == 0 and t > 20:
            f = hz(chord[-1] + 24)
            x = np.arange(3 * SR) / SR
            bell = (np.sin(2 * np.pi * f * x) + 0.4 * np.sin(2 * np.pi * 2.76 * f * x)) \
                * np.exp(-x * 1.6) * np.minimum(1, x * 200)
            add(bell, t + 0.02, rng.uniform(-.7, .7), 0.016)
        t += chord_seconds
        k += 1

    for cut in cuts:
        x = np.arange(int(1.2 * SR)) / SR
        for pan in (-0.3, 0.3):
            noise = np.convolve(rng.standard_normal(len(x)), np.ones(24) / 24, "same")
            add(noise * (x / 1.2) ** 2.5, cut - 1.2, pan, 0.05)
        x = np.arange(int(1.6 * SR)) / SR
        hit = np.sin(2 * np.pi * (55 * x - 9 * x * x)) * np.exp(-x * 3.2) * np.minimum(1, x * 300)
        add(hit, cut, 0, 0.16)

    # Reverb: a synthetic stereo impulse response, convolved by FFT.
    ir_n = int(2.8 * SR)
    x = np.arange(ir_n) / SR

    def impulse():
        v = np.convolve(rng.standard_normal(ir_n) * np.exp(-x * 2.4), np.ones(6) / 6, "same")
        v[:int(.012 * SR)] = 0
        return v / np.sqrt((v ** 2).sum())

    size = 1 << int(np.ceil(np.log2(n_total + ir_n)))

    def convolve(sig, h):
        return np.fft.irfft(np.fft.rfft(sig, size) * np.fft.rfft(h, size), size)[:n_total]

    wet_l, wet_r = convolve(left, impulse()), convolve(right, impulse())
    stereo = np.stack([0.72 * left + 0.42 * wet_l, 0.72 * right + 0.42 * wet_r], 1)
    fade = np.ones(n_total)
    fade[:int(2.5 * SR)] = np.linspace(0, 1, int(2.5 * SR))
    fade[-int(5 * SR):] = np.linspace(1, 0, int(5 * SR)) ** 1.3
    stereo *= fade[:, None]
    stereo = np.tanh(stereo * 1.2) / np.tanh(1.2)
    return stereo / (np.abs(stereo).max() / 0.89)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "music.wav"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else FILM_SECONDS
    cuts = [float(c) for c in sys.argv[3].split(",")] if len(sys.argv) > 3 else FILM_CUTS
    pcm = (compose(seconds, cuts) * 32767).astype("<i2")
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print(f"wrote {out}: {seconds:.0f} s, {len(cuts)} cuts")


if __name__ == "__main__":
    main()
