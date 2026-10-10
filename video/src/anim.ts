import {Easing, interpolate} from 'remotion';

/** 0 → 1 over `len` frames from `start`, eased. */
export const ease = (frame: number, start: number, len = 20) =>
  interpolate(frame, [start, start + len], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.bezier(0.33, 0, 0.2, 1),
  });

/** Rises over `len` frames at `start`, falls over `len` frames before `end`. */
export const hold = (frame: number, start: number, end: number, len = 15) =>
  Math.min(ease(frame, start, len), 1 - ease(frame, end - len, len));

export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export const lerp3 = (
  a: [number, number, number],
  b: [number, number, number],
  t: number,
): [number, number, number] => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];

/** Springy overshoot for things that pop in. */
export const pop = (frame: number, start: number, len = 18) => {
  const t = Math.max(0, Math.min(1, (frame - start) / len));
  if (t <= 0) return 0;
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
};

/** Position of a sub-step inside a caption line, for lines that list several things. */
export const split = (start: number, end: number, parts: number, i: number) =>
  start + ((end - start) * i) / parts;
