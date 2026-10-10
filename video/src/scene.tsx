import React, {useContext} from 'react';
import {useCurrentFrame} from 'remotion';
import timing from './timing.json';

export type Line = {text: string; start: number; end: number};
export type SceneTiming = {id: string; chapter: string; title: string; frames: number; lines: Line[]};

export const TIMING = timing as {fps: number; frames: number; scenes: SceneTiming[]};

export const SceneCtx = React.createContext<SceneTiming | null>(null);

/** The local frame, the scene's length, and when each narrated line starts and ends. */
export const useScene = () => {
  const s = useContext(SceneCtx);
  if (!s) throw new Error('useScene outside a scene');
  const frame = useCurrentFrame();
  return {
    frame,
    dur: s.frames,
    cue: (i: number) => s.lines[i]?.start ?? s.frames,
    end: (i: number) => s.lines[i]?.end ?? s.frames,
  };
};
