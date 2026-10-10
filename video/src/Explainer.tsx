import React, {useEffect, useState} from 'react';
import {AbsoluteFill, Html5Audio, Sequence, continueRender, delayRender, interpolate, staticFile, useCurrentFrame} from 'remotion';
import {fontsReady} from './fonts';
import {Brand, Caption, Header, Progress} from './overlay/ui';
import {SceneCtx, SceneTiming, TIMING} from './scene';
import {SCENES} from './scenes';
import {C} from './theme';

const FADE = 14;

/** Everything one scene needs around its picture: voice, captions, header, fades. */
export const SceneShell: React.FC<{s: SceneTiming}> = ({s}) => {
  const frame = useCurrentFrame();
  const Picture = SCENES[s.id];
  const current = [...s.lines].reverse().find((l) => frame >= l.start - 4);
  const capOpacity = current
    ? Math.min(interpolate(frame, [current.start - 4, current.start + 4], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
        interpolate(frame, [s.frames - FADE - 6, s.frames - FADE + 2], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}))
    : 0;
  const black = Math.max(
    interpolate(frame, [0, FADE], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
    interpolate(frame, [s.frames - FADE, s.frames], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}),
  );
  return (
    <SceneCtx.Provider value={s}>
      <AbsoluteFill style={{background: C.bg}}>
        {Picture ? <Picture /> : null}
        <Header chapter={s.chapter} title={s.id === 's01_hook' ? '' : s.title} frame={frame} />
        <Brand />
        {current ? <Caption text={current.text} opacity={capOpacity} /> : null}
        <AbsoluteFill style={{background: '#000', opacity: black}} />
        <Html5Audio src={staticFile(`audio/${s.id}.mp3`)} />
      </AbsoluteFill>
    </SceneCtx.Provider>
  );
};

const useFonts = () => {
  const [ready, setReady] = useState(false);
  const [handle] = useState(() => delayRender('fonts'));
  useEffect(() => {
    fontsReady().then(() => {
      setReady(true);
      continueRender(handle);
    });
  }, [handle]);
  return ready;
};

export const Explainer: React.FC = () => {
  const ready = useFonts();
  const frame = useCurrentFrame();
  if (!ready) return null;
  let from = 0;
  const starts = TIMING.scenes.map((s) => {
    const at = from;
    from += s.frames;
    return at;
  });
  const total = TIMING.frames;
  const musicVolume = (f: number) =>
    0.075 * interpolate(f, [0, 60, total - 150, total], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  return (
    <AbsoluteFill style={{background: C.bg}}>
      <Html5Audio src={staticFile('audio/music.mp3')} volume={musicVolume} />
      {TIMING.scenes.map((s, i) => (
        <Sequence key={s.id} from={starts[i]} durationInFrames={s.frames} name={s.id}>
          <SceneShell s={s} />
        </Sequence>
      ))}
      <Progress at={frame / total} marks={starts.slice(1).map((x) => x / total)} />
    </AbsoluteFill>
  );
};

/** One scene on its own, for previews and test renders. */
export const SingleScene: React.FC<{id: string}> = ({id}) => {
  const ready = useFonts();
  const s = TIMING.scenes.find((x) => x.id === id)!;
  if (!ready) return null;
  return <SceneShell s={s} />;
};
