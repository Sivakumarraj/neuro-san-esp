import React from 'react';
import {Composition} from 'remotion';
import {Explainer, SingleScene} from './Explainer';
import {TIMING} from './scene';
import {FPS, H, W} from './theme';

export const Root: React.FC = () => (
  <>
    <Composition id="Explainer" component={Explainer} durationInFrames={TIMING.frames} fps={FPS} width={W} height={H} />
    {TIMING.scenes.map((s) => (
      <Composition
        key={s.id}
        id={`Scene-${s.id.replaceAll('_', '-')}`}
        component={SingleScene}
        defaultProps={{id: s.id}}
        durationInFrames={s.frames}
        fps={FPS}
        width={W}
        height={H}
      />
    ))}
  </>
);
