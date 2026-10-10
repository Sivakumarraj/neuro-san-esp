import React from 'react';
import {AbsoluteFill} from 'remotion';
import {ease} from '../anim';
import {C, FONT} from '../theme';

/** Bottom caption for the line being spoken. */
export const Caption: React.FC<{text: string; opacity: number}> = ({text, opacity}) => (
  <AbsoluteFill style={{justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 54, pointerEvents: 'none'}}>
    <div
      style={{
        maxWidth: 1480,
        padding: '16px 30px',
        borderRadius: 18,
        background: 'rgba(4, 8, 22, 0.72)',
        border: '1px solid rgba(140, 170, 255, 0.18)',
        color: C.ink,
        fontFamily: FONT.body,
        fontWeight: 500,
        fontSize: 36,
        lineHeight: 1.32,
        textAlign: 'center',
        opacity,
        transform: `translateY(${(1 - opacity) * 10}px)`,
        textWrap: 'balance' as React.CSSProperties['textWrap'],
      }}
    >
      {text}
    </div>
  </AbsoluteFill>
);

/** Chapter and scene title, top-left. */
export const Header: React.FC<{chapter: string; title: string; frame: number}> = ({chapter, title, frame}) => {
  const t = ease(frame, 4, 22);
  if (!title) return null;
  return (
    <div style={{position: 'absolute', left: 72, top: 58, opacity: t, transform: `translateX(${(1 - t) * -24}px)`}}>
      {chapter ? (
        <div style={{fontFamily: FONT.body, fontWeight: 600, fontSize: 22, letterSpacing: 3, color: C.accent, textTransform: 'uppercase', marginBottom: 10}}>
          {chapter}
        </div>
      ) : null}
      <div style={{fontFamily: FONT.display, fontWeight: 700, fontSize: 50, color: C.ink, textShadow: '0 4px 30px rgba(0,0,0,0.6)'}}>{title}</div>
    </div>
  );
};

export const Brand: React.FC = () => (
  <div style={{position: 'absolute', right: 64, top: 60, fontFamily: FONT.mono, fontSize: 22, color: 'rgba(200,215,255,0.55)', letterSpacing: 1}}>
    neuro-san-esp
  </div>
);

export const Progress: React.FC<{at: number; marks: number[]}> = ({at, marks}) => (
  <div style={{position: 'absolute', left: 0, right: 0, top: 0, height: 6, background: 'rgba(120,150,230,0.12)'}}>
    <div style={{width: `${at * 100}%`, height: '100%', background: `linear-gradient(90deg, ${C.specialist}, ${C.coordinator})`}} />
    {marks.map((m) => (
      <div key={m} style={{position: 'absolute', left: `${m * 100}%`, top: 0, width: 3, height: 6, background: C.bg}} />
    ))}
  </div>
);

/** A floating glass card for numbers and short statements. */
export const Card: React.FC<{
  x: number;
  y: number;
  w?: number;
  t: number;
  accent?: string;
  children: React.ReactNode;
  center?: boolean;
  inline?: boolean;
}> = ({x, y, w = 520, t, accent = C.accent, children, center, inline}) => {
  if (t <= 0.001) return inline ? <div style={{width: w}} /> : null;
  return (
    <div
      style={{
        position: inline ? 'relative' : 'absolute',
        left: inline ? undefined : x,
        top: inline ? undefined : y,
        width: w,
        padding: '24px 30px',
        borderRadius: 22,
        background: 'linear-gradient(160deg, rgba(20,32,70,0.82), rgba(8,14,36,0.82))',
        border: `1.5px solid ${accent}55`,
        boxShadow: `0 20px 60px rgba(0,0,0,0.45), inset 0 0 40px ${accent}14`,
        color: C.ink,
        fontFamily: FONT.body,
        opacity: t,
        transform: `translateY(${(1 - t) * 26}px) scale(${0.96 + 0.04 * t})`,
        textAlign: center ? 'center' : 'left',
      }}
    >
      {children}
    </div>
  );
};

export const Big: React.FC<{children: React.ReactNode; color?: string; size?: number}> = ({children, color = C.ink, size = 64}) => (
  <div style={{fontFamily: FONT.display, fontWeight: 700, fontSize: size, color, lineHeight: 1.05}}>{children}</div>
);

export const Small: React.FC<{children: React.ReactNode; color?: string; size?: number}> = ({children, color = C.dim, size = 26}) => (
  <div style={{fontFamily: FONT.body, fontWeight: 500, fontSize: size, color, marginTop: 8, lineHeight: 1.3}}>{children}</div>
);

export const Mono: React.FC<{children: React.ReactNode; color?: string; size?: number}> = ({children, color = C.ink, size = 28}) => (
  <span style={{fontFamily: FONT.mono, fontSize: size, color}}>{children}</span>
);

/** A centred headline over the 3D scene. */
export const Headline: React.FC<{t: number; y?: number; children: React.ReactNode}> = ({t, y = 0, children}) =>
  t <= 0.001 ? null : (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', pointerEvents: 'none'}}>
      <div style={{opacity: t, transform: `translateY(${y + (1 - t) * 20}px) scale(${0.94 + 0.06 * t})`, textAlign: 'center'}}>{children}</div>
    </AbsoluteFill>
  );

export const Chip: React.FC<{text: string; color: string; t?: number; size?: number}> = ({text, color, t = 1, size = 28}) => (
  <span
    style={{
      display: 'inline-block',
      padding: '8px 18px',
      margin: 6,
      borderRadius: 999,
      background: `${color}22`,
      border: `1.5px solid ${color}aa`,
      color,
      fontFamily: FONT.mono,
      fontSize: size,
      fontWeight: 600,
      opacity: t,
      transform: `scale(${0.85 + 0.15 * t})`,
    }}
  >
    {text}
  </span>
);
