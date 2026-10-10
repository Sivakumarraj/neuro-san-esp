import React from 'react';
import {AbsoluteFill, random} from 'remotion';
import {ease, hold, lerp, lerp3, pop} from '../anim';
import {useScene} from '../scene';
import {Agent, Glow, Label, Link, Pulse, V3} from '../three/primitives';
import {Network} from '../three/network';
import {orbit, Stage} from '../three/Stage';
import {Big, Card, Chip, Headline, Mono, Small} from '../overlay/ui';
import {C, FONT} from '../theme';
import {designerEdges, designerNodes} from './intro';

const Coin: React.FC<{pos: V3; rot?: number}> = ({pos, rot = 0}) => (
  <mesh position={pos} rotation={[0, rot, 0]}>
    <cylinderGeometry args={[0.32, 0.32, 0.07, 32]} />
    <meshStandardMaterial color="#ffcf4a" emissive="#b8860b" emissiveIntensity={0.5} metalness={0.7} roughness={0.25} />
  </mesh>
);

const PHASES = [
  {key: 'A', name: 'Measure', text: 'a few starting teams, for real', color: C.coordinator, paid: true},
  {key: 'B', name: 'Predict', text: 'train a Predictor on those results', color: C.tool, paid: false},
  {key: 'C', name: 'Breed & rank', text: 'thousands of candidates · 0 AI calls', color: C.specialist, paid: false},
  {key: 'D', name: 'Pay for the best', text: 'measure only the elite, feed back to B', color: C.ok, paid: true},
];

export const S10Esp: React.FC = () => {
  const {frame, cue} = useScene();
  const coins = hold(frame, cue(0), cue(1) + 20, 15);
  const ring = ease(frame, cue(2) - 30, 40) * (1 - ease(frame, cue(6), 30));
  const phase = [cue(2), cue(3), cue(4), cue(5)].filter((t) => frame >= t - 5).length - 1;
  const car = ease(frame, cue(6) + 10, 30);
  const R = 4.2;
  const station = (i: number): V3 => {
    const a = Math.PI / 2 - (i * Math.PI) / 2;
    return [Math.cos(a) * R * 1.25, 0, -Math.sin(a) * R];
  };
  const camA = orbit(0.15, 12, 1.5, [0, 0.5, 0]);
  const camB = orbit(0.2 + Math.sin(frame * 0.004) * 0.15, 14.5, 8.5, [0, -0.6, 0]);
  const m = ease(frame, cue(1) + 20, 50);
  const cam = car > 0.5 ? orbit(0.9 + frame * 0.003, 11, 2.5, [0, 0, 0]) : {pos: lerp3(camA.pos, camB.pos, m), target: lerp3(camA.target!, camB.target!, m)};
  const orbT = phase < 0 ? 0 : phase + ease(frame, [cue(2), cue(3), cue(4), cue(5)][phase] - 5, 25) - 1;
  const orbPos = (() => {
    const a = Math.PI / 2 - (Math.max(0, orbT) * Math.PI) / 2;
    return [Math.cos(a) * R * 1.25, 0.25, -Math.sin(a) * R] as V3;
  })();
  return (
    <AbsoluteFill>
      <Stage cam={cam} floor={car > 0.5 ? -1.2 : -2.5}>
        {coins > 0.001 ? (
          <group>
            <group position={[-2.6, 0.8, 0]} scale={[0.55 * coins, 0.55 * coins, 0.55 * coins]}>
              <Network nodes={designerNodes(false)} edges={designerEdges} frame={frame} />
            </group>
            {Array.from({length: 16}).map((_, i) => {
              const born = cue(0) + 12 + i * 11;
              const t = Math.max(0, Math.min(1, (frame - born) / 18));
              if (frame < born) return null;
              const col = i % 4;
              const restY = -1.5 + Math.floor(i / 4) * 0.08;
              const pos: V3 = [1.6 + col * 0.75, lerp(2.5, restY, t * t), 0.4];
              return <Coin key={i} pos={pos} rot={i} />;
            })}
          </group>
        ) : null}
        {ring > 0.001 ? (
          <group scale={[ring, ring, ring]} position={[-1.8, 0, 0]}>
            <mesh rotation={[Math.PI / 2, 0, 0]} scale={[1.25, 1, 1]}>
              <torusGeometry args={[R, 0.06, 16, 160]} />
              <meshStandardMaterial color="#5b7fd6" emissive="#3a5bb0" emissiveIntensity={0.7} />
            </mesh>
            {PHASES.map((p, i) => {
              const on = i === phase;
              const pos = station(i);
              return (
                <group key={p.key} position={pos}>
                  <mesh>
                    <cylinderGeometry args={[0.9, 1.0, 0.22, 48]} />
                    <meshStandardMaterial color="#14214a" emissive={p.color} emissiveIntensity={on ? 0.8 : 0.15} />
                  </mesh>
                  <Glow pos={[0, 0.3, 0]} color={p.color} size={on ? 4.5 : 2} opacity={on ? 0.7 : 0.25} />
                  <Label text={`${p.key} · ${p.name}`} pos={[0, 1.55, 0]} height={on ? 0.62 : 0.46} opacity={phase >= 0 ? (on ? 1 : 0.55) : 0.8} style={{color: p.color, weight: 700, font: 'Space Grotesk'}} />
                  {p.paid ? <Label text="$ paid" pos={[0, -0.6, 0.9]} height={0.28} opacity={0.8} style={{color: '#ffd76a'}} /> : <Label text="free" pos={[0, -0.6, 0.9]} height={0.28} opacity={0.8} style={{color: '#9ae6b4'}} />}
                </group>
              );
            })}
            {/* A: three seed teams */}
            {[-0.7, 0, 0.7].map((x, i) => (
              <Agent key={i} pos={[station(0)[0] + x, 0.55, station(0)[2]]} r={0.2} color={i === 0 ? C.coordinator : C.specialist} s={ease(frame, cue(2) + i * 8, 14)} frame={frame} />
            ))}
            {/* B: the Predictor */}
            {phase >= 1 ? (
              <group position={[station(1)[0], 0.8, station(1)[2]]} rotation={[frame * 0.02, frame * 0.03, 0]} scale={[pop(frame, cue(3), 20), pop(frame, cue(3), 20), pop(frame, cue(3), 20)]}>
                <mesh>
                  <octahedronGeometry args={[0.55, 0]} />
                  <meshStandardMaterial color={C.tool} emissive={C.tool} emissiveIntensity={0.7} transparent opacity={0.9} />
                </mesh>
              </group>
            ) : null}
            {/* C: a swarm of candidates ranked for free */}
            {phase >= 2
              ? Array.from({length: 160}).map((_, i) => {
                  const st = station(2);
                  const a = random(`sw-a-${i}`) * Math.PI * 2 + frame * 0.01 * (0.5 + random(`sw-s-${i}`));
                  const r = 0.4 + random(`sw-r-${i}`) * 1.6 * ease(frame, cue(4), 40);
                  const y = 0.3 + random(`sw-y-${i}`) * 1.4;
                  const top = i < 6 && frame > cue(5) - 30;
                  return (
                    <mesh key={i} position={[st[0] + Math.cos(a) * r, y, st[2] + Math.sin(a) * r]}>
                      <sphereGeometry args={[top ? 0.09 : 0.05, 8, 6]} />
                      <meshBasicMaterial color={top ? C.ok : C.specialist} />
                    </mesh>
                  );
                })
              : null}
            {/* D: the elite travel to be paid for */}
            {phase >= 3
              ? Array.from({length: 4}).map((_, i) => {
                  const t = ease(frame, cue(5) + 10 + i * 10, 40);
                  const p = lerp3([station(2)[0], 1.2, station(2)[2]], [station(3)[0] + (i - 1.5) * 0.35, 0.5, station(3)[2]], t);
                  return <Agent key={i} pos={p} r={0.15} color={C.ok} frame={frame} />;
                })
              : null}
            <Pulse from={orbPos} to={orbPos} t={0.5} color="#ffffff" size={phase >= 0 ? 0.16 : 0.001} />
          </group>
        ) : null}
        {car > 0.001 ? (
          <group scale={[car, car, car]}>
            <mesh scale={[2.4, 0.7, 1.0]}>
              <sphereGeometry args={[1, 48, 32]} />
              <meshStandardMaterial color="#9fb8ff" emissive="#3a5bb0" emissiveIntensity={0.4} metalness={0.5} roughness={0.2} />
            </mesh>
            {Array.from({length: 9}).map((_, i) => {
              const y = -0.9 + i * 0.22;
              const x = ((frame * 0.12 + i * 1.3) % 10) - 5;
              const bend = Math.max(0, 1 - Math.abs(x) / 2.6) * (y >= 0 ? 0.5 : -0.5);
              return (
                <mesh key={i} position={[x, y + bend, 1.2 - (i % 3) * 0.6]}>
                  <boxGeometry args={[1.4, 0.03, 0.03]} />
                  <meshBasicMaterial color={C.accent} transparent opacity={0.7} />
                </mesh>
              );
            })}
          </group>
        ) : null}
      </Stage>
      <Card x={1240} y={230} w={560} t={coins} accent="#ffcf4a">
        <Small color="#ffcf4a" size={22}>EVERY REAL EXAM</Small>
        <div style={{fontSize: 32, fontWeight: 600, marginTop: 6}}>Each answer calls a real AI model.</div>
        <Small size={26}>Money and time, every single question.</Small>
      </Card>
      <Headline t={hold(frame, cue(1) + 5, cue(2) - 5, 16)} y={-320}>
        <Big size={88} color={C.accent}>ESP</Big>
        <Small size={32} color={C.ink}>Evolutionary Surrogate-assisted Prescription</Small>
        <Small size={24}>Cognizant AI Lab · for problems where every real test is expensive</Small>
      </Headline>
      {phase >= 0 && car < 0.5 ? (
        <div style={{position: 'absolute', right: 70, top: 170, width: 470}}>
          {PHASES.map((p, i) => (
            <div key={p.key} style={{display: 'flex', gap: 16, alignItems: 'flex-start', padding: '12px 18px', marginBottom: 10, borderRadius: 14, background: i === phase ? 'rgba(20,32,70,0.9)' : 'rgba(8,14,36,0.55)', border: `1.5px solid ${i === phase ? p.color : 'transparent'}`, opacity: i <= phase ? 1 : 0.35}}>
              <span style={{fontFamily: FONT.display, fontWeight: 700, fontSize: 40, color: p.color, width: 34}}>{p.key}</span>
              <div>
                <div style={{fontFamily: FONT.body, fontWeight: 700, fontSize: 26, color: C.ink}}>{p.name}</div>
                <div style={{fontFamily: FONT.body, fontSize: 21, color: C.dim}}>{p.text}</div>
              </div>
            </div>
          ))}
        </div>
      ) : null}
      <Headline t={ease(frame, cue(6) + 30, 20)} y={-310}>
        <div style={{display: 'flex', gap: 22, alignItems: 'center'}}>
          <Chip text="simulate thousands" color={C.specialist} size={30} />
          <span style={{fontSize: 40, color: C.dim}}>→</span>
          <Chip text="build the best few" color={C.ok} size={30} />
        </div>
      </Headline>
    </AbsoluteFill>
  );
};

const FEATURES = Array.from({length: 13}).map((_, i) => 0.25 + random(`feat-${i}`) * 1.3);

type Pt = {acc: number; cost: number; size: number; front?: boolean};
const POINTS: Pt[] = (() => {
  const pts: Pt[] = Array.from({length: 44}).map((_, i) => ({
    acc: 0.15 + random(`pa-${i}`) * 0.8,
    cost: 0.1 + random(`pc-${i}`) * 0.85,
    size: 0.1 + random(`ps-${i}`) * 0.85,
  }));
  pts.forEach((p) => {
    p.front = !pts.some((q) => q !== p && q.acc >= p.acc && q.cost <= p.cost && q.size <= p.size && (q.acc > p.acc || q.cost < p.cost || q.size < p.size));
  });
  return pts;
})();

export const S11Predictor: React.FC = () => {
  const {frame, cue} = useScene();
  const models = ease(frame, cue(0) + 10, 30);
  const feats = ease(frame, cue(1), 30);
  const math = ease(frame, cue(2), 24);
  const pareto = ease(frame, cue(3), 40);
  const pipeline = 1 - pareto;
  const breed = ease(frame, cue(4), 30);
  const camA = orbit(-0.1 + Math.sin(frame * 0.004) * 0.1, 13.5, 1.5, [0, 0.9, 0]);
  const camB = orbit(0.65 + frame * 0.0025, 11, 3.5, [0, 0.2, 0]);
  const cam = {pos: lerp3(camA.pos, camB.pos, pareto), target: lerp3(camA.target!, camB.target!, pareto)};
  const flow = ((frame - cue(1)) % 60) / 60;
  return (
    <AbsoluteFill>
      <Stage cam={cam} floor={-2.6}>
        {pipeline > 0.001 ? (
          <group scale={[pipeline, pipeline, pipeline]}>
            <group position={[-6.2, 0.4, 0]} scale={[0.5, 0.5, 0.5]}>
              <Network nodes={designerNodes(false)} edges={designerEdges} frame={frame} />
            </group>
            <Label text="a team's shape" pos={[-6.2, -1.3, 0]} height={0.28} style={{color: C.dim, weight: 500}} />
            {FEATURES.map((h, i) => {
              const g = Math.max(0, Math.min(1, feats * 2 - i / 13));
              return g > 0.001 ? (
                <mesh key={i} position={[-3.9 + i * 0.22, -0.6 + (h * g) / 2, 0]}>
                  <boxGeometry args={[0.16, Math.max(0.01, h * g), 0.16]} />
                  <meshStandardMaterial color={C.accent} emissive={C.accent} emissiveIntensity={0.5} />
                </mesh>
              ) : null;
            })}
            <Label text="13 shape features" pos={[-2.6, -1.3, 0]} height={0.28} opacity={feats} style={{color: C.accent}} />
            {[
              {y: 1.3, name: 'accuracy model', color: C.ok},
              {y: -1.1, name: 'token-cost model', color: C.tool},
            ].map((m, i) => (
              <group key={m.name} position={[1.4, m.y, 0]} scale={[models, models, models]}>
                <mesh>
                  <boxGeometry args={[2.4, 1.2, 1.2]} />
                  <meshStandardMaterial color="#1a2550" emissive={m.color} emissiveIntensity={0.35} transparent opacity={0.85} />
                </mesh>
                <Glow color={m.color} size={4} opacity={0.35} />
                <Label text={m.name} pos={[0, 0, 0.7]} height={0.38} style={{color: '#ffffff'}} />
                <Link from={[-4.0, -m.y, 0]} to={[-1.2, 0, 0]} progress={feats} color={C.accent} opacity={0.5 * feats} />
                <Pulse from={[-4.0, -m.y, 0]} to={[-1.2, 0, 0]} t={feats >= 1 ? (flow + i * 0.5) % 1 : 0} color={C.accent} size={0.08} />
                <Link from={[1.2, 0, 0]} to={[3.6, 0.1 - m.y, 0]} progress={math} color={m.color} opacity={0.7 * math} />
              </group>
            ))}
            <group position={[5.0, 0.1, 0]} scale={[math, math, math]}>
              <mesh>
                <sphereGeometry args={[0.55, 32, 24]} />
                <meshStandardMaterial color={C.coordinator} emissive={C.coordinator} emissiveIntensity={0.6} />
              </mesh>
              <Glow color={C.coordinator} size={3.4} opacity={0.6} />
              <Label text="fitness" pos={[0, 0.95, 0]} height={0.36} style={{color: '#ffd9a0', weight: 700}} />
              <Label text="plain arithmetic" pos={[0, -0.9, 0]} height={0.26} style={{color: C.dim, weight: 500}} />
            </group>
          </group>
        ) : null}
        {pareto > 0.001 ? (
          <group scale={[pareto, pareto, pareto]} position={[-1.8, -1.6, -1.8]}>
            {[
              {to: [4.6, 0, 0] as V3, label: 'accuracy →', color: C.ok},
              {to: [0, 3.6, 0] as V3, label: 'cheaper ↑', color: C.tool},
              {to: [0, 0, 4.6] as V3, label: 'smaller →', color: C.specialist},
            ].map((ax) => (
              <React.Fragment key={ax.label}>
                <Link from={[0, 0, 0]} to={ax.to} color={ax.color} radius={0.025} opacity={0.8} />
                <Label text={ax.label} pos={[ax.to[0] * 1.08, ax.to[1] * 1.08 + 0.2, ax.to[2] * 1.08]} height={0.3} style={{color: ax.color}} />
              </React.Fragment>
            ))}
            {POINTS.map((p, i) => {
              const pos: V3 = [p.acc * 4.4, (1 - p.cost) * 3.4, (1 - p.size) * 4.4];
              const lit = p.front ? ease(frame, cue(4) - 30 + (i % 7) * 4, 16) : 0;
              const fade = p.front ? 1 : 1 - 0.65 * ease(frame, cue(4) - 20, 20);
              return (
                <group key={i} position={pos}>
                  <mesh>
                    <sphereGeometry args={[p.front ? 0.13 + 0.05 * lit : 0.1, 16, 12]} />
                    <meshStandardMaterial color={p.front ? interpolate(lit) : '#6f86c4'} emissive={p.front ? C.strong : '#24366b'} emissiveIntensity={0.3 + 0.7 * lit} transparent opacity={fade} />
                  </mesh>
                  {lit > 0.01 ? <Glow color={C.strong} size={1.1} opacity={lit * 0.7} /> : null}
                </group>
              );
            })}
          </group>
        ) : null}
      </Stage>
      <Card x={1260} y={190} w={540} t={hold(frame, cue(0) + 30, cue(1) + 10, 14)} accent={C.tool}>
        <Small color={C.tool} size={22}>THE PREDICTOR</Small>
        <div style={{fontSize: 30, fontWeight: 600, marginTop: 6}}>One small model per outcome</div>
        <Small size={24}>gradient-boosted trees, learned from real exams</Small>
      </Card>
      <div style={{position: 'absolute', left: 0, right: 0, top: 165, display: 'flex', justifyContent: 'center', opacity: math * pipeline}}>
        <Card x={0} y={0} w={1500} t={math} accent={C.coordinator} center inline>
          <Mono size={28}>fitness = <span style={{color: C.ok}}>accuracy</span> − 0.06 · min(<span style={{color: C.tool}}>tokens</span>/600000, 1) − 0.02 · (<span style={{color: C.specialist}}>agents</span>/9)</Mono>
          <Small size={24}>
            <span style={{color: C.ok}}>predicted</span> · <span style={{color: C.tool}}>predicted</span> · <span style={{color: C.specialist}}>counted, never predicted</span>
          </Small>
        </Card>
      </div>
      <div style={{position: 'absolute', right: 70, top: 180, display: 'flex', flexDirection: 'column', gap: 14}}>
        {[
          ['cheapest', C.tool],
          ['best camera', C.ok],
          ['longest battery', C.specialist],
        ].map(([t, color], i) => (
          <Card key={t} x={0} y={0} w={380} t={hold(frame, cue(3) + 60 + i * 25, cue(4) + 20, 14)} accent={color} inline>
            <Small color={color} size={20}>PHONE {i + 1}</Small>
            <div style={{fontSize: 30, fontWeight: 700}}>{t}</div>
          </Card>
        ))}
      </div>
      <Card x={1240} y={210} w={560} t={breed} accent={C.strong}>
        <Small color={C.strong} size={22}>PARETO FRONT</Small>
        <div style={{fontSize: 30, fontWeight: 600, marginTop: 6}}>Beaten on every count by no one.</div>
        <Small size={24}>These teams get to breed.</Small>
      </Card>
    </AbsoluteFill>
  );
};

const interpolate = (t: number) => (t > 0.5 ? C.strong : '#8fa6e0');

/** Six teams in true order (top) and in the order a predictor puts them (bottom). */
const Ranking: React.FC<{x: number; order: number[]; color: string; title: string; t: number; muted: number; frame: number; shuffle?: number; axis?: boolean}> = ({x, order, color, title, t, muted, frame, shuffle = 0, axis}) => {
  if (t <= 0.001) return null;
  const xs = (i: number) => x + (i - 2.5) * 0.85;
  const shuffled = shuffle > 0 ? [...order].sort((a, b) => random(`sh-${Math.floor(frame / 10)}-${a}`) - random(`sh-${Math.floor(frame / 10)}-${b}`)) : order;
  const k = 1 - 0.7 * muted;
  return (
    <group scale={[t, t, t]}>
      <Label text={title} pos={[x, 2.4, 0]} height={0.36} opacity={k} style={{color, weight: 700}} />
      <Label text="true order" pos={[x - 3.3, 1.2, 0]} height={0.26} opacity={axis ? k : 0} style={{color: C.dim, weight: 500}} />
      <Label text="predicted" pos={[x - 3.3, -1.0, 0]} height={0.26} opacity={axis ? k : 0} style={{color: C.dim, weight: 500}} />
      {Array.from({length: 6}).map((_, i) => {
        const j = shuffled.indexOf(i);
        const top: V3 = [xs(i), 1.2, 0];
        const bot: V3 = [xs(j), -1.0, 0];
        const r = 0.12 + i * 0.035;
        return (
          <React.Fragment key={i}>
            <Link from={top} to={bot} color={color} radius={0.022} opacity={0.75 * k} />
            <mesh position={top}>
              <sphereGeometry args={[r, 20, 14]} />
              <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5 * k} transparent opacity={k} />
            </mesh>
            <mesh position={bot}>
              <sphereGeometry args={[r, 20, 14]} />
              <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5 * k} transparent opacity={k} />
            </mesh>
          </React.Fragment>
        );
      })}
    </group>
  );
};

export const S12GateNull: React.FC = () => {
  const {frame, cue} = useScene();
  const show = ease(frame, cue(0), 24);
  const acc = ease(frame, cue(1), 24);
  const tok = ease(frame, cue(1) + 90, 24);
  const combined = hold(frame, cue(2), cue(3), 14);
  const shuffle = frame >= cue(3) + 20 && frame < cue(3) + 150 ? 1 : 0;
  const muted = ease(frame, cue(3) + 160, 24);
  const cam = orbit(Math.sin(frame * 0.004) * 0.12, 13.5, 1.0, [0, 0.4, 0]);
  return (
    <AbsoluteFill>
      <Stage cam={cam} floor={-2.6}>
        <Ranking x={-3.6} order={[0, 1, 3, 2, 4, 5]} color={C.ok} title="accuracy model" t={show * Math.max(0.001, acc)} muted={0} frame={frame} axis />
        <Ranking x={3.6} order={[5, 4, 3, 2, 1, 0]} color={C.tool} title="token-cost model" t={show * Math.max(0.001, tok)} muted={muted} frame={frame} shuffle={shuffle} />
        {muted > 0.01 ? <Label text="MUTED" pos={[3.6, 0.1, 0.5]} height={0.8 * muted} opacity={muted} style={{color: C.danger, weight: 700, font: 'Space Grotesk', bg: 'rgba(40,6,14,0.75)', border: C.danger}} /> : null}
      </Stage>
      <div style={{position: 'absolute', left: 0, right: 0, top: 770, display: 'flex', justifyContent: 'space-around', padding: '0 260px', opacity: 1 - ease(frame, cue(3), 14)}}>
        <div style={{opacity: ease(frame, cue(1) + 40, 14), fontFamily: FONT.display, fontWeight: 700, fontSize: 40, color: C.ok}}>✓ better than chance</div>
        <div style={{opacity: ease(frame, cue(1) + 140, 14), fontFamily: FONT.display, fontWeight: 700, fontSize: 40, color: C.danger}}>✗ backwards</div>
      </div>
      <Headline t={combined} y={-310}>
        <Card x={0} y={0} w={900} t={combined} accent={C.coordinator} center inline>
          <Small color={C.coordinator} size={22}>ONE COMBINED SCORE</Small>
          <div style={{fontSize: 34, fontWeight: 600, marginTop: 6}}>looked healthy · accuracy carried the average</div>
        </Card>
      </Headline>
      <div style={{position: 'absolute', left: 0, right: 0, top: 780, textAlign: 'center', opacity: shuffle ? 1 : 0, fontFamily: FONT.mono, fontSize: 30, color: C.accent}}>shuffling the data… = what pure chance would score</div>
      <div style={{position: 'absolute', left: 0, right: 0, top: 770, display: 'flex', justifyContent: 'center', gap: 40}}>
        <Card x={0} y={0} w={560} t={ease(frame, cue(4) + 10, 16)} accent={C.ok} inline>
          <Small color={C.ok} size={22}>RAIN FORECAST</Small>
          <div style={{fontSize: 30, fontWeight: 600}}>Beats a coin flip → keep listening</div>
        </Card>
        <Card x={0} y={0} w={560} t={ease(frame, cue(4) + 50, 16)} accent={C.danger} inline>
          <Small color={C.danger} size={22}>WIND FORECAST</Small>
          <div style={{fontSize: 30, fontWeight: 600}}>Worse than a coin → stop asking</div>
        </Card>
      </div>
    </AbsoluteFill>
  );
};
