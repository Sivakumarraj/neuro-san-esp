import React from 'react';
import {AbsoluteFill, random} from 'remotion';
import {ease, hold, lerp, lerp3, pop, split} from '../anim';
import {useScene} from '../scene';
import {Agent, Doc, Glow, Label, Link, Pulse, Sheet, V3} from '../three/primitives';
import {Network} from '../three/network';
import {orbit, Stage} from '../three/Stage';
import {Big, Card, Chip, Headline, Mono, Small} from '../overlay/ui';
import {C, FONT} from '../theme';
import {designerEdges, designerNodes} from './intro';

const DEPOTS = Array.from({length: 24}).map((_, i) => {
  const a = random(`dep-a-${i}`) * Math.PI * 2;
  const r = 1.4 + random(`dep-r-${i}`) * 5.2;
  return {pos: [Math.sin(a) * r, 0, Math.cos(a) * r * 0.8] as V3, h: 0.4 + random(`dep-h-${i}`) * 1.1, at: random(`dep-t-${i}`)};
});

export const S05World: React.FC = () => {
  const {frame, cue, end} = useScene();
  const exam = hold(frame, cue(0), cue(2), 14);
  const memory = hold(frame, cue(1) + 10, cue(2), 14);
  const map = ease(frame, cue(2), 40);
  const docs = ease(frame, cue(3), 120);
  const book = ease(frame, cue(4), 30);
  const camA = orbit(0.2, 11, 1.2, [0, 0.4, 0]);
  const camB = orbit(0.6 + frame * 0.002, 15, 7.5, [0, 0.5, 0]);
  const m = ease(frame, cue(2) - 10, 50);
  const cam = {pos: lerp3(camA.pos, camB.pos, m), target: lerp3(camA.target!, camB.target!, m)};
  return (
    <AbsoluteFill>
      <Stage cam={cam} floor={-0.02} floorOpacity={0.25 * map}>
        <Doc
          spec={{head: 'EXAM', kind: 'question sheet', accent: C.coordinator, rows: [['Q1', 'Who manages depot D08?'], ['Q2', 'Which city is depot ...'], ['Q3', 'What goods were ...']]}}
          pos={[0, 0.5, 0]}
          scale={1.6 * pop(frame, cue(0), 20)}
          opacity={exam}
        />
        <group position={[-3.6, 0.4, 0.5]} scale={[memory, memory, memory]}>
          <Agent pos={[0, 0, 0]} color={C.specialist} r={0.6} frame={frame} />
          <Label text="“I remember this from training…”" pos={[0, 1.3, 0]} height={0.32} style={{color: '#cfe0ff', weight: 500}} />
        </group>
        {map > 0.001 ? (
          <group>
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} scale={[map, map, map]}>
              <circleGeometry args={[7.4, 96]} />
              <meshStandardMaterial color="#0c1a3d" emissive="#0a1638" emissiveIntensity={0.6} transparent opacity={0.85} />
            </mesh>
            {DEPOTS.map((d, i) => {
              const up = ease(frame, cue(2) + 10 + d.at * 50, 18);
              return up > 0.001 ? (
                <group key={i} position={d.pos}>
                  <mesh position={[0, (d.h * up) / 2, 0]} scale={[1, Math.max(0.01, up), 1]}>
                    <boxGeometry args={[0.42, d.h, 0.42]} />
                    <meshStandardMaterial color={C.coordinator} emissive={C.coordinator} emissiveIntensity={0.35} roughness={0.4} />
                  </mesh>
                  <Glow pos={[0, 0.05, 0]} color={C.coordinator} size={1.4} opacity={0.35 * up} />
                </group>
              ) : null;
            })}
            {Array.from({length: 124}).map((_, i) => {
              const d = DEPOTS[i % 24];
              const t = Math.max(0, Math.min(1, docs * 1.6 - (i / 124) * 0.6));
              const ang = random(`doc-a-${i}`) * Math.PI * 2 + frame * 0.004;
              const rad = 1.6 + random(`doc-r-${i}`) * 4.4;
              const dest: V3 = [Math.sin(ang) * rad, 1.9 + random(`doc-y-${i}`) * 1.5, Math.cos(ang) * rad];
              const intoBook = ease(frame, cue(4) + (i % 30), 40);
              const p = lerp3(lerp3([d.pos[0], d.h, d.pos[2]], dest, t), [0, 2.2, 0], intoBook);
              return t > 0.001 && intoBook < 1 ? <Sheet key={i} pos={p} rot={[0.3, ang, 0.1]} opacity={Math.min(1, t * 3)} scale={0.75 * (1 - intoBook)} /> : null;
            })}
            <Label text="MERIDIAN LOGISTICS · invented" pos={[0, -0.6, 7.6]} height={0.5} opacity={map} style={{color: '#ffd9a0', weight: 700, font: 'Space Grotesk'}} />
          </group>
        ) : null}
        {book > 0.001 ? (
          <group position={[0, 2.2, 0]} scale={[book * 1.6, book * 1.6, book * 1.6]} rotation={[0.35, frame * 0.004, 0]}>
            <mesh position={[-0.75, 0, 0]} rotation={[0, 0, 0.18]}>
              <boxGeometry args={[1.5, 0.06, 1.9]} />
              <meshStandardMaterial color={C.paper} emissive={C.paper} emissiveIntensity={0.4} />
            </mesh>
            <mesh position={[0.75, 0, 0]} rotation={[0, 0, -0.18]}>
              <boxGeometry args={[1.5, 0.06, 1.9]} />
              <meshStandardMaterial color={C.paper} emissive={C.paper} emissiveIntensity={0.4} />
            </mesh>
            <Glow color={C.accent} size={6} opacity={0.6} />
          </group>
        ) : null}
      </Stage>
      <Card x={1180} y={300} w={560} t={memory} accent={C.danger}>
        <Small color={C.danger} size={22}>REAL COMPANY</Small>
        <div style={{fontSize: 32, fontWeight: 600, marginTop: 6}}>The model may answer from memory.</div>
        <Small size={26}>Then the exam says nothing about the team.</Small>
      </Card>
      <div style={{position: 'absolute', right: 80, top: 170, display: 'flex', flexDirection: 'column', gap: 16, opacity: 1 - ease(frame, cue(4) + 20, 20)}}>
        {[
          ['24', 'depots', C.coordinator],
          ['40', 'contracts', C.specialist],
          ['60', 'incidents', C.danger],
          ['124', 'documents', C.paper],
        ].map(([n, label, color], i) => {
          const t = ease(frame, split(cue(3), end(3), 4, i), 14);
          return t > 0.001 ? (
            <div key={label} style={{opacity: t, transform: `translateX(${(1 - t) * 30}px)`, display: 'flex', alignItems: 'baseline', gap: 16, padding: '12px 26px', borderRadius: 16, background: 'rgba(8,14,36,0.78)', border: `1.5px solid ${color}66`, minWidth: 330}}>
              <span style={{fontFamily: FONT.display, fontWeight: 700, fontSize: 60, color}}>{n}</span>
              <span style={{fontFamily: FONT.body, fontSize: 30, color: C.ink}}>{label}</span>
            </div>
          ) : null;
        })}
      </div>
      <Headline t={ease(frame, cue(4) + 40, 24)} y={-290}>
        <Card x={0} y={0} w={980} t={1} accent={C.accent} center inline>
          <Big size={52}>An open-book exam,</Big>
          <Big size={52} color={C.accent}>on a book nobody has read.</Big>
        </Card>
      </Headline>
    </AbsoluteFill>
  );
};

const CHAIN: {x: number; spec: Parameters<typeof Doc>[0]['spec']}[] = [
  {x: -7.2, spec: {head: 'T11', kind: 'exam question', accent: C.coordinator, rows: [['Question', 'Incident INC-4413 hit a'], ['', 'contract serviced by a'], ['', "depot. Who is that"], ['', "depot's manager?"]]}},
  {x: -3.6, spec: {head: 'INC-4413', kind: 'incident report', accent: '#ff8a9a', rows: [['Contract', 'C-2113'], ['Hours late', '18'], ['Cause', 'a bridge closure'], ['Year', '2024']], highlight: [0]}},
  {x: 0, spec: {head: 'C-2113', kind: 'contract', accent: '#7dd3fc', rows: [['Client', 'Ironbridge Steel'], ['Depot', 'D04'], ['Goods', 'flat-pack furniture'], ['Penalty', '1,875 per hour']], highlight: [1]}},
  {x: 3.6, spec: {head: 'D04', kind: 'depot', accent: '#ffc76a', rows: [['City', 'Ashford'], ['Manager', 'R. Delacroix'], ['Loading bays', '8'], ['Opened', '2003']], highlight: [1]}},
];
const ANSWER_X = 7.2;

export const S06Hops: React.FC = () => {
  const {frame, cue, end} = useScene();
  const z = (x: number) => -Math.abs(x) * 0.12;
  const focus = [
    {t: cue(0), x: -7.2},
    {t: cue(1), x: -3.6},
    {t: cue(2), x: 0},
    {t: cue(3), x: 3.6},
    {t: split(cue(3), end(3), 2, 1), x: 6.0},
  ];
  let fx = focus[0].x;
  for (let i = 1; i < focus.length; i++) fx = lerp(fx, focus[i].x, ease(frame, focus[i].t - 10, 30));
  const wide = ease(frame, cue(4), 45);
  const near = {pos: [fx * 0.92, 0.6, 7.2] as V3, target: [fx, 0.1, z(fx)] as V3};
  const far = {pos: [0 + Math.sin(frame * 0.004) * 1.5, 2.2, 15.5] as V3, target: [0, 0, -0.5] as V3};
  const cam = {pos: lerp3(near.pos, far.pos, wide), target: lerp3(near.target, far.target, wide)};
  const hopAt = [cue(1), cue(2), cue(3), split(cue(3), end(3), 2, 1)];
  const lit = hopAt.map((t) => ease(frame, t, 24));
  const hops = lit.filter((x) => x > 0.5).length;
  const points: V3[] = [...CHAIN.map((c) => [c.x, 0, z(c.x)] as V3), [ANSWER_X, 0, z(ANSWER_X)]];
  const glass = (frame - cue(4)) / Math.max(30, end(4) - cue(4));
  const glassPos = (() => {
    const t = Math.max(0, Math.min(0.999, glass)) * (points.length - 1);
    const i = Math.floor(t);
    return lerp3(points[i], points[i + 1], t - i);
  })();
  const correct = ease(frame, cue(5), 20);
  return (
    <AbsoluteFill>
      <Stage cam={cam} floor={-2.4}>
        {CHAIN.map((c, i) => (
          <Doc key={c.spec.head} spec={c.spec} pos={[c.x, 0, z(c.x)]} rot={[0, -c.x * 0.03, 0]} scale={1.3 * pop(frame, i === 0 ? cue(0) : hopAt[i - 1] - 6, 18)} glow={i === 0 ? 0.6 : lit[i - 1]} />
        ))}
        {points.slice(0, -1).map((p, i) => (
          <React.Fragment key={i}>
            <Link from={[p[0] + 1.05, 0, p[2]]} to={[points[i + 1][0] - 1.05, 0, points[i + 1][2]]} progress={lit[i]} color={C.coordinator} radius={0.05} opacity={0.9} />
            <Pulse from={[p[0] + 1.05, 0, p[2]]} to={[points[i + 1][0] - 1.05, 0, points[i + 1][2]]} t={(frame - hopAt[i]) / 24} color={C.coordinator} size={0.14} />
            <Label text={`hop ${i + 1}`} pos={[(p[0] + points[i + 1][0]) / 2, 0.42, (p[2] + points[i + 1][2]) / 2]} height={0.26} opacity={lit[i]} style={{color: '#ffd9a0', size: 56}} />
          </React.Fragment>
        ))}
        <Agent pos={[ANSWER_X, 0, z(ANSWER_X)]} color={C.ok} r={0.55} s={pop(frame, hopAt[3] + 10, 20)} label="R. Delacroix" sub="the answer" frame={frame} />
        {glass > 0 && glass < 1.05 ? (
          <group position={[glassPos[0], glassPos[1] + 1.65, glassPos[2] + 0.5]} rotation={[0.2, 0, -0.5]}>
            <mesh>
              <torusGeometry args={[0.42, 0.06, 16, 48]} />
              <meshStandardMaterial color="#dfe8ff" emissive="#9fb8ff" emissiveIntensity={0.6} metalness={0.6} roughness={0.2} />
            </mesh>
            <mesh>
              <circleGeometry args={[0.4, 48]} />
              <meshBasicMaterial color="#9fd8ff" transparent opacity={0.18} />
            </mesh>
            <mesh position={[0, -0.75, 0]}>
              <cylinderGeometry args={[0.06, 0.08, 0.7, 12]} />
              <meshStandardMaterial color="#c9a46a" />
            </mesh>
          </group>
        ) : null}
      </Stage>
      <div style={{position: 'absolute', right: 80, top: 170, opacity: ease(frame, cue(1), 14) * (1 - correct), fontFamily: FONT.body}}>
        <div style={{fontSize: 22, letterSpacing: 3, color: C.coordinator}}>HOPS</div>
        <div style={{fontFamily: FONT.display, fontSize: 84, fontWeight: 700, color: C.ink}}>
          {hops}<span style={{color: C.faint}}> / 4</span>
        </div>
      </div>
      <Headline t={correct} y={-320}>
        <div style={{display: 'inline-flex', alignItems: 'center', gap: 22, padding: '18px 34px', borderRadius: 20, background: 'rgba(8,30,20,0.8)', border: `1.5px solid ${C.ok}88`}}>
          <span style={{fontFamily: FONT.display, fontSize: 64, color: C.ok}}>✓</span>
          <span style={{fontFamily: FONT.display, fontWeight: 700, fontSize: 48, color: C.ink}}>Correct by construction</span>
        </div>
      </Headline>
    </AbsoluteFill>
  );
};

/** Real measurement: the designer's shape on the 17 built-in questions, first (Gemini) search. */
const BARS = [
  {key: 'Accuracy', value: '14 / 17', sub: '82.4%', h: 0.8235 * 3.6, color: C.ok},
  {key: 'Tokens', value: '385,280', sub: 'for 17 questions', h: (385280 / 600000) * 3.6, color: C.tool},
  {key: 'Agents', value: '4', sub: 'team size', h: (4 / 9) * 3.6, color: C.specialist},
];

export const S07Fitness: React.FC = () => {
  const {frame, cue, end} = useScene();
  const team = ease(frame, cue(0), 24);
  const formula = ease(frame, cue(4), 24);
  const clock = ease(frame, cue(5), 24);
  const lift = ease(frame, cue(4) - 10, 30);
  const cam = orbit(-0.25 + Math.sin(frame * 0.004) * 0.12, lerp(13.5, 15.5, lift), 2.2, [0.6, lerp(0.2, 1.4, lift), 0]);
  const barsOut = ease(frame, cue(5), 24);
  const flow = ((frame - cue(0)) % 50) / 50;
  return (
    <AbsoluteFill>
      <Stage cam={cam} floor={-1.8}>
        <group position={[-5.4, 0.7, 0]} scale={[0.62 * team, 0.62 * team, 0.62 * team]}>
          <Network nodes={designerNodes(false)} edges={designerEdges} frame={frame} pulses={[{edge: ['co', 'b'], t: flow, color: C.coordinator}]} />
        </group>
        <Label text="the designer's team sits the exam" pos={[-5.4, -1.4, 0]} height={0.26} opacity={team} style={{color: C.dim, weight: 500}} />
        {BARS.map((b, i) => {
          const g = ease(frame, cue(i + 1), 30) * (1 - barsOut);
          const x = -1.6 + i * 2.6;
          return g > 0.001 ? (
            <group key={b.key} position={[x, -1.8, 0]}>
              <mesh position={[0, (b.h * g) / 2, 0]}>
                <boxGeometry args={[1.3, Math.max(0.01, b.h * g), 1.3]} />
                <meshStandardMaterial color={b.color} emissive={b.color} emissiveIntensity={0.45} roughness={0.35} transparent opacity={0.92} />
              </mesh>
              <Glow pos={[0, b.h * g, 0]} color={b.color} size={2.4} opacity={0.5 * g} />
              <Label text={b.value} pos={[0, b.h * g + 1.05, 0]} height={0.5} opacity={g} style={{weight: 700, font: 'Space Grotesk'}} />
              <Label text={`${b.key} · ${b.sub}`} pos={[0, b.h * g + 0.55, 0]} height={0.3} opacity={g} style={{color: '#c7d3f2', weight: 500}} />
            </group>
          ) : null;
        })}
        {clock > 0.001 ? (
          <group position={[1.2, 0.4, 0.5]} scale={[clock * 1.6, clock * 1.6, clock * 1.6]}>
            <mesh>
              <torusGeometry args={[0.8, 0.07, 16, 64]} />
              <meshStandardMaterial color={C.coordinator} emissive={C.coordinator} emissiveIntensity={0.7} />
            </mesh>
            <mesh rotation={[0, 0, -frame * 0.08]} position={[0, 0, 0.02]}>
              <boxGeometry args={[0.06, 0.6, 0.04]} />
              <meshBasicMaterial color={C.ink} />
            </mesh>
            <Glow color={C.coordinator} size={3} opacity={0.4} />
          </group>
        ) : null}
      </Stage>
      <div style={{position: 'absolute', left: 0, right: 0, top: 160, display: 'flex', justifyContent: 'center', opacity: formula * (1 - ease(frame, cue(5) - 5, 15))}}>
        <Card x={0} y={0} w={1560} t={formula} accent={C.coordinator} center inline>
          <Small color={C.coordinator} size={22}>FITNESS · one score per team</Small>
          <div style={{marginTop: 10}}>
            <Mono size={31}>fitness = <span style={{color: C.ok}}>accuracy</span> − 0.06 · min(<span style={{color: C.tool}}>tokens</span> / 600000, 1) − 0.02 · (<span style={{color: C.specialist}}>agents</span> / 9)</Mono>
          </div>
          <div style={{marginTop: 14, opacity: ease(frame, split(cue(4), end(4), 3, 1), 16)}}>
            <Mono size={30} color={C.dim}>
              0.8235 − 0.0385 − 0.0089 = <span style={{color: C.coordinator, fontWeight: 700}}>0.7761</span>
            </Mono>
          </div>
          <div style={{marginTop: 14, opacity: ease(frame, split(cue(4), end(4), 3, 1.6), 16)}}>
            <Chip text="your marks" color={C.ok} size={24} />
            <Chip text="− small fine: expensive" color={C.tool} size={24} />
            <Chip text="− tiny fine: big" color={C.specialist} size={24} />
          </div>
        </Card>
      </div>
      <div style={{position: 'absolute', left: 0, right: 0, top: 190, display: 'flex', justifyContent: 'center', gap: 40}}>
        <Card x={0} y={0} w={560} t={ease(frame, cue(5) + 20, 18)} accent={C.coordinator} center inline>
          <Small color={C.coordinator} size={22}>RAN OUT OF TIME</Small>
          <Big size={58}>unfinished</Big>
        </Card>
        <Card x={0} y={0} w={560} t={ease(frame, cue(5) + 50, 18)} accent={C.danger} center inline>
          <Small color={C.danger} size={22}>WRONG ANSWER</Small>
          <Big size={58}>wrong</Big>
        </Card>
      </div>
      <Headline t={ease(frame, cue(5) + 80, 18)} y={300}>
        <Big size={44} color={C.dim}>not the same thing</Big>
      </Headline>
    </AbsoluteFill>
  );
};
