import React from 'react';
import {AbsoluteFill, random} from 'remotion';
import * as THREE from 'three';
import {ease, hold, lerp, lerp3, pop, split} from '../anim';
import {useScene} from '../scene';
import {Agent, Glow, Label, Link, Plaque, V3} from '../three/primitives';
import {Network, NodeSpec} from '../three/network';
import {orbit, Stage} from '../three/Stage';
import {Big, Card, Chip, Headline, Small} from '../overlay/ui';
import {C, FONT} from '../theme';
import {designerEdges, designerNodes} from './intro';

const GRID = Array.from({length: 120}).map((_, i) => ({
  pos: [((i % 12) - 5.5) * 0.62, 0, (Math.floor(i / 12) - 4.5) * 0.62] as V3,
  q: random(`pool-q-${i}`),
}));

/** A replicate search: a short walk across the pool, seeded by run. */
const walk = (run: number) => Array.from({length: 6}).map((_, k) => Math.floor(random(`walk-${run}-${k}`) * 120));

export const S16V2: React.FC = () => {
  const {frame, cue} = useScene();
  const coins = hold(frame, cue(1), cue(2), 16);
  const pq = hold(frame, cue(2), cue(3), 16);
  const pool = hold(frame, cue(3), cue(4) + 10, 16);
  const rehearsal = ease(frame, cue(4), 24);
  const cam = pool > 0.5 ? orbit(0.3 + frame * 0.002, 10, 6, [0, 0, 0]) : orbit(Math.sin(frame * 0.004) * 0.2, 11, 1.8, [0, 0.4, 0]);
  const run = Math.floor((frame - cue(3)) / 28);
  const runT = ((frame - cue(3)) % 28) / 28;
  return (
    <AbsoluteFill>
      <Stage cam={cam} floor={-2.2}>
        {coins > 0.001
          ? Array.from({length: 18}).map((_, i) => {
              const t = ease(frame, cue(1) + 10 + i * 6, 14);
              const pricey = i >= 5;
              const level = pricey ? i - 5 : i;
              return t > 0.001 ? (
                <mesh key={i} position={[pricey ? 1.5 : -1.5, lerp(3.5, -1.2 + level * 0.12, t), 0]} rotation={[0, i, 0]} scale={[coins, coins, coins]}>
                  <cylinderGeometry args={[0.6, 0.6, 0.11, 40]} />
                  <meshStandardMaterial color="#ffcf4a" emissive="#b8860b" emissiveIntensity={0.5} metalness={0.7} roughness={0.25} />
                </mesh>
              ) : null;
            })
          : null}
        {coins > 0.001 ? (
          <>
            <Label text="cheap model" pos={[-1.5, -1.75, 0.8]} height={0.34} opacity={coins} />
            <Label text="pricey model" pos={[1.5, -1.75, 0.8]} height={0.34} opacity={coins} />
          </>
        ) : null}
        {pq > 0.001
          ? [0.82, 0.55, 0.7, 0.38].map((m, k) => (
              <group key={k} position={[(k - 1.5) * 1.9, -1.3, 0]} scale={[pq * 0.85, pq * 0.85, pq * 0.85]}>
                {Array.from({length: 7}).map((_, e) => {
                  const h = Math.max(0.1, (m + (random(`ens-${k}-${e}`) - 0.5) * 0.25) * 3.6);
                  return (
                    <mesh key={e} position={[(e - 3) * 0.13, h / 2, 0]}>
                      <boxGeometry args={[0.1, h, 0.1]} />
                      <meshStandardMaterial color={C.accent} emissive={C.accent} emissiveIntensity={0.4} transparent opacity={0.55} />
                    </mesh>
                  );
                })}
                <mesh position={[0, m * 3.6, 0]}>
                  <boxGeometry args={[1.1, 0.06, 0.3]} />
                  <meshBasicMaterial color={C.coordinator} />
                </mesh>
                <Label text={`kind ${'ABCD'[k]}`} pos={[0, -0.4, 0.4]} height={0.3} />
              </group>
            ))
          : null}
        {pool > 0.001 ? (
          <group scale={[pool, pool, pool]}>
            {GRID.map((g, i) => {
              const inRun = walk(run).indexOf(i);
              const lit = inRun >= 0 && runT * 6 > inRun ? 1 : 0;
              return (
                <mesh key={i} position={[g.pos[0], 0.15 + g.q * 0.3, g.pos[2]]}>
                  <sphereGeometry args={[lit ? 0.14 : 0.1, 12, 8]} />
                  <meshStandardMaterial color={lit ? C.coordinator : '#5f7fc8'} emissive={lit ? C.coordinator : '#203a78'} emissiveIntensity={lit ? 1 : 0.5} />
                </mesh>
              );
            })}
            {walk(run).slice(0, Math.max(1, Math.ceil(runT * 6))).map((n, k, arr) =>
              k === 0 ? null : (
                <Link key={k} from={[GRID[arr[k - 1]].pos[0], 0.3, GRID[arr[k - 1]].pos[2]]} to={[GRID[n].pos[0], 0.3, GRID[n].pos[2]]} color={C.coordinator} radius={0.03} opacity={0.9} />
              ),
            )}
            <group position={[0, -1.4, -1.5]} rotation={[0.2, 0, 0]}>
              {GRID.filter((_, i) => i % 2 === 0).map((g, i) => (
                <mesh key={i} position={[g.pos[0], 0, g.pos[2]]}>
                  <sphereGeometry args={[0.07, 8, 6]} />
                  <meshBasicMaterial color="#8a6bff" transparent opacity={0.35} />
                </mesh>
              ))}
            </group>
            <Label text="shuffled pool = chance" pos={[0, -1.7, 3.8]} height={0.32} style={{color: '#c9b8ff'}} />
          </group>
        ) : null}
        {rehearsal > 0.001 ? (
          <group scale={[rehearsal, rehearsal, rehearsal]} rotation={[0.3, frame * 0.006, 0]}>
            <mesh rotation={[Math.PI / 2, 0, 0]}>
              <torusGeometry args={[3, 0.04, 8, 64]} />
              <meshBasicMaterial color={C.accent} wireframe transparent opacity={0.7} />
            </mesh>
            {[0, 1, 2, 3].map((i) => {
              const a = (i * Math.PI) / 2;
              return (
                <mesh key={i} position={[Math.cos(a) * 3, 0, Math.sin(a) * 3]}>
                  <icosahedronGeometry args={[0.45, 0]} />
                  <meshBasicMaterial color={C.accent} wireframe />
                </mesh>
              );
            })}
          </group>
        ) : null}
      </Stage>
      <Card x={1200} y={200} w={600} t={coins} accent="#ffcf4a">
        <Small color="#ffcf4a" size={22}>DOLLAR FITNESS</Small>
        <div style={{fontSize: 30, fontWeight: 600, marginTop: 6}}>accuracy − dollars per question − size</div>
        <Small size={24}>dollars from the provider's own accounting of each run</Small>
      </Card>
      <Card x={1200} y={200} w={600} t={pq} accent={C.accent}>
        <Small color={C.accent} size={22}>PER-QUESTION PREDICTOR</Small>
        <div style={{fontSize: 30, fontWeight: 600, marginTop: 6}}>chance of a right answer, per kind of question</div>
        <Small size={24}>thin bars = an ensemble; their spread = how sure it is</Small>
      </Card>
      <Card x={110} y={200} w={560} t={pool} accent={C.coordinator}>
        <Small color={C.coordinator} size={22}>POOL BENCHMARK</Small>
        <div style={{fontSize: 30, fontWeight: 600, marginTop: 6}}>120 teams, measured once</div>
        <Small size={24}>hundreds of free replayed searches, each tested against shuffled data</Small>
      </Card>
      <Headline t={rehearsal} y={-280}>
        <Big size={120} color={C.ok}>$0</Big>
        <Small size={32} color={C.ink}>rehearsal against a simulated provider</Small>
      </Headline>
    </AbsoluteFill>
  );
};

const NNLayers: React.FC<{t: number; frame: number}> = ({t, frame}) => {
  if (t <= 0.001) return null;
  const layers = [4, 6, 6, 3];
  return (
    <group scale={[t, t, t]} rotation={[0, Math.sin(frame * 0.01) * 0.3, 0]}>
      {layers.map((n, li) =>
        Array.from({length: n}).map((_, k) => {
          const p: V3 = [(li - 1.5) * 1.1, (k - (n - 1) / 2) * 0.45, 0];
          return (
            <React.Fragment key={`${li}-${k}`}>
              {li < layers.length - 1
                ? Array.from({length: layers[li + 1]}).map((__, k2) => (
                    <Link key={k2} from={p} to={[(li - 0.5) * 1.1, (k2 - (layers[li + 1] - 1) / 2) * 0.45, 0]} color="#5f7fc8" radius={0.008} opacity={0.4} />
                  ))
                : null}
              <mesh position={p}>
                <sphereGeometry args={[0.1, 12, 8]} />
                <meshStandardMaterial color={C.tool} emissive={C.tool} emissiveIntensity={0.6} />
              </mesh>
            </React.Fragment>
          );
        }),
      )}
    </group>
  );
};

const Box: React.FC<{pos: V3; color: string; label: string; sub: string; learned: number; s: number}> = ({pos, color, label, sub, learned, s}) =>
  s <= 0.001 ? null : (
    <group position={pos} scale={[s, s, s]}>
      <mesh>
        <boxGeometry args={[3.0, 1.6, 1.2]} />
        <meshStandardMaterial color="#17224a" emissive={color} emissiveIntensity={0.15 + 0.4 * learned} transparent opacity={0.35 + 0.5 * learned} wireframe={learned < 0.5} />
      </mesh>
      {learned > 0.5 ? <Glow color={color} size={4.5} opacity={0.4} /> : null}
      <Label text={label} pos={[0, 0.25, 0.7]} height={0.4} style={{weight: 700, color: '#ffffff'}} />
      <Label text={sub} pos={[0, -0.3, 0.7]} height={0.28} style={{weight: 500, color: learned > 0.5 ? '#cfe0ff' : '#9aa6c4'}} />
    </group>
  );

export const S17Names: React.FC = () => {
  const {frame, cue} = useScene();
  const boxes = hold(frame, cue(0), cue(2), 16);
  const here = ease(frame, cue(1), 24);
  const leaf = hold(frame, cue(2), cue(3), 16);
  const morph = ease(frame, cue(2) + 60, 60);
  const ctx = ease(frame, cue(3), 24);
  const future = ease(frame, cue(4), 20);
  const cam = orbit(Math.sin(frame * 0.004) * 0.15, 11, 1.4, [0, 0.3, 0]);
  const slider = 0.5 + 0.4 * Math.sin((frame - cue(3)) * 0.02);
  const pick = slider > 0.5 ? 0 : 1;
  const teamA: NodeSpec[] = designerNodes(false).slice(0, 4);
  const teamB: NodeSpec[] = [
    {id: 'co', pos: [0, 1.4, 0], color: C.coordinator},
    {id: 'r', pos: [0, -0.4, 0], color: C.researcher},
  ];
  return (
    <AbsoluteFill>
      <Stage cam={cam} floor={-2.8}>
        <Box pos={[-2.0, 0.3, 0]} color={C.tool} label="Predictor" sub="learned from real results" learned={1} s={boxes} />
        <Box pos={[2.0, 0.3, 0]} color={here > 0.5 ? '#7d8bb0' : C.coordinator} label="Prescriptor" sub={here > 0.5 ? '7 mutations + selection' : 'learned, in the ESP paper'} learned={1 - here} s={boxes} />
        <group position={[-3.2, 0.3, 0]}>
          <NNLayers t={leaf * (1 - morph)} frame={frame} />
        </group>
        {leaf > 0.001 ? (
          <group position={[-3.2, 0.6, 0]} scale={[0.7 * morph * leaf, 0.7 * morph * leaf, 0.7 * morph * leaf]}>
            <Network nodes={designerNodes(false)} edges={designerEdges} frame={frame} />
          </group>
        ) : null}
        <Label text={morph < 0.5 ? 'LEAF: evolves layers' : 'here: evolves agents'} pos={[-3.2, -1.6, 0]} height={0.42} opacity={leaf} style={{color: morph < 0.5 ? '#d9ccff' : '#ffd9a0', weight: 700}} />
        {ctx > 0.001 ? (
          <group scale={[ctx, ctx, ctx]}>
            <group position={[-2.6, 0.2, 0]} scale={[0.6, 0.6, 0.6]}>
              <Network nodes={teamA.map((n) => ({...n, dim: pick === 0 ? 0 : 0.8}))} edges={designerEdges.slice(0, 3)} frame={frame} />
            </group>
            <group position={[2.6, 0.2, 0]} scale={[0.6, 0.6, 0.6]}>
              <Network nodes={teamB.map((n) => ({...n, dim: pick === 1 ? 0 : 0.8}))} edges={[['co', 'r']]} frame={frame} />
            </group>
            <Glow pos={[pick === 0 ? -2.6 : 2.6, 0.2, -0.5]} color={C.ok} size={5} opacity={0.5} />
          </group>
        ) : null}
      </Stage>
      <div style={{position: 'absolute', left: 0, right: 0, top: 175, textAlign: 'center', opacity: boxes}}>
        <Chip text={here > 0.5 ? 'this project' : 'the ESP paper'} color={here > 0.5 ? C.coordinator : C.tool} size={30} />
      </div>
      {ctx > 0.001 ? (
        <div style={{position: 'absolute', left: 560, right: 560, top: 190, opacity: ctx * (1 - future * 0.5)}}>
          <div style={{display: 'flex', justifyContent: 'space-between', fontFamily: FONT.body, fontSize: 26, color: C.dim}}>
            <span>mostly totals</span>
            <span>mostly linking facts</span>
          </div>
          <div style={{position: 'relative', height: 14, borderRadius: 7, background: 'rgba(125,211,252,0.2)', marginTop: 10}}>
            <div style={{position: 'absolute', left: `${(1 - slider) * 100}%`, top: -9, width: 32, height: 32, marginLeft: -16, borderRadius: 16, background: C.accent, boxShadow: `0 0 20px ${C.accent}`}} />
          </div>
          <div style={{textAlign: 'center', marginTop: 16, fontFamily: FONT.mono, fontSize: 26, color: C.ink}}>workload + price cap → pick a team</div>
        </div>
      ) : null}
      <Headline t={future} y={300}>
        <div style={{display: 'inline-block', padding: '14px 30px', borderRadius: 16, border: `2px dashed ${C.dim}`, background: 'rgba(8,14,36,0.8)', fontFamily: FONT.display, fontWeight: 700, fontSize: 36, color: C.ink}}>
          future work: a <span style={{color: C.coordinator}}>learned</span> Prescriptor
        </div>
      </Headline>
    </AbsoluteFill>
  );
};

const LINES: {at: number; cmd: string; note: string}[] = [
  {at: 0, cmd: 'pip install -e ".[dev]"', note: '# no account, no key'},
  {at: 1, cmd: 'make validate', note: '# lint, docs, the full suite, an offline search'},
  {at: 2, cmd: 'make offline', note: '# breed and rank candidates, zero AI calls'},
  {at: 2.5, cmd: 'make holdout', note: '# select on half the questions, judge on the other'},
  {at: 3, cmd: 'cp .env.example .env', note: '# add a Gemini, Claude or GPT key'},
  {at: 3.5, cmd: 'make smoke', note: '# four questions to a team, end to end'},
];

export const S18Run: React.FC = () => {
  const {frame, cue, end} = useScene();
  const when = (at: number) => {
    const i = Math.floor(at);
    return at % 1 ? split(cue(i), end(i), 2, 1) : cue(i);
  };
  const service = ease(frame, cue(4), 30);
  const term = 1 - ease(frame, cue(4) - 5, 25);
  const cam = orbit(frame * 0.003, 12, 2, [0, 0, 0]);
  const tick = (frame * 0.02) % (Math.PI * 2);
  return (
    <AbsoluteFill>
      <Stage cam={cam} floor={-3}>
        <group position={[0, 0, -3]} scale={[0.8 * (1 - service), 0.8 * (1 - service), 0.8 * (1 - service)]} rotation={[0, frame * 0.004, 0]}>
          <Network nodes={designerNodes(false)} edges={designerEdges} frame={frame} />
        </group>
        {service > 0.001 ? (
          <group position={[0, 1.3, 0]} scale={[service * 0.8, service * 0.8, service * 0.8]} rotation={[0.35, 0, 0]}>
            <mesh rotation={[Math.PI / 2, 0, 0]}>
              <torusGeometry args={[2.6, 0.05, 12, 96]} />
              <meshStandardMaterial color={C.accent} emissive={C.accent} emissiveIntensity={0.6} />
            </mesh>
            <Agent pos={[Math.cos(tick) * 2.6, 0, Math.sin(tick) * 2.6]} color={C.coordinator} r={0.22} frame={frame} />
          </group>
        ) : null}
      </Stage>
      <div style={{position: 'absolute', left: 300, right: 300, top: 180, opacity: term, transform: `perspective(1600px) rotateX(${8 * (1 - term)}deg) scale(${0.85 + 0.15 * term})`}}>
        <div style={{borderRadius: 18, overflow: 'hidden', border: '1.5px solid rgba(125,211,252,0.35)', boxShadow: '0 30px 80px rgba(0,0,0,0.6)'}}>
          <div style={{background: '#10193a', padding: '12px 18px', display: 'flex', gap: 10}}>
            {['#ff5f57', '#febc2e', '#28c840'].map((c) => (
              <div key={c} style={{width: 16, height: 16, borderRadius: 8, background: c}} />
            ))}
            <span style={{marginLeft: 12, fontFamily: FONT.mono, fontSize: 20, color: C.dim}}>neuro-san-esp</span>
          </div>
          <div style={{background: 'rgba(5,9,22,0.98)', padding: '22px 28px', minHeight: 420}}>
            {LINES.map((l) => {
              const start = when(l.at);
              const t = Math.max(0, Math.min(1, (frame - start) / 22));
              if (frame < start) return null;
              return (
                <div key={l.cmd} style={{fontFamily: FONT.mono, fontSize: 30, lineHeight: 1.75}}>
                  <span style={{color: C.ok}}>$ </span>
                  <span style={{color: C.ink}}>{l.cmd.slice(0, Math.ceil(l.cmd.length * t))}</span>
                  <span style={{color: '#7f8db3', opacity: t >= 1 ? 1 : 0}}>{'  '}{l.note}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
      <div style={{position: 'absolute', left: 0, right: 0, top: 640, display: 'flex', justifyContent: 'center', gap: 16, opacity: ease(frame, cue(4) + 40, 20)}}>
        <Chip text="hourly wake" color={C.coordinator} size={28} />
        <Chip text="spend what the budget allows" color={C.accent} size={28} />
        <Chip text="save after every candidate" color={C.ok} size={28} />
        <Chip text="stop" color={C.danger} size={28} />
      </div>
      <div style={{position: 'absolute', left: 0, right: 0, top: 730, textAlign: 'center', opacity: ease(frame, split(cue(4), end(4), 2, 1), 20)}}>
        <Chip text="web page: ask a team · measure teams" color={C.tool} size={28} />
      </div>
    </AbsoluteFill>
  );
};

const LIMITS = [
  {title: 'ONE DOMAIN', body: 'stability across questions,\nnot transfer to new kinds of work', color: C.coordinator},
  {title: 'THREE MODELS', body: 'gemini-3.1-flash-lite\ngpt-5.4-nano · gpt-5.4-mini', color: C.specialist},
  {title: 'NOT A NEW IDEA', body: 'AgentSquare (ICLR 2025) also uses\na predictor to search agent designs', color: C.danger},
  {title: 'WHAT IS NEW', body: 'a fitness function for neuro-san,\nwhich had none at all', color: C.ok},
  {title: 'CHECKED BY TESTS', body: 'every README number is\nrecomputed from committed data', color: C.tool},
];

export const S19Limits: React.FC = () => {
  const {frame, cue, end} = useScene();
  const at = [cue(1), cue(2), cue(3), split(cue(3), end(3), 2, 1), cue(4)];
  const shown = at.map((t) => ease(frame, t, 24));
  const focus = shown.reduce((a, b) => a + b, 0) - 1;
  const xOf = (i: number) => (i - 2) * 4.7;
  const fx = lerp(xOf(0), xOf(4), Math.max(0, focus) / 4);
  const wide = ease(frame, end(4) - 40, 50);
  const near = {pos: [fx * 0.85, 0.8, 8] as V3, target: [fx, 0.2, 0] as V3};
  const far = {pos: [0, 1.5, 21] as V3, target: [0, 0.2, 0] as V3};
  const cam = {pos: lerp3(near.pos, far.pos, wide), target: lerp3(near.target, far.target, wide)};
  return (
    <AbsoluteFill>
      <Stage cam={cam} floor={-2.4}>
        {LIMITS.map((l, i) => {
          const s = pop(frame, at[i], 22);
          return s > 0.001 ? (
            <group key={l.title} position={[xOf(i), 0.3, -Math.abs(i - 2) * 0.6]} rotation={[0, -(i - 2) * 0.12, 0]} scale={[s, s, s]}>
              <mesh position={[0, 0, -0.05]}>
                <planeGeometry args={[4.35, 2.65]} />
                <meshBasicMaterial color={l.color} transparent opacity={0.6} side={THREE.DoubleSide} />
              </mesh>
              <mesh position={[0, 0, -0.03]}>
                <planeGeometry args={[4.2, 2.5]} />
                <meshBasicMaterial color="#0e1838" side={THREE.DoubleSide} />
              </mesh>
              <Glow pos={[0, 0, -0.2]} color={l.color} size={6} opacity={0.25} />
              <Plaque text={l.title} pos={[0, 0.72, 0]} height={0.34} style={{color: l.color, weight: 700, font: 'Space Grotesk'}} />
              <Plaque text={l.body} pos={[0, -0.25, 0]} height={0.62} style={{size: 60, weight: 500, color: '#e6ecff'}} />
            </group>
          ) : null;
        })}
      </Stage>
    </AbsoluteFill>
  );
};

const PILLARS = [
  {word: 'MEASURE', sub: 'an exam no model has seen', color: C.specialist},
  {word: 'SEARCH', sub: 'evolution + a cheap Predictor', color: C.coordinator},
  {word: 'PROVE', sub: 'on questions never chosen with', color: C.ok},
];

export const S20Recap: React.FC = () => {
  const {frame, cue, dur} = useScene();
  const at = [cue(1), cue(2), cue(3)];
  const best = ease(frame, cue(4), 30);
  const outro = ease(frame, cue(5), 30);
  const cam = orbit(Math.sin(frame * 0.004) * 0.25, lerp(13, 16, outro), 2.4, [0, 0.8, 0]);
  return (
    <AbsoluteFill>
      <Stage cam={cam} floor={-2}>
        <group position={[0, 0, 0]} scale={[1 - best * 0.35, 1 - best * 0.35, 1 - best * 0.35]}>
          {PILLARS.map((p, i) => {
            const h = 3.2 * ease(frame, at[i], 30);
            const x = (i - 1) * 4.2;
            return h > 0.01 ? (
              <group key={p.word} position={[x, -2 - best * 0.6, -best * 3]}>
                <mesh position={[0, h / 2, 0]}>
                  <boxGeometry args={[2.6, h, 0.9]} />
                  <meshStandardMaterial color="#14214a" emissive={p.color} emissiveIntensity={0.35} transparent opacity={0.9} />
                </mesh>
                <Glow pos={[0, h, 0]} color={p.color} size={4} opacity={0.5} />
                <Plaque text={p.word} pos={[0, h * 0.62, 0.48]} height={0.5} opacity={h / 3.2} style={{color: '#ffffff', weight: 700, font: 'Space Grotesk'}} />
                <Plaque text={p.sub} pos={[0, h * 0.62 - 0.6, 0.48]} height={0.24} opacity={h / 3.2} style={{color: '#cfe0ff', weight: 500}} />
              </group>
            ) : null;
          })}
        </group>
        {best > 0.001 ? (
          <group position={[0, 3.6, 1]} scale={[0.6 * best * (1 - outro), 0.6 * best * (1 - outro), 0.6 * best * (1 - outro)]}>
            <Network
              nodes={[
                {id: 'co', pos: [0, 1.2, 0], color: C.coordinator},
                {id: 'r', pos: [0, -0.6, 0], color: C.strong, strong: 1},
                {id: 't', pos: [0, -2.2, 0], tool: true},
              ]}
              edges={[['co', 'r'], ['r', 't']]}
              frame={frame}
            />
          </group>
        ) : null}
      </Stage>
      <Headline t={outro * (1 - ease(frame, dur - 20, 15))} y={-250}>
        <Card x={0} y={0} w={1100} t={1} accent={C.accent} center inline>
          <Big size={72}>neuro-san-esp</Big>
          <div style={{fontFamily: FONT.mono, fontSize: 34, color: C.accent, marginTop: 12}}>github.com/Sivakumarraj/neuro-san-esp</div>
          <Small size={24}>Built on neuro-san by Cognizant AI Lab · Apache-2.0 · narration: synthetic voice</Small>
        </Card>
      </Headline>
    </AbsoluteFill>
  );
};
