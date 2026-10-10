import React from 'react';
import {AbsoluteFill, interpolateColors, random} from 'remotion';
import * as THREE from 'three';
import {ease, hold, lerp, lerp3, pop, split} from '../anim';
import {useScene} from '../scene';
import {Agent, Glow, Label, Link, Tool, V3} from '../three/primitives';
import {orbit, Stage} from '../three/Stage';
import {Card, Chip, Headline, Small} from '../overlay/ui';
import {C, FONT} from '../theme';

const Helix: React.FC<{frame: number; s: number}> = ({frame, s}) => {
  if (s <= 0.001) return null;
  const n = 22;
  return (
    <group scale={[s, s, s]} rotation={[0, frame * 0.02, 0.25]}>
      {Array.from({length: n}).map((_, i) => {
        const y = (i - n / 2) * 0.32;
        const a = i * 0.55;
        const p1: V3 = [Math.cos(a) * 1.1, y, Math.sin(a) * 1.1];
        const p2: V3 = [-p1[0], y, -p1[2]];
        return (
          <React.Fragment key={i}>
            <Link from={p1} to={p2} color="#8fb3ff" radius={0.025} opacity={0.5} />
            <mesh position={p1}>
              <sphereGeometry args={[0.13, 16, 12]} />
              <meshStandardMaterial color={C.coordinator} emissive={C.coordinator} emissiveIntensity={0.6} />
            </mesh>
            <mesh position={p2}>
              <sphereGeometry args={[0.13, 16, 12]} />
              <meshStandardMaterial color={C.specialist} emissive={C.specialist} emissiveIntensity={0.6} />
            </mesh>
          </React.Fragment>
        );
      })}
    </group>
  );
};

type MNode = {pos: V3; color?: string; strong?: boolean; tool?: boolean; sub?: string};
type MState = {nodes: Record<string, MNode>; edges: [string, string][]};

const CO: MNode = {pos: [0, 1.7, 0], color: C.coordinator, sub: 'nano'};
const A: MNode = {pos: [-2.3, -0.3, 0], sub: 'nano'};
const B: MNode = {pos: [2.3, -0.3, 0], sub: 'nano'};
const TB: MNode = {pos: [2.3, -2.0, 0], tool: true};
const Cn: MNode = {pos: [0, -0.9, 1.0], sub: 'nano'};

/** Base network, then the state after each of the seven operators in turn. */
const STATES: MState[] = [
  {nodes: {co: CO, a: A, b: B, tb: TB}, edges: [['co', 'a'], ['co', 'b'], ['b', 'tb']]},
  {nodes: {co: CO, a: A, b: B, tb: TB, c: Cn}, edges: [['co', 'a'], ['co', 'b'], ['b', 'tb'], ['co', 'c']]},
  {nodes: {co: CO, b: B, tb: TB, c: Cn}, edges: [['co', 'b'], ['b', 'tb'], ['co', 'c']]},
  {nodes: {co: CO, b: B, tb: TB, c: Cn}, edges: [['c', 'b'], ['b', 'tb'], ['co', 'c']]},
  {nodes: {co: CO, b: B, tb: TB, c1: {pos: [-1.1, -0.9, 1.0], sub: 'nano'}, c2: {pos: [0.9, -0.9, 1.0], sub: 'nano'}}, edges: [['c2', 'b'], ['b', 'tb'], ['co', 'c1'], ['co', 'c2']]},
  {nodes: {co: CO, b: B, tb: TB, c: Cn}, edges: [['c', 'b'], ['b', 'tb'], ['co', 'c']]},
  {nodes: {co: CO, b: B, tb: TB, c: Cn, tc: {pos: [0, -2.6, 1.0], tool: true}}, edges: [['c', 'b'], ['b', 'tb'], ['co', 'c'], ['c', 'tc']]},
  {nodes: {co: CO, b: {...B, strong: true, sub: 'mini'}, tb: TB, c: Cn, tc: {pos: [0, -2.6, 1.0], tool: true}}, edges: [['c', 'b'], ['b', 'tb'], ['co', 'c'], ['c', 'tc']]},
];

/** Where split/merge nodes come from and go to, so they glide instead of popping. */
const ORIGIN: Record<string, string> = {c1: 'c', c2: 'c'};

const MorphNet: React.FC<{u: number; frame: number}> = ({u, frame}) => {
  const k = Math.max(0, Math.min(STATES.length - 1, Math.floor(u)));
  const k2 = Math.min(STATES.length - 1, k + 1);
  const f = Math.min(1, Math.max(0, (u - k) * 1.6));
  const a = STATES[k];
  const b = STATES[k2];
  const ids = Array.from(new Set([...Object.keys(a.nodes), ...Object.keys(b.nodes)]));
  const place = (st: MState, id: string): MNode | undefined => st.nodes[id] ?? (ORIGIN[id] ? st.nodes[ORIGIN[id]] : undefined);
  const nodes: Record<string, {pos: V3; s: number; n: MNode}> = {};
  ids.forEach((id) => {
    const na = a.nodes[id];
    const nb = b.nodes[id];
    const pa = place(a, id) ?? place(b, id) ?? {pos: [0, 0, 0] as V3};
    const pb = place(b, id) ?? place(a, id) ?? {pos: [0, 0, 0] as V3};
    let sa = na ? 1 : 0;
    let sb = nb ? 1 : 0;
    if (!na && ORIGIN[id] && a.nodes[ORIGIN[id]]) sa = 1;
    if (!nb && ORIGIN[id] && b.nodes[ORIGIN[id]]) sb = 1;
    nodes[id] = {pos: lerp3(pa.pos, pb.pos, f), s: lerp(sa, sb, f), n: (f < 0.5 ? na : nb) ?? na ?? nb!};
  });
  // a merged/split parent fades while its children carry the motion
  if (nodes.c && (a.nodes.c1 || b.nodes.c1)) nodes.c.s = a.nodes.c1 ? f : 1 - f;
  const edgeKey = (e: [string, string]) => e.join('-');
  const allEdges = Array.from(new Map([...a.edges, ...b.edges].map((e) => [edgeKey(e), e])).values());
  const inA = new Set(a.edges.map(edgeKey));
  const inB = new Set(b.edges.map(edgeKey));
  return (
    <group>
      {allEdges.map((e) => {
        const o = lerp(inA.has(edgeKey(e)) ? 1 : 0, inB.has(edgeKey(e)) ? 1 : 0, f);
        const p = nodes[e[0]];
        const q = nodes[e[1]];
        if (!p || !q) return null;
        return <Link key={edgeKey(e)} from={p.pos} to={q.pos} opacity={0.75 * o * Math.min(p.s, q.s)} />;
      })}
      {Object.entries(nodes).map(([id, nd]) =>
        nd.n.tool ? (
          <Tool key={id} pos={nd.pos} s={nd.s} frame={frame} />
        ) : (
          <Agent
            key={id}
            pos={nd.pos}
            color={nd.n.strong && f > 0.3 ? C.strong : nd.n.color ?? C.specialist}
            r={0.45}
            s={nd.s}
            sub={nd.n.sub}
            strong={nd.n.strong ? Math.min(1, f * 2) : 0}
            frame={frame}
          />
        ),
      )}
    </group>
  );
};

const OPS = ['add_agent', 'remove_agent', 'rewire', 'split_agent', 'merge_agents', 'toggle_search', 'reassign_model'];
const OPS_SAY = ['add an agent', 'remove an agent', 'rewire who talks to whom', 'split one into two', 'merge two into one', 'switch searching on/off', 'give an agent a different model'];

export const S08Evolve: React.FC = () => {
  const {frame, cue, end} = useScene();
  const helix = pop(frame, cue(0), 24) * (1 - ease(frame, cue(1), 30));
  const genome = ease(frame, cue(1) + 10, 30);
  const chips = ease(frame, cue(2), 20);
  const u = frame < cue(3) ? 0 : Math.min(7, ((frame - cue(3)) / Math.max(1, end(3) + 40 - cue(3))) * 7);
  const active = frame >= cue(3) ? Math.min(6, Math.floor(u)) : -1;
  const chef = ease(frame, cue(4), 20);
  const cam = orbit(Math.sin(frame * 0.005) * 0.35, 12.5, 1.8, [0, 0.1, 0]);
  return (
    <AbsoluteFill>
      <Stage cam={cam}>
        <Helix frame={frame} s={helix} />
        <group scale={[genome, genome, genome]} position={[0, -0.2, 0]}>
          <MorphNet u={u} frame={frame} />
        </group>
      </Stage>
      <Card x={110} y={210} w={520} t={hold(frame, cue(1) + 60, cue(2) + 10, 16)} accent={C.specialist}>
        <Small color={C.specialist} size={22}>THE GENOME</Small>
        <div style={{fontSize: 30, fontWeight: 600, marginTop: 6}}>neuro-san's own config file</div>
        <div style={{fontSize: 30, fontWeight: 600}}>+ the model each agent runs</div>
        <Small size={24}>so every candidate is a real, runnable team</Small>
      </Card>
      <div style={{position: 'absolute', left: 60, right: 60, top: 165, textAlign: 'center', opacity: chips * (1 - chef)}}>
        {OPS.map((op, i) => (
          <Chip key={op} text={op} color={i === active ? C.coordinator : C.accent} t={i === active ? 1 : active >= 0 ? 0.45 : chips} size={24} />
        ))}
        <div style={{height: 64, marginTop: 10, fontFamily: FONT.display, fontWeight: 700, fontSize: 46, color: C.coordinator}}>
          {active >= 0 ? OPS_SAY[active] : ''}
        </div>
      </div>
      <Headline t={chef} y={-300}>
        <div style={{display: 'flex', gap: 22, alignItems: 'center'}}>
          <Chip text="change one ingredient" color={C.coordinator} size={30} />
          <span style={{fontSize: 40, color: C.dim}}>→</span>
          <Chip text="taste it (measure)" color={C.ok} size={30} />
          <span style={{fontSize: 40, color: C.dim}}>→</span>
          <Chip text="keep the better dish" color={C.accent} size={30} />
        </div>
      </Headline>
    </AbsoluteFill>
  );
};

type Mini = {nodes: V3[]; edges: [number, number][]; dim?: number[]; label: string; bad: boolean};
const BAD: Mini[] = [
  {nodes: [[0, 0.8, 0], [-0.8, -0.5, 0], [0.8, -0.5, 0], [1.9, 0.3, 0]], edges: [[0, 1], [0, 2]], dim: [3], label: 'unreachable agent', bad: true},
  {nodes: [[0, 0.8, 0], [-0.8, -0.5, 0], [0.8, -0.5, 0]], edges: [[0, 1], [1, 2], [2, 0]], label: 'endless loop', bad: true},
  {nodes: [[-0.9, 0, 0], [0, 0, 0], [0.9, 0, 0]], edges: [], label: 'no front man', bad: true},
];
const GOOD: Mini = {nodes: [[0, 0.8, 0], [-0.8, -0.5, 0], [0.8, -0.5, 0]], edges: [[0, 1], [0, 2]], label: 'valid', bad: false};

const MiniNet: React.FC<{m: Mini; pos: V3; s: number; frame: number; burst: number}> = ({m, pos, s, frame, burst}) => {
  if (burst >= 1) return null;
  return (
    <group position={pos}>
      {burst > 0
        ? Array.from({length: 14}).map((_, i) => {
            const a = random(`frag-${m.label}-${i}`) * Math.PI * 2;
            const e = random(`frag-e-${m.label}-${i}`) * 2 - 1;
            const d = burst * 3;
            return (
              <mesh key={i} position={[Math.cos(a) * d, e * d, Math.sin(a) * d]} scale={[1 - burst, 1 - burst, 1 - burst]}>
                <tetrahedronGeometry args={[0.18]} />
                <meshBasicMaterial color={C.danger} transparent opacity={1 - burst} />
              </mesh>
            );
          })
        : null}
      <group scale={[0.95 * s * (1 - burst), 0.95 * s * (1 - burst), 0.95 * s * (1 - burst)]}>
        {m.edges.map(([a, b], i) => (
          <Link key={i} from={m.nodes[a]} to={m.nodes[b]} color={m.bad ? '#ff9aa8' : '#8fb3ff'} />
        ))}
        {m.nodes.map((p, i) => (
          <Agent key={i} pos={p} r={0.32} color={i === 0 && m.label !== 'no front man' ? C.coordinator : C.specialist} dim={m.dim?.includes(i) ? 1 : 0} frame={frame} />
        ))}
      </group>
      <Label text={m.label} pos={[0, -1.25, 0]} height={0.42} opacity={s * (1 - burst)} style={{color: m.bad ? '#ffb3bd' : '#b5f5c9'}} />
    </group>
  );
};

export const S09Gate: React.FC = () => {
  const {frame, cue, end} = useScene();
  const cam = orbit(0.5 + Math.sin(frame * 0.004) * 0.1, 12.5, 2.6, [0, 0.2, 0]);
  const gateX = 0;
  const badStart = (i: number) => split(cue(1) - 20, end(1) + 50, 3, i);
  const verdict = frame >= cue(3) ? 1 : 0;
  const gateColor = interpolateColors(ease(frame, cue(3), 15), [0, 1], [C.danger, C.ok]);
  const scan = frame >= cue(1) - 10 && frame < cue(2) ? 1 : verdict;
  return (
    <AbsoluteFill>
      <Stage cam={cam} floor={-2}>
        {[-1.9, 1.9].map((z) => (
          <mesh key={z} position={[gateX, 0, z]}>
            <boxGeometry args={[0.35, 4.2, 0.35]} />
            <meshStandardMaterial color="#26355f" emissive={gateColor} emissiveIntensity={0.5} />
          </mesh>
        ))}
        <mesh position={[gateX, 2.2, 0]}>
          <boxGeometry args={[0.35, 0.35, 4.15]} />
          <meshStandardMaterial color="#26355f" emissive={gateColor} emissiveIntensity={0.5} />
        </mesh>
        <mesh position={[gateX, 0, 0]} rotation={[0, Math.PI / 2, 0]}>
          <planeGeometry args={[3.5, 4.0]} />
          <meshBasicMaterial color={gateColor} transparent opacity={0.12 + 0.12 * scan} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
        <Label text="VALIDITY GATE" pos={[gateX, 2.85, 0]} height={0.36} style={{color: '#e9eefb', weight: 700, font: 'Space Grotesk'}} />
        {BAD.map((m, i) => {
          const appear = ease(frame, split(cue(0), end(0), 3, i), 18);
          const go = ease(frame, badStart(i), 40);
          const home: V3 = [-6.2, 1.6 - i * 2.2, 0];
          const pos = lerp3(home, [gateX - 0.6, 0, 0], go);
          const burst = Math.max(0, Math.min(1, (frame - badStart(i) - 40) / 22));
          return <MiniNet key={m.label} m={m} pos={pos} s={appear} frame={frame} burst={burst} />;
        })}
        {[0, 1, 2].map((i) => {
          const start = split(cue(3), end(3) + 20, 3, i);
          const t = ease(frame, start, 70);
          if (frame < start) return null;
          const pos = lerp3([-7, 0, 0], [6.4, 0.3 + i * 0.05, -1.2 + i * 1.2], t);
          return <MiniNet key={i} m={GOOD} pos={pos} s={1} frame={frame} burst={0} />;
        })}
        {verdict ? (
          <group position={[6.4, -1.6, 0]}>
            <mesh>
              <cylinderGeometry args={[2.4, 2.6, 0.25, 48]} />
              <meshStandardMaterial color="#123a2a" emissive={C.ok} emissiveIntensity={0.25} />
            </mesh>
            <Label text="real test" pos={[0, -0.6, 2.4]} height={0.36} style={{color: '#b5f5c9'}} />
            <Glow color={C.ok} size={6} opacity={0.3} />
          </group>
        ) : null}
      </Stage>
      <Card x={1180} y={200} w={600} t={hold(frame, cue(2), cue(3), 14)} accent={C.coordinator}>
        <Small color={C.coordinator} size={22}>LIKE A BUILDING INSPECTOR</Small>
        <div style={{fontSize: 34, fontWeight: 600, marginTop: 6}}>A house with no door is rejected.</div>
        <Small size={26}>Not patched. Thrown away.</Small>
      </Card>
    </AbsoluteFill>
  );
};

