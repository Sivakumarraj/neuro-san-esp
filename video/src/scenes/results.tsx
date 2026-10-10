import React from 'react';
import {AbsoluteFill, random} from 'remotion';
import * as THREE from 'three';
import {ease, hold, lerp3, pop} from '../anim';
import {useScene} from '../scene';
import {Agent, Glow, Label, Plaque, Sheet, V3} from '../three/primitives';
import {Network, NodeSpec} from '../three/network';
import {orbit, Stage} from '../three/Stage';
import {Big, Card, Chip, Headline, Small} from '../overlay/ui';
import {C, FONT} from '../theme';
import {designerEdges} from './intro';

const Trophy: React.FC<{pos: V3; s: number; frame: number}> = ({pos, s, frame}) =>
  s <= 0.001 ? null : (
    <group position={pos} scale={[s, s, s]} rotation={[0, frame * 0.02, 0]}>
      <mesh position={[0, 0.55, 0]}>
        <cylinderGeometry args={[0.55, 0.22, 0.9, 32, 1, true]} />
        <meshStandardMaterial color="#ffcf4a" emissive="#b8860b" emissiveIntensity={0.5} metalness={0.8} roughness={0.2} side={THREE.DoubleSide} />
      </mesh>
      {[-1, 1].map((k) => (
        <mesh key={k} position={[k * 0.58, 0.62, 0]} rotation={[0, 0, Math.PI / 2]}>
          <torusGeometry args={[0.2, 0.05, 12, 32, Math.PI]} />
          <meshStandardMaterial color="#ffcf4a" metalness={0.8} roughness={0.2} />
        </mesh>
      ))}
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[0.08, 0.08, 0.3, 16]} />
        <meshStandardMaterial color="#ffcf4a" metalness={0.8} roughness={0.2} />
      </mesh>
      <mesh position={[0, -0.22, 0]}>
        <boxGeometry args={[0.7, 0.16, 0.5]} />
        <meshStandardMaterial color="#3a2a10" metalness={0.3} />
      </mesh>
      <Glow pos={[0, 0.5, 0]} color="#ffcf4a" size={3} opacity={0.5} />
    </group>
  );

export const S13Holdout: React.FC = () => {
  const {frame, cue} = useScene();
  const deck = hold(frame, cue(0), cue(2), 16);
  const split = ease(frame, cue(2), 40);
  const trophy = pop(frame, cue(3), 24);
  const cross = ease(frame, cue(4), 50);
  const camA = orbit(0.1, 11, 2.5, [0, 0.3, 0]);
  const camB = orbit(0.15 + Math.sin(frame * 0.004) * 0.1, 15, 7, [0, 0, 0]);
  const cam = {pos: lerp3(camA.pos, camB.pos, split), target: lerp3(camA.target!, camB.target!, split)};
  const trophyPos = lerp3([-3.4, 0.4, 0], [3.4, 0.4, 0], cross);
  return (
    <AbsoluteFill>
      <Stage cam={cam} floor={-0.05} floorOpacity={0.2}>
        {Array.from({length: 18}).map((_, i) => (
          <Sheet key={i} pos={[(random(`dk-x-${i}`) - 0.5) * 0.2, -0.6 + i * 0.07, (random(`dk-z-${i}`) - 0.5) * 0.2]} rot={[-Math.PI / 2, 0, random(`dk-r-${i}`) * 0.3]} scale={4 * deck} opacity={deck} />
        ))}
        <Trophy pos={[0, 1.4, 0]} s={deck * pop(frame, cue(0) + 60, 24)} frame={frame} />
        {split > 0.001 ? (
          <group scale={[split, split, split]}>
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[-3.4, 0, 0]}>
              <planeGeometry args={[6.4, 6.4]} />
              <meshStandardMaterial color="#0e2a5a" emissive={C.specialist} emissiveIntensity={0.12} transparent opacity={0.9} />
            </mesh>
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[3.4, 0, 0]}>
              <planeGeometry args={[6.4, 6.4]} />
              <meshStandardMaterial color="#3a2a0e" emissive={C.coordinator} emissiveIntensity={0.12} transparent opacity={0.9} />
            </mesh>
            <mesh position={[0, 1.2, 0]}>
              <boxGeometry args={[0.08, 2.4, 6.4]} />
              <meshBasicMaterial color="#cfe0ff" transparent opacity={0.55} />
            </mesh>
            <Glow pos={[0, 1.2, 0]} color="#cfe0ff" size={5} opacity={0.3} />
            {Array.from({length: 30}).map((_, i) => (
              <Sheet key={`s${i}`} pos={[-3.4 + (random(`sx-${i}`) - 0.5) * 5, 0.15 + random(`sy-${i}`) * 0.6, (random(`sz-${i}`) - 0.5) * 5]} rot={[-1.2, random(`sr-${i}`) * 3, 0]} color="#cfe8ff" scale={0.9} />
            ))}
            {Array.from({length: 30}).map((_, i) => (
              <Sheet key={`j${i}`} pos={[3.4 + (random(`jx-${i}`) - 0.5) * 5, 0.15 + random(`jy-${i}`) * 0.6, (random(`jz-${i}`) - 0.5) * 5]} rot={[-1.2, random(`jr-${i}`) * 3, 0]} color="#ffe6b8" scale={0.9} />
            ))}
            <Plaque text={'SELECT\n60 questions · used to choose'} pos={[-3.4, 0.05, 3.6]} rot={[-Math.PI / 2.6, 0, 0]} height={0.9} style={{size: 64, color: '#cfe8ff', weight: 700}} />
            <Plaque text={'JUDGE\nonly used to grade'} pos={[3.4, 0.05, 3.6]} rot={[-Math.PI / 2.6, 0, 0]} height={0.9} style={{size: 64, color: '#ffe6b8', weight: 700}} />
          </group>
        ) : null}
        <Trophy pos={trophyPos} s={trophy * 1.3} frame={frame} />
      </Stage>
      <Card x={1200} y={210} w={600} t={hold(frame, cue(0) + 70, cue(1) + 20, 14)} accent={C.danger}>
        <Small color={C.danger} size={22}>CHOOSE AND GRADE ON THE SAME QUESTIONS</Small>
        <div style={{fontSize: 32, fontWeight: 600, marginTop: 6}}>…and you fool yourself.</div>
        <Small size={26}>Like memorising last year's paper.</Small>
      </Card>
      <Card x={110} y={210} w={560} t={hold(frame, cue(3) + 20, cue(4) + 10, 14)} accent={C.specialist}>
        <Small color={C.specialist} size={22}>FIRST SEARCH · 17 QUESTIONS</Small>
        <Big size={70}>94%</Big>
        <Small size={24}>on the questions it was chosen on</Small>
      </Card>
      <Card x={1190} y={210} w={620} t={ease(frame, cue(4) + 50, 16) * (1 - ease(frame, cue(5) + 40, 16))} accent={C.coordinator}>
        <Small color={C.coordinator} size={22}>ON FRESH QUESTIONS</Small>
        <Big size={52}>tied the designer</Big>
        <div style={{fontSize: 30, fontWeight: 600, marginTop: 8, color: C.danger}}>at 2.2× the cost per question</div>
        <Small size={24}>the gain came from a stronger coordinator model, not a better structure</Small>
      </Card>
      <Headline t={ease(frame, cue(5), 18)} y={-300}>
        <Chip text="said so openly → searched again" color={C.ok} size={34} />
      </Headline>
    </AbsoluteFill>
  );
};

const WINNER: NodeSpec[] = [
  {id: 'co', pos: [0, 1.4, 0], color: C.coordinator, label: 'Coordinator', sub: 'nano'},
  {id: 'r', pos: [0, -0.4, 0], color: C.researcher, label: 'Researcher', sub: 'mini', strong: 1},
  {id: 't', pos: [0, -2.0, 0], tool: true},
];
const DESIGNER: NodeSpec[] = [
  {id: 'co', pos: [0, 1.4, 0], color: C.coordinator, sub: 'nano'},
  {id: 'a', pos: [-1.6, -0.4, 0], color: C.specialist, sub: 'nano', r: 0.36},
  {id: 'b', pos: [0, -0.6, 0.4], color: C.specialist, sub: 'nano', r: 0.36},
  {id: 'c', pos: [1.6, -0.4, 0], color: C.specialist, sub: 'nano', r: 0.36},
  {id: 'ta', pos: [-1.6, -2.0, 0], tool: true},
  {id: 'tb', pos: [0, -2.2, 0.4], tool: true},
  {id: 'tc', pos: [1.6, -2.0, 0], tool: true},
];

const Bar: React.FC<{pos: V3; h: number; color: string; label: string; t: number}> = ({pos, h, color, label, t}) =>
  t <= 0.001 ? null : (
    <group position={pos}>
      <mesh position={[0, (h * t) / 2, 0]}>
        <boxGeometry args={[1.0, Math.max(0.01, h * t), 1.0]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.45} transparent opacity={0.92} />
      </mesh>
      <Glow pos={[0, h * t, 0]} color={color} size={2} opacity={0.5 * t} />
      <Label text={label} pos={[0, h * t + 0.5, 0]} height={0.55} opacity={t} style={{weight: 700, font: 'Space Grotesk'}} />
    </group>
  );

export const S14Result: React.FC = () => {
  const {frame, cue, end} = useScene();
  const teams = ease(frame, cue(1), 30);
  const promote = ease(frame, cue(1) + (end(1) - cue(1)) * 0.6, 30);
  const acc = ease(frame, cue(2), 50);
  const tok = ease(frame, cue(3), 24);
  const dots = ease(frame, cue(4), 20);
  const cam = orbit(Math.sin(frame * 0.003) * 0.15, 15.5, 2.0, [0, -0.3, 0]);
  const flow = ((frame - cue(1)) % 60) / 60;
  return (
    <AbsoluteFill>
      <Stage cam={cam} floor={-2.9}>
        <group position={[-4.8, 0, 0]} scale={[teams, teams, teams]}>
          <Network nodes={WINNER.map((n) => (n.id === 'r' ? {...n, strong: promote, color: promote > 0.5 ? C.strong : C.researcher} : n))} edges={[['co', 'r'], ['r', 't']]} frame={frame} pulses={[{edge: ['r', 't'], t: flow, color: C.strong}]} />
          <Label text="2cc4 · 2 agents" pos={[0, -3.0, 0]} height={0.46} style={{color: '#b5f5d8', weight: 700}} />
        </group>
        <group position={[4.8, 0, 0]} scale={[teams, teams, teams]}>
          <Network nodes={DESIGNER} edges={designerEdges} frame={frame} />
          <Label text="designer's shape · 4 agents" pos={[0, -3.0, 0]} height={0.46} style={{color: '#bfeaff', weight: 700}} />
        </group>
        <Bar pos={[-1.1, -2.9, 0]} h={0.945 * 4.6} color={C.researcher} label="94.5%" t={acc * (1 - dots * 0.6)} />
        <Bar pos={[1.1, -2.9, 0]} h={0.84 * 4.6} color={C.specialist} label="84.0%" t={acc * (1 - dots * 0.6)} />
      </Stage>
      <Card x={690} y={170} w={540} t={hold(frame, cue(1) + 140, cue(2), 14)} accent={C.strong}>
        <Small color={C.strong} size={22}>ONE AGENT PROMOTED</Small>
        <div style={{fontSize: 30, fontWeight: 600, marginTop: 6}}>Researcher → gpt-5.4-mini</div>
        <Small size={24}>the agent that reads the documents; everyone else on gpt-5.4-nano</Small>
      </Card>
      <div style={{position: 'absolute', left: 0, right: 0, top: 168, textAlign: 'center', opacity: hold(frame, cue(2) + 20, cue(3), 14)}}>
        <Small size={26} color={C.dim}>100 judge questions · asked twice · never used to choose</Small>
      </div>
      <Card x={660} y={180} w={600} t={hold(frame, cue(3), cue(4), 14)} accent={C.tool} center>
        <Small color={C.tool} size={22}>TOKENS PER QUESTION</Small>
        <Big size={58}>6,500 <span style={{color: C.dim, fontSize: 36}}>vs</span> 10,306</Big>
        <div style={{fontSize: 30, fontWeight: 700, color: C.ok, marginTop: 6}}>37% fewer</div>
      </Card>
      {dots > 0.001 ? (
        <div style={{position: 'absolute', left: 0, right: 0, top: 180, display: 'flex', flexDirection: 'column', alignItems: 'center', opacity: dots}}>
          <div style={{display: 'flex', gap: 12, padding: '18px 26px', borderRadius: 20, background: 'rgba(8,14,36,0.82)', border: '1.5px solid rgba(125,211,252,0.3)'}}>
            {Array.from({length: 19}).map((_, i) => {
              const t = ease(frame, cue(4) + 10 + i * 5, 10);
              const good = i < 17;
              return <div key={i} style={{width: 40, height: 40, borderRadius: 20, background: good ? C.ok : C.danger, opacity: t, transform: `scale(${0.5 + 0.5 * t})`, boxShadow: `0 0 18px ${good ? C.ok : C.danger}`}} />;
            })}
          </div>
          <div style={{display: 'flex', gap: 40, marginTop: 14, fontFamily: FONT.body, fontSize: 28}}>
            <span style={{color: C.ok}}>● 17 questions: 2cc4 right more often</span>
            <span style={{color: C.danger}}>● 2: the designer</span>
          </div>
          <div style={{marginTop: 14, opacity: ease(frame, cue(4) + 140, 16), fontFamily: FONT.display, fontWeight: 700, fontSize: 52, color: C.ink}}>
            sign test p = 0.0007 <span style={{fontSize: 30, color: C.dim}}>≈ 7 in 10,000 by luck</span>
          </div>
        </div>
      ) : null}
    </AbsoluteFill>
  );
};

const PANELS = [
  {title: 'COST', color: '#ffcf4a', body: '$0.00537 vs $0.00314 per question  (+71%)\n$0.00568 vs $0.00374 per correct answer  (+52%)\nthe stronger model costs more per token'},
  {title: 'MODEL OR STRUCTURE?', color: C.specialist, body: "the designer's team was never measured\nwith its specialists promoted,\nso the two effects are not separated"},
  {title: 'THE BENCHMARK CHANGED', color: C.tool, body: 'search now returns 10 documents, not 3\n(designer: 69% → 84% on the same questions)\nnumbers at 3 and 10 are never compared'},
  {title: 'WHERE 2cc4 CAME FROM', color: C.ok, body: 'the hand-written flat_pair seed,\nwith its Arithmetic agent removed:\none mutation · Predictor not used'},
];

export const S15FinePrint: React.FC = () => {
  const {frame, cue} = useScene();
  const show = ease(frame, cue(0), 30);
  const at = [cue(1), cue(2), cue(3), cue(4)];
  let turn = 0;
  at.forEach((t, i) => {
    if (i > 0) turn += ease(frame, t - 10, 30);
  });
  const finale = ease(frame, cue(5), 30);
  const cam = orbit(Math.sin(frame * 0.004) * 0.08, 10.5, 1.2, [0, 0.5, 0]);
  const R = 3.4;
  return (
    <AbsoluteFill>
      <Stage cam={cam} floor={-2.5}>
        <group rotation={[0, -turn * (Math.PI / 2), 0]} scale={[show * (1 - finale), show * (1 - finale), show * (1 - finale)]} position={[0, 0.6, -R]}>
          {PANELS.map((p, i) => {
            const a = (i * Math.PI) / 2;
            const lit = ease(frame, at[i] - 10, 20);
            const facing = Math.cos(a - turn * (Math.PI / 2));
            if (facing < 0.25) return null;
            return (
              <group key={p.title} position={[Math.sin(a) * R, 0, Math.cos(a) * R]} rotation={[0, a, 0]}>
                <mesh position={[0, 0, -0.05]}>
                  <planeGeometry args={[5.75, 3.45]} />
                  <meshBasicMaterial color={p.color} transparent opacity={0.35 + 0.4 * lit} side={THREE.DoubleSide} />
                </mesh>
                <mesh position={[0, 0, -0.03]}>
                  <planeGeometry args={[5.6, 3.3]} />
                  <meshBasicMaterial color="#0e1838" side={THREE.DoubleSide} />
                </mesh>
                <Plaque text={p.title} pos={[0, 1.05, 0]} height={0.42} style={{color: p.color, weight: 700, font: 'Space Grotesk'}} />
                <Plaque text={p.body} pos={[0, -0.35, 0]} height={0.95} style={{size: 60, weight: 500, color: '#e6ecff'}} />
              </group>
            );
          })}
        </group>
        {finale > 0.001 ? (
          <group scale={[finale, finale, finale]}>
            <group position={[-3.2, 0.3, 0]} scale={[0.75, 0.75, 0.75]}>
              <Network nodes={WINNER.map((n) => ({...n, color: n.id === 'r' ? C.strong : n.color}))} edges={[['co', 'r'], ['r', 't']]} frame={frame} />
            </group>
            <Label text=">" pos={[0, 0.3, 0]} height={1.0} style={{color: C.ok, weight: 700, font: 'Space Grotesk'}} />
            <group position={[3.2, 0.3, 0]} scale={[0.75, 0.75, 0.75]}>
              <Network nodes={DESIGNER} edges={designerEdges} frame={frame} />
            </group>
          </group>
        ) : null}
      </Stage>
      <Headline t={finale} y={-310}>
        <Card x={0} y={0} w={1300} t={1} accent={C.ok} center inline>
          <Small color={C.ok} size={22}>ON THIS BENCHMARK, SO FAR</Small>
          <div style={{fontSize: 34, fontWeight: 600, marginTop: 6}}>a small team with the strong model where the hard reading happens</div>
          <div style={{fontSize: 34, fontWeight: 600}}>beat a bigger team of cheap specialists</div>
        </Card>
      </Headline>
    </AbsoluteFill>
  );
};

