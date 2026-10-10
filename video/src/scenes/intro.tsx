import React from 'react';
import {AbsoluteFill} from 'remotion';
import {ease, hold, lerp, pop} from '../anim';
import {useScene} from '../scene';
import {Agent, Doc, Glow, Label, Link, Pulse, Sheet, Tool, V3} from '../three/primitives';
import {Network, NodeSpec, EdgeSpec} from '../three/network';
import {orbit, Stage} from '../three/Stage';
import {Big, Card, Chip, Headline, Small} from '../overlay/ui';
import {C, FONT} from '../theme';

/** The four-agent shape neuro-san's designer builds, with each specialist's search tool. */
export const designerNodes = (labels = true): NodeSpec[] => [
  {id: 'co', pos: [0, 1.5, 0], color: C.coordinator, label: labels ? 'Coordinator' : undefined},
  {id: 'a', pos: [-2.6, -0.4, 0], color: C.specialist, label: labels ? 'Depots' : undefined},
  {id: 'b', pos: [0, -0.7, 0.5], color: C.specialist, label: labels ? 'Contracts' : undefined},
  {id: 'c', pos: [2.6, -0.4, 0], color: C.specialist, label: labels ? 'Incidents' : undefined},
  {id: 'ta', pos: [-2.6, -2.1, 0], tool: true},
  {id: 'tb', pos: [0, -2.4, 0.5], tool: true},
  {id: 'tc', pos: [2.6, -2.1, 0], tool: true},
];
export const designerEdges: EdgeSpec[] = [
  ['co', 'a'], ['co', 'b'], ['co', 'c'], ['a', 'ta'], ['b', 'tb'], ['c', 'tc'],
];

const typed = (text: string, t: number) => text.slice(0, Math.round(text.length * Math.max(0, Math.min(1, t))));

const PromptBar: React.FC<{text: string; t: number; opacity: number; y?: number}> = ({text, t, opacity, y = 210}) =>
  opacity <= 0.001 ? null : (
    <div
      style={{
        position: 'absolute', left: '50%', top: y, transform: 'translateX(-50%)', opacity,
        padding: '22px 34px', borderRadius: 20, minWidth: 980,
        background: 'rgba(12,20,48,0.85)', border: '1.5px solid rgba(125,211,252,0.45)',
        boxShadow: '0 0 60px rgba(56,208,255,0.18)', fontFamily: FONT.body, fontSize: 36, color: C.ink,
      }}
    >
      <span style={{color: C.accent, marginRight: 16}}>›</span>
      {typed(text, t)}
      <span style={{opacity: Math.floor(t * 40) % 2 ? 1 : 0.2, color: C.accent}}>▍</span>
    </div>
  );

export const S01Hook: React.FC = () => {
  const {frame, cue, end, dur} = useScene();
  const sentence = 'Build me a team that answers questions about my company.';
  const typing = ease(frame, cue(0) + 6, Math.max(20, (end(0) - cue(0)) * 0.55));
  const build = ease(frame, end(0) - 50, 55);
  const promptOut = 1 - ease(frame, cue(1) + 30, 20);
  const pullBack = ease(frame, cue(3), 50);
  const cam = orbit(Math.sin(frame * 0.006) * 0.5, lerp(10.5, 15, pullBack), lerp(1.2, 2.5, pullBack), [0, 0, 0]);
  const flow = (k: number) => ((frame - end(0) + k * 18) % 70) / 70;
  const alt = ease(frame, cue(2), 24) * (1 - ease(frame, cue(3), 24));
  const q = hold(frame, cue(1), cue(2) + 10, 12);
  const title = ease(frame, cue(3) + 20, 30);
  return (
    <AbsoluteFill>
      <Stage cam={cam}>
        <group position={[0, lerp(0, 2.4, pullBack), lerp(0, -3, pullBack)]} scale={[lerp(1, 0.75, pullBack), lerp(1, 0.75, pullBack), lerp(1, 0.75, pullBack)]}>
        <Network
          nodes={designerNodes(false)}
          edges={designerEdges}
          build={build}
          frame={frame}
          pulses={build >= 1 ? [
            {edge: ['co', 'a'], t: flow(0), color: C.coordinator},
            {edge: ['co', 'b'], t: flow(1), color: C.coordinator},
            {edge: ['co', 'c'], t: flow(2), color: C.coordinator},
          ] : []}
        />
        </group>
        <group position={[-6.2, 0, -2]} scale={[0.8 * alt, 0.8 * alt, 0.8 * alt]}>
          <Agent pos={[0, 1, 0]} color={C.coordinator} r={0.4} frame={frame} />
          <Agent pos={[0, -0.8, 0]} color={C.researcher} r={0.4} frame={frame} />
          <Link from={[0, 1, 0]} to={[0, -0.8, 0]} />
          <Label text="smaller?" pos={[0, -2.1, 0]} height={0.45} />
        </group>
        <group position={[6.2, 0, -2]} scale={[0.8 * alt, 0.8 * alt, 0.8 * alt]}>
          <Agent pos={[-0.7, 0.6, 0]} color={C.specialist} r={0.35} frame={frame} />
          <Agent pos={[0.7, 0.6, 0]} color={C.specialist} r={0.35} frame={frame} />
          <Agent pos={[0, -0.7, 0]} color={C.coordinator} r={0.35} frame={frame} />
          <Label text="cheaper?" pos={[0, -2.1, 0]} height={0.45} />
        </group>
        <Glow pos={[0, 0, -4]} color="#2856c8" size={22 * (0.6 + 0.4 * build)} opacity={0.35} />
      </Stage>
      <PromptBar text={sentence} t={typing} opacity={ease(frame, cue(0), 12) * promptOut} />
      {q > 0.001 ? (
        <div style={{position: 'absolute', right: 230, top: 250, opacity: q, transform: `scale(${0.8 + 0.2 * pop(frame, cue(1), 20)})`, fontFamily: FONT.display, fontWeight: 700, fontSize: 220, color: C.coordinator, textShadow: '0 0 60px rgba(255,181,71,0.6)'}}>?</div>
      ) : null}
      <Headline t={hold(frame, cue(1) + 10, cue(2) + 5, 12)} y={-330}>
        <Big size={58}>Is it a <span style={{color: C.coordinator}}>good</span> team?</Big>
      </Headline>
      <Headline t={title * (1 - ease(frame, dur - 30, 15))} y={170}>
        <Big size={120}>neuro-san-esp</Big>
        <Small size={34} color={C.accent}>Measure a team of AI agents. Then search for a better one.</Small>
      </Headline>
    </AbsoluteFill>
  );
};

export const S02Agent: React.FC = () => {
  const {frame, cue, end} = useScene();
  const agentIn = pop(frame, cue(0), 22);
  const brain = ease(frame, cue(1), 24);
  const instr = ease(frame, cue(2), 24);
  const tool = ease(frame, cue(3), 24);
  const run = (frame - cue(4)) / Math.max(30, end(4) - cue(4));
  const leg = (i: number) => Math.max(0, Math.min(1, run * 4 - i));
  const cam = orbit(-0.25 + frame * 0.0015, 10.5, 1.4, [0, 0.2, 0]);
  const agent: V3 = [0, 0, 0];
  const toolPos: V3 = [3.3, -0.2, 0];
  return (
    <AbsoluteFill>
      <Stage cam={cam}>
        <Agent pos={agent} color={C.specialist} r={0.75} s={agentIn} label="AI agent" frame={frame} />
        {brain > 0.01 ? (
          <group position={[0, 0, 0]} rotation={[frame * 0.01, frame * 0.013, 0]} scale={[brain, brain, brain]}>
            <mesh>
              <icosahedronGeometry args={[1.25, 1]} />
              <meshBasicMaterial color="#d38bff" wireframe transparent opacity={0.55} />
            </mesh>
          </group>
        ) : null}
        <Label text="brain: a language model" pos={[0, 1.95, 0]} height={0.36} opacity={brain} style={{color: '#e6c2ff'}} />
        <Doc
          spec={{
            head: 'Instructions', kind: 'job description', accent: '#7dd3fc',
            rows: [['Role', 'Answer questions about'], ['', 'Meridian Logistics'], ['Rule', 'Every fact from your tools'], ['Style', 'Reply with the answer alone']],
          }}
          pos={[-3.6, 0.2, 0.2]}
          rot={[0, 0.45, 0]}
          scale={1.1 * instr}
          opacity={instr * (1 - ease(frame, cue(4) - 10, 15))}
        />
        <Link from={agent} to={toolPos} progress={tool} color={C.tool} opacity={0.8 * tool} />
        <Tool pos={toolPos} s={pop(frame, cue(3), 20)} label="search tool" frame={frame} />
        {Array.from({length: 7}).map((_, i) => (
          <Sheet key={i} pos={[4.9 + i * 0.05, -0.2 + i * 0.04, -0.9 - i * 0.12]} rot={[0, -0.35, 0]} scale={2.2} opacity={tool} />
        ))}
        <Label text="company files" pos={[5.1, -1.0, -1.0]} height={0.24} opacity={tool} style={{size: 56, weight: 500, color: C.dim}} />
        <Pulse from={[-7, 0, 0]} to={agent} t={leg(0)} color={C.accent} />
        <Pulse from={agent} to={toolPos} t={leg(1)} color={C.tool} />
        <Pulse from={toolPos} to={agent} t={leg(2)} color={C.tool} />
        <Pulse from={agent} to={[7, 0, 0]} t={leg(3)} color={C.ok} />
      </Stage>
      <Card x={110} y={250} w={560} t={ease(frame, cue(4), 12)} accent={C.accent}>
        <Small color={C.accent} size={22}>QUESTION</Small>
        <div style={{fontSize: 34, fontWeight: 600, marginTop: 6}}>Who is the manager of depot D08?</div>
      </Card>
      <Card x={1290} y={250} w={480} t={ease(frame, cue(4) + (end(4) - cue(4)) * 0.75, 14)} accent={C.ok}>
        <Small color={C.ok} size={22}>ANSWER</Small>
        <div style={{fontSize: 40, fontWeight: 700, marginTop: 6}}>J. Vasquez</div>
      </Card>
    </AbsoluteFill>
  );
};

const KITCHEN: NodeSpec[] = [
  {id: 'chef', pos: [0, 1.6, 0], color: C.coordinator},
  {id: 'g', pos: [-3.0, -0.8, 0], color: C.specialist},
  {id: 's', pos: [0, -1.2, 0.6], color: C.specialist},
  {id: 'd', pos: [3.0, -0.8, 0], color: C.specialist},
];

const HOCON = `{
  "llm_config": {"model_name": "gpt-5.4-nano"},
  "tools": [
    {
      "name": "Coordinator",
      "instructions": "You coordinate specialists...",
      "tools": ["ContractSpecialist",
                "DepotSpecialist",
                "IncidentSpecialist"]
    },
    {
      "name": "DepotSpecialist",
      "function": {"description":
        "Look up depots: city, manager..."},
      "tools": ["CorpusSearch"]
    },
    ...
  ]
}`;

export const S03Team: React.FC = () => {
  const {frame, cue, end} = useScene();
  const build = ease(frame, cue(0) + 20, 45);
  const order = (frame - cue(1)) / 40;
  const fan = (frame - cue(1) - 40) / 40;
  const rename = ease(frame, cue(2), 24);
  const shift = ease(frame, cue(3), 30);
  const designer = ease(frame, cue(4), 24);
  const cam = orbit(0.15 * Math.sin(frame * 0.008), 11.5, 1.6, [0, 0, 0]);
  const off: V3 = [lerp(0, -3.2, shift), -0.3, 0];
  const L = (a: string, b: string) => (rename < 0.5 ? a : b);
  const lo = Math.abs(rename - 0.5) * 2;
  return (
    <AbsoluteFill>
      <Stage cam={cam}>
        <group position={off}>
          <Network
            nodes={KITCHEN.map((n) => ({...n, r: n.id === 'chef' ? 0.55 : 0.45}))}
            edges={[['chef', 'g'], ['chef', 's'], ['chef', 'd']]}
            build={build}
            frame={frame}
            pulses={[
              {edge: ['chef', 'g'], t: fan, color: C.coordinator},
              {edge: ['chef', 's'], t: fan - 0.1, color: C.coordinator},
              {edge: ['chef', 'd'], t: fan - 0.2, color: C.coordinator},
            ]}
          />
          <Pulse from={[0, 4.5, 0]} to={[0, 1.6, 0]} t={order} color={C.ink} />
          <Label text={L('Head chef', 'Coordinator · front man')} pos={[0, 2.65, 0]} height={0.38} opacity={lo * build} style={{color: '#ffd9a0'}} />
          <Label text={L('Grill', 'Specialist')} pos={[-3.0, -1.75, 0]} height={0.32} opacity={lo * build} />
          <Label text={L('Sauces', 'Specialist')} pos={[0, -2.2, 0.6]} height={0.32} opacity={lo * build} />
          <Label text={L('Desserts', 'Specialist')} pos={[3.0, -1.75, 0]} height={0.32} opacity={lo * build} />
          <Glow pos={[0, 0, 0]} color={C.coordinator} size={9} opacity={0.25 * designer} />
        </group>
      </Stage>
      {shift > 0.001 ? (
        <div style={{position: 'absolute', right: 90, top: 190, width: 720, opacity: shift * (1 - designer), transform: `translateX(${(1 - shift) * 40}px) perspective(1400px) rotateY(-8deg)`}}>
          <div style={{fontFamily: FONT.body, fontSize: 22, color: C.accent, letterSpacing: 2, marginBottom: 10}}>CONFIG FILE (HOCON) · neuro-san runs it</div>
          <pre style={{margin: 0, padding: '22px 26px', borderRadius: 18, background: 'rgba(6,12,30,0.9)', border: '1.5px solid rgba(125,211,252,0.35)', fontFamily: FONT.mono, fontSize: 21, lineHeight: 1.42, color: '#cfe0ff'}}>
            {HOCON}
          </pre>
        </div>
      ) : null}
      <Card x={1160} y={330} w={640} t={designer * (1 - ease(frame, end(4) + 20, 15))} accent={C.coordinator}>
        <Small color={C.coordinator} size={22}>ONE SENTENCE IN</Small>
        <div style={{fontSize: 30, fontWeight: 600, marginTop: 6}}>“Build me a team for my company.”</div>
        <Small size={24}>→ agent_network_designer → a whole team</Small>
      </Card>
    </AbsoluteFill>
  );
};

const ringNodes = (count: number, radius: number, y: number, color: string, prefix: string): NodeSpec[] =>
  Array.from({length: count}).map((_, i) => {
    const a = (i / count) * Math.PI * 2;
    return {id: `${prefix}${i}`, pos: [Math.sin(a) * radius, y, Math.cos(a) * radius * 0.6] as V3, color, r: 0.3};
  });

export const S04Gap: React.FC = () => {
  const {frame, cue} = useScene();
  const one = 1 - ease(frame, cue(1), 20);
  const two = ease(frame, cue(1), 24) * (1 - ease(frame, cue(4), 20));
  const sizes = ease(frame, cue(2), 24);
  const nine = [{id: 'top', pos: [0, 1.4, 0] as V3, color: C.coordinator, r: 0.42}, ...ringNodes(8, 2.1, -0.6, C.specialist, 'n')];
  const five = [{id: 'top', pos: [0, 1.4, 0] as V3, color: C.coordinator, r: 0.42}, ...ringNodes(4, 1.6, -0.6, C.specialist, 'm')];
  const cam = orbit(Math.sin(frame * 0.005) * 0.3, 13, 2, [0, 0, 0]);
  const bigWords = ease(frame, cue(4), 24);
  return (
    <AbsoluteFill>
      <Stage cam={cam}>
        <group scale={[one, one, one]}>
          <Network nodes={designerNodes(false)} edges={designerEdges} frame={frame} />
          <Glow pos={[0, 0, -1]} color={C.coordinator} size={10} opacity={0.2} />
        </group>
        <group position={[-4.2, 0, 0]} scale={[two, two, two]}>
          <Network nodes={sizes > 0.5 ? nine : designerNodes(false).slice(0, 4)} edges={sizes > 0.5 ? nine.slice(1).map((n) => ['top', n.id] as EdgeSpec) : designerEdges.slice(0, 3)} frame={frame} />
          <Label text={sizes > 0.5 ? '9 agents' : 'team A'} pos={[0, -2.6, 0]} height={0.45} />
        </group>
        <group position={[4.2, 0, 0]} scale={[two, two, two]}>
          <Network nodes={sizes > 0.5 ? five : [{id: 'top', pos: [0, 1.4, 0], color: C.coordinator}, {id: 'x', pos: [0, -0.8, 0], color: C.researcher}]} edges={sizes > 0.5 ? five.slice(1).map((n) => ['top', n.id] as EdgeSpec) : [['top', 'x']]} frame={frame} />
          <Label text={sizes > 0.5 ? '5 agents' : 'team B'} pos={[0, -2.6, 0]} height={0.45} />
        </group>
        <Label text="vs" pos={[0, 0.2, 0]} height={0.6 * two} opacity={two} style={{color: C.coordinator, weight: 700}} />
      </Stage>
      <Card x={1250} y={190} w={560} t={hold(frame, cue(0) + 20, cue(1) + 5, 12)} accent={C.danger}>
        <Small color={C.danger} size={22}>SCORE</Small>
        <Big size={64}>— never measured</Big>
      </Card>
      <Headline t={hold(frame, cue(1) + 15, cue(2), 12)} y={-300}>
        <Big size={56}>Which team is better? <span style={{color: C.coordinator}}>Nobody knows.</span></Big>
      </Headline>
      {sizes > 0.001 ? (
        <div style={{position: 'absolute', left: 0, right: 0, top: 170, textAlign: 'center', opacity: hold(frame, cue(2) + 40, cue(3) + 30, 14)}}>
          <div style={{fontFamily: FONT.body, fontSize: 26, color: C.dim, marginBottom: 6}}>Which model should each agent run?</div>
          <Chip text="gpt-5.4-nano" color={C.specialist} />
          <Chip text="gpt-5.4-mini" color={C.strong} />
          <Chip text="gemini-3.1-flash-lite" color={C.researcher} />
          <Chip text="claude-haiku-4-5" color={C.tool} />
        </div>
      ) : null}
      <Headline t={hold(frame, cue(3) + 30, cue(4), 12)} y={-300}>
        <Big size={50}>A whole department hired. <span style={{color: C.danger}}>No one ever tested.</span></Big>
      </Headline>
      <Headline t={bigWords} y={-20}>
        <div style={{display: 'flex', gap: 60}}>
          <Card x={0} y={0} w={560} t={ease(frame, cue(4) + 10, 20)} accent={C.specialist} center inline>
            <Small color={C.specialist} size={24}>STEP 1</Small>
            <Big size={92}>MEASURE</Big>
            <Small size={26}>score any team</Small>
          </Card>
          <Card x={0} y={0} w={560} t={ease(frame, cue(4) + 40, 20)} accent={C.coordinator} center inline>
            <Small color={C.coordinator} size={24}>STEP 2</Small>
            <Big size={92}>SEARCH</Big>
            <Small size={26}>find a better one</Small>
          </Card>
        </div>
      </Headline>
    </AbsoluteFill>
  );
};
