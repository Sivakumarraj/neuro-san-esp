import React from 'react';
import {Agent, Link, Pulse, Tool, V3} from './primitives';
import {C} from '../theme';

export type NodeSpec = {
  id: string;
  pos: V3;
  color?: string;
  label?: string;
  sub?: string;
  tool?: boolean;
  strong?: number;
  r?: number;
  s?: number;
  dim?: number;
};

export type EdgeSpec = [string, string];

/**
 * A network that builds itself: node i appears at build ≈ i / n, each edge once
 * both of its ends are up. `pulses` send messages along edges (t in 0..1).
 */
export const Network: React.FC<{
  nodes: NodeSpec[];
  edges: EdgeSpec[];
  build?: number;
  frame: number;
  labels?: number;
  edgeColor?: string;
  edgeOpacity?: number;
  pulses?: {edge: EdgeSpec; t: number; color?: string; back?: boolean}[];
  offset?: V3;
  scale?: number;
}> = ({nodes, edges, build = 1, frame, labels = 1, edgeColor = '#7fa6ff', edgeOpacity = 0.7, pulses = [], offset = [0, 0, 0], scale = 1}) => {
  const n = nodes.length;
  const appear = (i: number) => Math.max(0, Math.min(1, (build * (n + 1) - i) * 1.2));
  const index = new Map(nodes.map((nd, i) => [nd.id, i]));
  const at = (id: string) => nodes[index.get(id)!].pos;
  return (
    <group position={offset} scale={[scale, scale, scale]}>
      {edges.map(([a, b]) => {
        const ia = index.get(a);
        const ib = index.get(b);
        if (ia === undefined || ib === undefined) return null;
        const p = Math.min(appear(ia), appear(ib)) * Math.min(nodes[ia].s ?? 1, nodes[ib].s ?? 1);
        return <Link key={`${a}-${b}`} from={at(a)} to={at(b)} progress={p} color={edgeColor} opacity={edgeOpacity * p} />;
      })}
      {nodes.map((nd, i) => {
        const s = appear(i) * (nd.s ?? 1);
        return nd.tool ? (
          <Tool key={nd.id} pos={nd.pos} s={s} label={nd.label} frame={frame} />
        ) : (
          <Agent
            key={nd.id}
            pos={nd.pos}
            color={nd.color ?? C.specialist}
            r={nd.r}
            s={s}
            label={nd.label}
            sub={nd.sub}
            labelOpacity={labels}
            strong={nd.strong}
            frame={frame}
            dim={nd.dim}
          />
        );
      })}
      {pulses.map((p, i) => {
        const [a, b] = p.back ? [p.edge[1], p.edge[0]] : p.edge;
        if (!index.has(a) || !index.has(b)) return null;
        return <Pulse key={i} from={at(a)} to={at(b)} t={p.t} color={p.color ?? '#ffffff'} />;
      })}
    </group>
  );
};
