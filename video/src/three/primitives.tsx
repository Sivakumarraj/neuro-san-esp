import React, {useMemo} from 'react';
import * as THREE from 'three';
import {random} from 'remotion';
import {docTexture, DocSpec, glowTexture, textTexture, TextStyle} from './textures';
import {C} from '../theme';

export type V3 = [number, number, number];

export const Glow: React.FC<{pos?: V3; color: string; size: number; opacity?: number}> = ({
  pos = [0, 0, 0],
  color,
  size,
  opacity = 1,
}) => (
  <sprite position={pos} scale={[size, size, 1]} renderOrder={-1}>
    <spriteMaterial
      map={glowTexture()}
      color={color}
      transparent
      opacity={opacity}
      depthWrite={false}
      blending={THREE.AdditiveBlending}
    />
  </sprite>
);

/** Camera-facing text. `height` is in world units. */
export const Label: React.FC<{
  text: string;
  pos: V3;
  height?: number;
  opacity?: number;
  style?: TextStyle;
}> = ({text, pos, height = 0.32, opacity = 1, style}) => {
  const {tex, aspect} = textTexture(text, {size: 72, ...style});
  if (opacity <= 0.001) return null;
  return (
    <sprite position={pos} scale={[height * aspect, height, 1]} renderOrder={10}>
      <spriteMaterial map={tex} transparent opacity={opacity} depthTest={false} depthWrite={false} />
    </sprite>
  );
};

/** Flat text on a plane that keeps its own orientation (does not face the camera). */
export const Plaque: React.FC<{
  text: string;
  pos: V3;
  rot?: V3;
  height?: number;
  opacity?: number;
  style?: TextStyle;
}> = ({text, pos, rot = [0, 0, 0], height = 0.4, opacity = 1, style}) => {
  const {tex, aspect} = textTexture(text, {size: 72, ...style});
  if (opacity <= 0.001) return null;
  return (
    <mesh position={pos} rotation={rot} renderOrder={5}>
      <planeGeometry args={[height * aspect, height]} />
      <meshBasicMaterial map={tex} transparent opacity={opacity} side={THREE.DoubleSide} depthWrite={false} />
    </mesh>
  );
};

/** An agent: a lit sphere with a glow, an optional "strong model" ring and a label. */
export const Agent: React.FC<{
  pos: V3;
  color: string;
  r?: number;
  s?: number;
  label?: string;
  sub?: string;
  labelOpacity?: number;
  strong?: number;
  frame?: number;
  dim?: number;
}> = ({pos, color, r = 0.42, s = 1, label, sub, labelOpacity = 1, strong = 0, frame = 0, dim = 0}) => {
  if (s <= 0.001) return null;
  const k = 1 - dim * 0.75;
  return (
    <group position={pos}>
      <group scale={[s, s, s]}>
        <mesh>
          <sphereGeometry args={[r, 48, 32]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={0.55 * k}
            roughness={0.35}
            metalness={0.1}
            transparent={dim > 0}
            opacity={1 - dim * 0.6}
          />
        </mesh>
        <Glow color={color} size={r * 5.2} opacity={0.75 * k} />
        {strong > 0.001 ? (
          <group rotation={[Math.PI / 2.4, frame * 0.02, 0]}>
            <mesh scale={[strong, strong, strong]}>
              <torusGeometry args={[r * 1.55, r * 0.09, 16, 96]} />
              <meshStandardMaterial color={C.strong} emissive={C.strong} emissiveIntensity={1.1} />
            </mesh>
          </group>
        ) : null}
      </group>
      {label ? (
        <Label
          text={label}
          pos={[0, -r * s - 0.34, 0]}
          height={0.3}
          opacity={labelOpacity * Math.min(1, s * 1.5) * k}
          style={{size: 64, weight: 600}}
        />
      ) : null}
      {sub ? (
        <Label
          text={sub}
          pos={[0, -r * s - 0.68, 0]}
          height={0.28}
          opacity={labelOpacity * Math.min(1, s * 1.5) * k}
          style={{size: 56, weight: 600, color: '#b9c6e6', font: 'JetBrains Mono'}}
        />
      ) : null}
    </group>
  );
};

/** A tool: a small rounded cube. */
export const Tool: React.FC<{pos: V3; s?: number; label?: string; color?: string; frame?: number}> = ({
  pos,
  s = 1,
  label,
  color = C.tool,
  frame = 0,
}) => {
  if (s <= 0.001) return null;
  return (
    <group position={pos}>
      <group scale={[s, s, s]} rotation={[0.5, frame * 0.015, 0.2]}>
        <mesh>
          <boxGeometry args={[0.42, 0.42, 0.42]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} roughness={0.3} />
        </mesh>
      </group>
      <Glow color={color} size={1.5 * s} opacity={0.55} />
      {label ? (
        <Label text={label} pos={[0, -0.55 * s, 0]} height={0.24} opacity={Math.min(1, s * 1.5)} style={{size: 56, weight: 500, color: '#d9ccff'}} />
      ) : null}
    </group>
  );
};

const UP = new THREE.Vector3(0, 1, 0);

/** A link between two points that grows from `from` as progress goes 0 → 1. */
export const Link: React.FC<{
  from: V3;
  to: V3;
  progress?: number;
  color?: string;
  radius?: number;
  opacity?: number;
}> = ({from, to, progress = 1, color = '#8fb3ff', radius = 0.035, opacity = 0.8}) => {
  const a = new THREE.Vector3(...from);
  const b = new THREE.Vector3(...to);
  const dir = b.clone().sub(a);
  const len = dir.length() * Math.max(0, Math.min(1, progress));
  if (len < 0.001 || opacity <= 0.001) return null;
  const mid = a.clone().add(dir.clone().normalize().multiplyScalar(len / 2));
  const q = new THREE.Quaternion().setFromUnitVectors(UP, dir.clone().normalize());
  return (
    <mesh position={mid} quaternion={q}>
      <cylinderGeometry args={[radius, radius, len, 12, 1]} />
      <meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} />
    </mesh>
  );
};

/** A message travelling along a link; t in 0..1. */
export const Pulse: React.FC<{from: V3; to: V3; t: number; color?: string; size?: number}> = ({
  from,
  to,
  t,
  color = '#ffffff',
  size = 0.12,
}) => {
  if (t <= 0 || t >= 1) return null;
  const p: V3 = [
    from[0] + (to[0] - from[0]) * t,
    from[1] + (to[1] - from[1]) * t,
    from[2] + (to[2] - from[2]) * t,
  ];
  return (
    <group position={p}>
      <mesh>
        <sphereGeometry args={[size, 16, 12]} />
        <meshBasicMaterial color={color} />
      </mesh>
      <Glow color={color} size={size * 9} opacity={0.9} />
    </group>
  );
};

/** A paper document in space. */
export const Doc: React.FC<{
  spec: DocSpec;
  pos: V3;
  rot?: V3;
  scale?: number;
  opacity?: number;
  glow?: number;
}> = ({spec, pos, rot = [0, 0, 0], scale = 1, opacity = 1, glow = 0}) => {
  if (opacity <= 0.001 || scale <= 0.001) return null;
  const tex = docTexture(spec);
  return (
    <group position={pos} rotation={rot} scale={[scale, scale, scale]}>
      {glow > 0.001 ? <Glow pos={[0, 0, -0.05]} color={spec.accent} size={4.2} opacity={0.55 * glow * opacity} /> : null}
      <mesh>
        <planeGeometry args={[1.6, 2.0]} />
        <meshBasicMaterial map={tex} transparent opacity={opacity} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
};

/** A small blank sheet, for crowds of documents. */
export const Sheet: React.FC<{pos: V3; rot?: V3; color?: string; opacity?: number; scale?: number}> = ({
  pos,
  rot = [0, 0, 0],
  color = C.paper,
  opacity = 1,
  scale = 1,
}) =>
  opacity <= 0.001 || scale <= 0.001 ? null : (
    <mesh position={pos} rotation={rot} scale={[scale, scale, scale]}>
      <planeGeometry args={[0.32, 0.4]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.25} transparent opacity={opacity} side={THREE.DoubleSide} />
    </mesh>
  );

/** A field of stars on a large sphere, fixed by seed. */
export const Stars: React.FC<{count?: number; radius?: number; seed?: string; opacity?: number}> = ({
  count = 1400,
  radius = 60,
  seed = 'stars',
  opacity = 0.9,
}) => {
  const geo = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const u = random(`${seed}-u-${i}`) * 2 - 1;
      const th = random(`${seed}-t-${i}`) * Math.PI * 2;
      const rr = radius * (0.6 + 0.4 * random(`${seed}-r-${i}`));
      const s = Math.sqrt(1 - u * u);
      pos[i * 3] = rr * s * Math.cos(th);
      pos[i * 3 + 1] = rr * u;
      pos[i * 3 + 2] = rr * s * Math.sin(th);
      const tint = random(`${seed}-c-${i}`);
      col[i * 3] = 0.7 + 0.3 * tint;
      col[i * 3 + 1] = 0.8 + 0.2 * (1 - tint);
      col[i * 3 + 2] = 1;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    return g;
  }, [count, radius, seed]);
  return (
    <points geometry={geo}>
      <pointsMaterial size={0.16} vertexColors transparent opacity={opacity} sizeAttenuation depthWrite={false} />
    </points>
  );
};

/** A faint grid floor that gives the space depth. */
export const Floor: React.FC<{y?: number; opacity?: number; size?: number}> = ({y = -3, opacity = 0.35, size = 60}) => {
  const grid = useMemo(() => {
    const g = new THREE.GridHelper(size, size, '#2b4a8a', '#16264a');
    (g.material as THREE.Material).transparent = true;
    return g;
  }, [size]);
  (grid.material as THREE.Material).opacity = opacity;
  return <primitive object={grid} position={[0, y, 0]} />;
};
