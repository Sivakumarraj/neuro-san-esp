import React from 'react';
import {ThreeCanvas} from '@remotion/three';
import {useThree} from '@react-three/fiber';
import * as THREE from 'three';
import {useCurrentFrame} from 'remotion';
import {Floor, Stars, V3} from './primitives';
import {C, H, W} from '../theme';

/** Points the camera each frame. Mutating during render keeps it in step with the frame. */
const Cam: React.FC<{pos: V3; target: V3; fov: number}> = ({pos, target, fov}) => {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  camera.position.set(...pos);
  camera.up.set(0, 1, 0);
  camera.lookAt(...target);
  if (camera.fov !== fov) {
    camera.fov = fov;
    camera.updateProjectionMatrix();
  }
  return null;
};

export type CamSpec = {pos: V3; target?: V3; fov?: number};

/** Orbit helper: a camera circling `target` at `radius`, `height` above it. */
export const orbit = (angle: number, radius: number, height: number, target: V3 = [0, 0, 0]): CamSpec => ({
  pos: [target[0] + Math.sin(angle) * radius, target[1] + height, target[2] + Math.cos(angle) * radius],
  target,
});

export const Stage: React.FC<{
  cam: CamSpec;
  floor?: number | false;
  floorOpacity?: number;
  starsOpacity?: number;
  children: React.ReactNode;
}> = ({cam, floor = -3, floorOpacity = 0.3, starsOpacity = 0.9, children}) => {
  const frame = useCurrentFrame();
  return (
    <ThreeCanvas width={W} height={H} gl={{antialias: true}} camera={{fov: 42, near: 0.1, far: 300, position: [0, 0, 12]}}>
      <color attach="background" args={[C.bg]} />
      <fog attach="fog" args={[C.bg, 22, 70]} />
      <ambientLight intensity={0.55} />
      <directionalLight position={[6, 10, 8]} intensity={1.4} />
      <pointLight position={[-8, 4, 6]} intensity={40} color="#6aa8ff" />
      <Cam pos={cam.pos} target={cam.target ?? [0, 0, 0]} fov={cam.fov ?? 42} />
      <group rotation={[0, frame * 0.0006, 0]}>
        <Stars opacity={starsOpacity} />
      </group>
      {floor !== false ? <Floor y={floor} opacity={floorOpacity} /> : null}
      {children}
    </ThreeCanvas>
  );
};
