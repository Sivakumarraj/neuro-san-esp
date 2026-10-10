import * as THREE from 'three';

const cache = new Map<string, THREE.Texture>();

const remember = (key: string, make: () => THREE.Texture) => {
  const hit = cache.get(key);
  if (hit) return hit;
  const made = make();
  cache.set(key, made);
  return made;
};

/** A soft radial glow, drawn once and shared by every node. */
export const glowTexture = () =>
  remember('glow', () => {
    const size = 256;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d')!;
    const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(0.2, 'rgba(255,255,255,0.55)');
    g.addColorStop(0.5, 'rgba(255,255,255,0.12)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  });

export type TextStyle = {
  size?: number;
  weight?: number;
  color?: string;
  font?: string;
  bg?: string;
  border?: string;
  padX?: number;
  padY?: number;
  radius?: number;
};

/** Text drawn onto a canvas, sized to fit; returns the texture and its aspect. */
export const textTexture = (text: string, style: TextStyle = {}) => {
  const {
    size = 64,
    weight = 600,
    color = '#eef3ff',
    font = 'Inter',
    bg,
    border,
    padX = bg ? size * 0.55 : size * 0.15,
    padY = bg ? size * 0.32 : size * 0.12,
    radius = size * 0.4,
  } = style;
  const key = JSON.stringify([text, size, weight, color, font, bg, border, padX, padY, radius]);
  const tex = remember(key, () => {
    const lines = text.split('\n');
    const measure = document.createElement('canvas').getContext('2d')!;
    measure.font = `${weight} ${size}px "${font}"`;
    const width = Math.ceil(Math.max(...lines.map((l) => measure.measureText(l).width)) + padX * 2);
    const lineH = size * 1.22;
    const height = Math.ceil(lineH * lines.length + padY * 2);
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d')!;
    if (bg) {
      ctx.fillStyle = bg;
      ctx.beginPath();
      ctx.roundRect(2, 2, width - 4, height - 4, radius);
      ctx.fill();
      if (border) {
        ctx.strokeStyle = border;
        ctx.lineWidth = Math.max(2, size * 0.06);
        ctx.stroke();
      }
    }
    ctx.font = `${weight} ${size}px "${font}"`;
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    lines.forEach((l, i) => ctx.fillText(l, width / 2, padY + lineH * (i + 0.5)));
    const t = new THREE.CanvasTexture(canvas);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 4;
    t.userData = {aspect: width / height};
    return t;
  });
  return {tex, aspect: tex.userData.aspect as number};
};

export type DocSpec = {
  head: string;
  kind: string;
  rows: [string, string][];
  accent: string;
  highlight?: number[];
};

/** A paper document: a coloured header, then key/value rows. */
export const docTexture = (doc: DocSpec) =>
  remember('doc' + JSON.stringify(doc), () => {
    const w = 640;
    const h = 800;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#eef2fc';
    ctx.beginPath();
    ctx.roundRect(0, 0, w, h, 28);
    ctx.fill();
    ctx.fillStyle = doc.accent;
    ctx.beginPath();
    ctx.roundRect(0, 0, w, 150, [28, 28, 0, 0]);
    ctx.fill();
    ctx.fillStyle = '#0d1530';
    ctx.font = '600 30px "Inter"';
    ctx.fillText(doc.kind.toUpperCase(), 40, 58);
    ctx.font = '700 58px "JetBrains Mono"';
    ctx.fillText(doc.head, 40, 122);
    let y = 225;
    doc.rows.forEach(([k, v], i) => {
      const lit = doc.highlight?.includes(i);
      if (lit) {
        ctx.fillStyle = 'rgba(255, 199, 71, 0.45)';
        ctx.beginPath();
        ctx.roundRect(24, y - 50, w - 48, 92, 14);
        ctx.fill();
      }
      ctx.fillStyle = '#5a6788';
      ctx.font = '500 26px "Inter"';
      ctx.fillText(k, 40, y - 14);
      ctx.fillStyle = '#121a33';
      ctx.font = `${lit ? 700 : 600} 38px "Inter"`;
      ctx.fillText(v, 40, y + 28);
      y += 112;
    });
    // faint body lines below the facts, so it reads as a document
    ctx.fillStyle = '#d3dbef';
    for (let ly = y; ly < h - 40; ly += 34) {
      ctx.fillRect(40, ly, w - 80 - ((ly * 37) % 160), 12);
    }
    const t = new THREE.CanvasTexture(canvas);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 4;
    return t;
  });
