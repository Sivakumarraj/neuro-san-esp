// The film's 3D scenes. film.html holds the words; timeline.js, written by narration.py,
// says when each scene starts and when each spoken line begins (cue times, s.c[...]).
// window.renderAt(t) draws the frame at t seconds from t alone.
import * as THREE from 'three';
import {EffectComposer} from 'three/addons/postprocessing/EffectComposer.js';
import {RenderPass} from 'three/addons/postprocessing/RenderPass.js';
import {UnrealBloomPass} from 'three/addons/postprocessing/UnrealBloomPass.js';
import {OutputPass} from 'three/addons/postprocessing/OutputPass.js';

const W = 3840, H = 2160;
const C = {cyan: 0x46d2ff, teal: 0x2fe3a5, violet: 0x9b8cff, lilac: 0xc9bfff, gold: 0xffc857, coral: 0xff6f6f,
  blue: 0x5b8cff, grey: 0x6b778c, edge: 0x6f9cff, white: 0xffffff};
const CSS = {cyan: '#46d2ff', teal: '#2fe3a5', violet: '#9b8cff', lilac: '#c9bfff', gold: '#ffc857', coral: '#ff7a7a',
  ink: '#dbe2ec', mute: '#8f9bb0', white: '#f3f6fb'};
const hex = c => '#' + new THREE.Color(c).getHexString();

await Promise.all(["600 128px Inter", "700 128px Inter", "400 80px Inter", "128px 'DejaVu Sans Mono'",
  "700 128px 'Inter Display'"].map(f => document.fonts.load(f)));

const QP = new URLSearchParams(location.search);
// ?dpr=2 draws the same layout at twice the pixels: 7680x4320.
const DPR = Math.max(1, parseFloat(QP.get('dpr') || '1'));
const renderer = new THREE.WebGLRenderer({antialias: false, preserveDrawingBuffer: true});
renderer.setPixelRatio(DPR); renderer.setSize(W, H);
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.0;
document.getElementById('gl').appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x04060c); scene.fog = new THREE.FogExp2(0x04060c, 0.0085);
const camera = new THREE.PerspectiveCamera(40, W / H, 0.1, 800);
scene.add(new THREE.AmbientLight(0x8899bb, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 1.5); key.position.set(8, 14, 10); scene.add(key);
const rim = new THREE.DirectionalLight(0x5b8cff, 1.1); rim.position.set(-10, -4, -8); scene.add(rim);

const rt = new THREE.WebGLRenderTarget(W * DPR, H * DPR, {type: THREE.HalfFloatType, samples: parseInt(QP.get('msaa') ?? '0')});
const composer = new EffectComposer(renderer, rt);
composer.addPass(new RenderPass(scene, camera));
// addPass sizes a pass to the device pixels; the glow is set back to 3840x2160 so it looks,
// and costs, the same at every scale.
if (QP.get('bloom') !== '0') { const b = new UnrealBloomPass(new THREE.Vector2(W / 2, H / 2), 0.62, 0.5, 0.88); composer.addPass(b); b.setSize(W, H); }
composer.addPass(new OutputPass());

// ---------- helpers ----------
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const sm = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const eo = t => { t = clamp(t); return 1 - Math.pow(1 - t, 3); };
const eio = t => { t = clamp(t); return t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
const lerp = (a, b, t) => a + (b - a) * t;
const V = (x, y, z) => new THREE.Vector3(x, y, z);
function mulberry(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const SPH = new THREE.SphereGeometry(1, 48, 32);
const SPH_LO = new THREE.SphereGeometry(1, 14, 10);
const BOX = new THREE.BoxGeometry(1, 1, 1);
const BOX_UP = new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0);
function glow(color, ei = 1.5, o = {}) { return new THREE.MeshStandardMaterial({color, emissive: color, emissiveIntensity: ei, roughness: .35, metalness: .15, ...o}); }
function glass(color, op = .16) { return new THREE.MeshStandardMaterial({color, emissive: color, emissiveIntensity: .22, transparent: true, opacity: op, roughness: .2, metalness: 0, depthWrite: false, side: THREE.DoubleSide}); }
function setOp(obj, a) { obj.traverse(o => { const ms = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : []; for (const m of ms) { if (m.userData.base === undefined) m.userData.base = m.opacity; m.transparent = true; m.opacity = m.userData.base * a; } }); obj.visible = a > 0.002; }
const dotTex = (() => { const c = document.createElement('canvas'); c.width = c.height = 128; const x = c.getContext('2d'); const g = x.createRadialGradient(64, 64, 0, 64, 64, 64); g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(.25, 'rgba(255,255,255,.85)'); g.addColorStop(1, 'rgba(255,255,255,0)'); x.fillStyle = g; x.fillRect(0, 0, 128, 128); return new THREE.CanvasTexture(c); })();

function label(text, {size = .6, color = CSS.ink, weight = 600, sub = null, subColor = CSS.mute, subScale = .62, align = 'center', mono = false, display = false} = {}) {
  const px = 128 * DPR; const fam = mono ? "'DejaVu Sans Mono'" : (display ? "'Inter Display'" : 'Inter');
  const f1 = `${weight} ${px}px ${fam}`; const f2 = `400 ${Math.round(px * subScale)}px ${mono ? fam : 'Inter'}`;
  const c = document.createElement('canvas'); const x = c.getContext('2d');
  x.font = f1; let w = x.measureText(text).width;
  const lines = sub ? (Array.isArray(sub) ? sub : [sub]) : [];
  x.font = f2; for (const l of lines) w = Math.max(w, x.measureText(l).width);
  const pad = 28 * DPR, lh2 = px * subScale * 1.32;
  c.width = Math.ceil(w + pad * 2); c.height = Math.ceil(px * 1.22 + lines.length * lh2 + pad * 2);
  x.textBaseline = 'top'; x.textAlign = align === 'center' ? 'center' : 'left';
  const ax = align === 'center' ? c.width / 2 : pad;
  x.font = f1; x.fillStyle = color; x.fillText(text, ax, pad);
  x.font = f2; x.fillStyle = subColor; lines.forEach((l, i) => x.fillText(l, ax, pad + px * 1.22 + i * lh2));
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
  const m = new THREE.SpriteMaterial({map: tex, transparent: true, depthWrite: false, depthTest: true, toneMapped: false, fog: false});
  const s = new THREE.Sprite(m); const k = size / px; s.scale.set(c.width * k, c.height * k, 1);
  if (align === 'left') s.center.set(0, .5);
  s.renderOrder = 10;
  return s;
}
// A flat panel drawn in 2D: draw(ctx, unit) gets a context scaled so one world unit is `unit` px.
function panel(w, h, draw, {ppu = 150, front = false} = {}) {
  const unit = ppu * DPR; const c = document.createElement('canvas');
  c.width = Math.round(w * unit); c.height = Math.round(h * unit);
  const x = c.getContext('2d'); draw(x, unit, c.width, c.height);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({map: tex, transparent: true, toneMapped: false, depthWrite: false, side: front ? THREE.FrontSide : THREE.DoubleSide, fog: false}));
  m.renderOrder = 5;
  return m;
}
function rrect(x, X, Y, w, h, r, fill, stroke, lw) { x.beginPath(); x.roundRect(X, Y, w, h, r); if (fill) { x.fillStyle = fill; x.fill(); } if (stroke) { x.strokeStyle = stroke; x.lineWidth = lw; x.stroke(); } }
function beam(a, b, r, mat) { const d = new THREE.Vector3().subVectors(b, a); const L = d.length(); const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, L, 12, 1, true), mat); m.position.copy(a).addScaledVector(d, .5); m.quaternion.setFromUnitVectors(V(0, 1, 0), d.clone().normalize()); return m; }
function setBeam(m, a, b) { const d = new THREE.Vector3().subVectors(b, a); const L = Math.max(1e-4, d.length()); m.position.copy(a).addScaledVector(d, .5); m.quaternion.setFromUnitVectors(V(0, 1, 0), d.clone().normalize()); m.scale.set(1, L, 1); }
const UNIT_BEAM = r => new THREE.CylinderGeometry(r, r, 1, 12, 1, true);
function flow(g, a, b, {n = 3, color = C.white, size = .13, speed = .45, phase = 0} = {}) {
  const arr = []; for (let k = 0; k < n; k++) { const s = new THREE.Mesh(SPH_LO, glow(color, 3)); g.add(s); arr.push(s); }
  return (t, alpha = 1, A = a, B = b) => arr.forEach((s, k) => { const u = (((t * speed + k / n + phase) % 1) + 1) % 1; s.position.lerpVectors(A, B, u); s.scale.setScalar(Math.max(1e-4, size * Math.sin(Math.PI * u) * alpha)); });
}
function node(g, p, color, {r = .8, ei = 1.6, ring = true} = {}) {
  const m = new THREE.Mesh(SPH, glow(color, ei * .72)); m.scale.setScalar(r); m.position.copy(p); g.add(m);
  let rg = null; if (ring) { rg = new THREE.Mesh(new THREE.TorusGeometry(1.6, .035, 10, 120), glow(color, 1.1)); rg.scale.setScalar(r); rg.position.copy(p); g.add(rg); }
  return {m, rg, p, r};
}
function spin(nodes, t) { nodes.forEach((n, i) => { if (n.rg) { n.rg.rotation.x = t * .6 + i; n.rg.rotation.y = t * .45 + i * .3; } }); }
function edgesOf(geo, color, op = .8) { return new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({color, transparent: true, opacity: op})); }
function block(g, p, w, h, d, color, title, sub, {size = .5} = {}) {
  const grp = new THREE.Group(); grp.position.copy(p); g.add(grp);
  const geo = new THREE.BoxGeometry(w, h, d); const fm = new THREE.MeshStandardMaterial({color: new THREE.Color(color).multiplyScalar(.28), emissive: color, emissiveIntensity: .08, transparent: true, opacity: .85, roughness: .4});
  const fill = new THREE.Mesh(geo, fm); grp.add(fill);
  const e = edgesOf(geo, color, .95); grp.add(e);
  const L = label(title, {size, sub, subScale: .62, color: '#eef2f8', subColor: '#c3cddc'}); L.material.depthTest = false; L.renderOrder = 30; L.position.set(0, 0, d / 2 + .4); grp.add(L);
  return {grp, fill, e, L};
}
// A bar that grows from a base: returns {m, e, set(k)}.
function bar(g, x, base, w, d, h, color, {ei = .42, z = 0} = {}) {
  const m = new THREE.Mesh(BOX_UP, glow(color, ei)); m.position.set(x, base, z); m.scale.set(w, 1e-4, d); m.visible = false; g.add(m);
  const e = edgesOf(BOX_UP, color, .9); e.position.copy(m.position); e.visible = false; g.add(e);
  return {m, e, h, set(k) { const v = this.h * k + 1e-4; m.visible = e.visible = k > .001; m.scale.y = v; e.scale.set(w, v, d); return v; }};
}
function slab(g, x, color, title, sub, {w = 4.8, h = 7, ry = 0} = {}) {
  const grp = new THREE.Group(); grp.position.set(x, 0, 0); grp.rotation.y = ry; g.add(grp);
  const geo = new THREE.BoxGeometry(w, h, .28); grp.add(new THREE.Mesh(geo, glass(color, .13))); grp.add(edgesOf(geo, color, .95));
  const L = label(title, {size: .58, color: hex(color), weight: 700, sub, subScale: .6, subColor: CSS.ink}); L.position.set(0, h / 2 - 1.15, .3); grp.add(L);
  return grp;
}
function orbit(lt, {c = V(0, 0, 0), r = 30, h = 2, a0 = 0, w = .02, look = null} = {}) { const a = a0 + lt * w; return {p: V(c.x + Math.sin(a) * r, c.y + h, c.z + Math.cos(a) * r), l: look || c}; }
const plate = (g, y, w = 18, d = 4) => { const p = new THREE.Mesh(new THREE.BoxGeometry(w, .12, d), new THREE.MeshStandardMaterial({color: 0x101a2e, emissive: 0x0c1426, roughness: .6})); p.position.y = y - .06; g.add(p); return p; };

// persistent star field
const stars = (() => {
  const R = mulberry(99); const n = 3800; const pos = new Float32Array(n * 3); const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { const u = R() * 2 - 1, th = R() * Math.PI * 2, rr = 70 + R() * 170; const s = Math.sqrt(1 - u * u); pos.set([Math.cos(th) * s * rr, u * rr * .6, Math.sin(th) * s * rr], i * 3); const c = new THREE.Color().setHSL(.58 + R() * .08, .6, .55 + R() * .35); col.set([c.r, c.g, c.b], i * 3); }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const m = new THREE.PointsMaterial({size: .7, map: dotTex, vertexColors: true, transparent: true, opacity: .7, depthWrite: false, blending: THREE.AdditiveBlending, fog: false});
  const p = new THREE.Points(g, m); scene.add(p); return p;
})();

// ---------- scenes ----------
const TL = window.TIMELINE; const TLS = Object.fromEntries(TL.scenes.map(s => [s.id, s]));
const SCENES = [];
function def(id, o) {
  const t = TLS[id]; if (!t) throw new Error(`timeline.js has no scene ${id}`);
  const g = new THREE.Group(); g.visible = false; scene.add(g);
  const s = {id, dur: t.dur, start: t.start, chap: t.chap, c: t.cues, g, shift: .15, shade: 'left', ...o};
  s.el = document.getElementById('s-' + id); if (!s.el) throw new Error(`film.html has no #s-${id}`);
  const at = e => (e.dataset.c !== undefined ? s.c[+e.dataset.c] + (+e.dataset.d || 0) : +(e.dataset.t || 0));
  s.items = [...s.el.querySelectorAll('[data-c],[data-t]')].filter(e => !e.hasAttribute('data-count')).map(e => ({e, t: at(e)}));
  s.counts = [...s.el.querySelectorAll('[data-count]')].map(e => ({e, t: at(e)}));
  SCENES.push(s); s.build && s.build(g, s); return s;
}

// title
def('title', {shift: .17,
  build(g, s) {
    const R = mulberry(7); const N = 18; const cols = [C.teal, C.cyan, C.violet, C.gold, C.cyan, C.teal]; s.nodes = [];
    for (let i = 0; i < N; i++) { const y = 1 - 2 * (i + .5) / N, r = Math.sqrt(1 - y * y), th = i * 2.39996; const p = V(Math.cos(th) * r, y, Math.sin(th) * r).multiplyScalar(7.2 + R() * 1.2); s.nodes.push(node(g, p, cols[i % cols.length], {r: .42 + R() * .38, ei: 1.8})); }
    const em = glow(C.edge, .55, {transparent: true, opacity: 1}); s.em = em; s.flows = []; const seen = new Set();
    s.nodes.forEach((a, i) => { const d = s.nodes.map((b, j) => [a.p.distanceTo(b.p), j]).filter(x => x[1] !== i).sort((x, y) => x[0] - y[0]).slice(0, 2);
      d.forEach(([, j]) => { const k = i < j ? i + '-' + j : j + '-' + i; if (seen.has(k)) return; seen.add(k); g.add(beam(a.p, s.nodes[j].p, .045, em)); s.flows.push(flow(g, a.p, s.nodes[j].p, {n: 2, speed: .35, phase: R()})); }); });
    s.r1 = new THREE.Mesh(new THREE.TorusGeometry(11.5, .03, 8, 200), glow(C.cyan, 1.2)); g.add(s.r1);
    s.r2 = new THREE.Mesh(new THREE.TorusGeometry(13.6, .02, 8, 200), glow(C.violet, 1.0)); g.add(s.r2);
  },
  update(lt, s) {
    s.g.rotation.set(.22, lt * .1, 0);
    s.nodes.forEach((n, i) => { const k = eo((lt - .4 - i * .11) / .9); n.m.scale.setScalar(n.r * k + 1e-4); if (n.rg) n.rg.scale.setScalar(n.r * k + 1e-4); }); spin(s.nodes, lt);
    s.em.opacity = sm(1.6, 3.6, lt); s.flows.forEach(f => f(lt, sm(2.5, 4, lt)));
    s.r1.rotation.set(1.2 + lt * .05, .3, lt * .1); s.r2.rotation.set(1.9, -.4 + lt * .04, lt * .06);
    s.r1.scale.setScalar(eo((lt - .6) / 2)); s.r2.scale.setScalar(eo((lt - 1) / 2.4));
  },
  cam(lt, s) { return {p: V(Math.sin(lt * .05) * 5, 1.5 + lt * .08, lerp(54, 31, eio(lt / (s.dur - .5)))), l: V(0, 0, 0)}; }});

// lens
def('lens', {shift: .2,
  build(g, s) {
    s.beads = []; const names = ['Context', 'Surrogate', 'Held out', 'Cost', 'Production']; const cols = [C.cyan, C.teal, C.violet, C.gold, C.coral];
    const arc = new THREE.Mesh(new THREE.TorusGeometry(8, .03, 8, 200, Math.PI), glow(C.edge, .8)); arc.position.y = -3; g.add(arc);
    for (let i = 0; i < 5; i++) { const a = Math.PI * (1 - (i + .5) / 5); const p = V(Math.cos(a) * 8, Math.sin(a) * 8 - 3, 0);
      const n = node(g, p, cols[i], {r: .85, ei: .12}); const L = label('0' + (i + 1), {size: .8, mono: true, color: CSS.ink}); L.position.copy(p).add(V(0, 1.75, 0)); g.add(L);
      const L2 = label(names[i], {size: .55, color: CSS.mute}); L2.position.copy(p).add(V(0, -1.85, 0)); g.add(L2); s.beads.push({n, L, L2}); }
    s.core = new THREE.Mesh(new THREE.IcosahedronGeometry(1.6, 1), glow(C.blue, .5, {wireframe: true})); s.core.position.y = -1.2; g.add(s.core);
  },
  update(lt, s) {
    s.beads.forEach((b, i) => { const t0 = s.c[1] + 1.4 + i * .8; const k = sm(t0, t0 + .6, lt); b.n.m.material.emissiveIntensity = .12 + 1.25 * k; b.n.rg.material.emissiveIntensity = .1 + 1.0 * k; setOp(b.L, .25 + .75 * k); setOp(b.L2, .3 + .7 * k); });
    spin(s.beads.map(b => b.n), lt); s.core.rotation.set(lt * .2, lt * .3, 0);
  },
  cam(lt) { return orbit(lt, {c: V(0, 1.2, 0), r: 27, h: .5, a0: -.25, w: .03}); }});

// gap
def('gap', {shift: .2,
  build(g, s) {
    const P = [V(0, 4.2, 0), V(-5.2, -1.6, 0), V(0, -2.6, 2), V(5.2, -1.6, 0)]; const names = ['Coordinator', 'DepotSpecialist', 'ContractSpecialist', 'IncidentSpecialist'];
    s.nodes = P.map((p, i) => node(g, p, C.violet, {r: i ? .72 : .95, ei: 1.5}));
    P.forEach((p, i) => { const L = label(names[i], {size: .48, color: CSS.ink}); L.position.copy(p).add(V(0, i ? -1.75 : 1.95, 0)); g.add(L); });
    const em = glow(C.edge, .6); s.flows = []; for (let i = 1; i < 4; i++) { g.add(beam(P[0], P[i], .05, em)); s.flows.push(flow(g, P[0], P[i], {n: 3, speed: .4, phase: i * .2})); }
    s.q = label('fitness = ?', {size: 1.05, color: CSS.coral, mono: true, weight: 400}); s.q.position.set(0, 8.2, 0); g.add(s.q);
    const n = 90; const pos = new Float32Array(n * 3); for (let i = 0; i < n; i++) { const a = i / n * Math.PI * 2; pos.set([Math.cos(a) * 9.5, Math.sin(a) * 9.5 + .8, 0], i * 3); }
    const gg = new THREE.BufferGeometry(); gg.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    s.dash = new THREE.Points(gg, new THREE.PointsMaterial({size: .32, map: dotTex, color: C.coral, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending})); g.add(s.dash);
  },
  update(lt, s) {
    s.g.rotation.y = Math.sin(lt * .3) * .35; spin(s.nodes, lt); s.flows.forEach(f => f(lt));
    setOp(s.q, sm(s.c[1], s.c[1] + 1, lt)); s.dash.material.opacity = .8 * sm(s.c[1], s.c[1] + 1.2, lt); s.dash.rotation.z = lt * .05; s.q.position.y = 8.2 + Math.sin(lt * 1.4) * .12;
  },
  cam(lt) { return orbit(lt, {c: V(0, 1, 0), r: 30, h: 2, a0: .2, w: -.02}); }});

// which
def('which', {shift: .2,
  build(g, s) {
    s.L = slab(g, -3.6, C.lilac, 'ESP INSIDE', ['the network', 'helps a person decide'], {w: 5.6, h: 7.6, ry: .16});
    s.R = slab(g, 3.6, C.teal, 'ESP OUTSIDE', ['the loop', 'designs the network'], {w: 5.6, h: 7.6, ry: -.16});
    // inside: a small network with predictor and prescriptor agents and a person
    const a = s.L; s.ln = [node(a, V(0, .2, .4), C.lilac, {r: .45}), node(a, V(-1.5, -1.9, .4), C.lilac, {r: .36}), node(a, V(1.5, -1.9, .4), C.lilac, {r: .36})];
    const em = glow(C.edge, .6); a.add(beam(V(0, .2, .4), V(-1.5, -1.9, .4), .04, em)); a.add(beam(V(0, .2, .4), V(1.5, -1.9, .4), .04, em));
    [['predictor', -1.5], ['prescriptor', 1.5]].forEach(([t, x]) => { const L = label(t, {size: .34, color: CSS.ink}); L.position.set(x, -2.85, .5); a.add(L); });
    const person = label('you', {size: .42, color: CSS.lilac, weight: 700}); person.position.set(0, 1.25, .5); a.add(person);
    // outside: a ring of four phases around a network
    const b = s.R; const ring = new THREE.Mesh(new THREE.TorusGeometry(2.2, .04, 8, 120), glow(C.teal, 1.0)); ring.position.set(0, -.9, .4); b.add(ring); s.ring = ring;
    s.rn = [node(b, V(0, .1, .4), C.teal, {r: .38}), node(b, V(-.9, -1.6, .4), C.teal, {r: .3}), node(b, V(.9, -1.6, .4), C.teal, {r: .3})];
    b.add(beam(V(0, .1, .4), V(-.9, -1.6, .4), .035, em)); b.add(beam(V(0, .1, .4), V(.9, -1.6, .4), .035, em));
    s.comet = new THREE.Mesh(SPH_LO, glow(C.white, 2)); b.add(s.comet);
  },
  update(lt, s) {
    const kl = eo((lt - s.c[0] - .9) / 1.2), kr = eo((lt - s.c[1] + .2) / 1.2);
    s.L.scale.set(1, kl + 1e-4, 1); s.R.scale.set(1, kr + 1e-4, 1);
    spin([...s.ln, ...s.rn], lt); const a = lt * 1.4; s.comet.position.set(Math.cos(a) * 2.2, -.9 + Math.sin(a) * 2.2, .45); s.comet.scale.setScalar(.16 * kr + 1e-4);
  },
  cam(lt) { return orbit(lt, {c: V(0, 0, 0), r: 27, h: 1.8, a0: -.12, w: .02}); }});

// q1
def('q1', {shift: .23,
  build(g, s) {
    const X = [-6.6, 0, 6.6]; const cols = [C.cyan, C.gold, C.teal]; const T = ['CONTEXT', 'ACTION', 'OUTCOMES'];
    const SUB = [['the question'], ['a network +', 'a model per agent'], ['accuracy · tokens', 'dollars · agents']];
    s.slabs = X.map((x, i) => slab(g, x, cols[i], T[i], SUB[i], {ry: -.18 * (i - 1)}));
    const q = label('?', {size: 2.6, color: CSS.cyan, weight: 700, display: true}); q.position.set(0, -1.2, .35); s.slabs[0].add(q);
    const a = s.slabs[1]; s.mini = [node(a, V(0, .1, .4), C.gold, {r: .42, ei: 1.8}), node(a, V(-1.2, -2, .4), C.gold, {r: .32, ei: 1.6}), node(a, V(1.2, -2, .4), C.gold, {r: .32, ei: 1.6})];
    const em = glow(C.edge, .6); a.add(beam(V(0, .1, .4), V(-1.2, -2, .4), .04, em)); a.add(beam(V(0, .1, .4), V(1.2, -2, .4), .04, em));
    s.bars = []; [2.4, 1.3, 1.0, 1.7].forEach((h, i) => { const b = new THREE.Mesh(BOX_UP, glow([C.teal, C.cyan, C.gold, C.lilac][i], .9)); b.scale.set(.55, h, .3); b.position.set(-1.35 + i * .9, -2.8, .4); b.userData.h = h; s.slabs[2].add(b); s.bars.push(b); });
    const R = mulberry(11); s.stream = []; for (let i = 0; i < 70; i++) { const m = new THREE.Mesh(SPH_LO, glow(i % 5 ? C.cyan : C.white, 2.6)); g.add(m); s.stream.push({m, ph: R(), y: (R() - .5) * 4.2, z: (R() - .5) * 1.6, sp: .05 + R() * .03, sz: .05 + R() * .07}); }
  },
  update(lt, s) {
    s.slabs.forEach((sl, i) => { const k = eo((lt - s.c[i + 1] + .2) / 1.1); sl.scale.set(1, k + 1e-4, 1); }); spin(s.mini, lt);
    s.bars.forEach((b, i) => { b.scale.y = b.userData.h * (.55 + .45 * Math.abs(Math.sin(lt * .7 + i))) * eo((lt - s.c[3]) / 1.4) + 1e-4; });
    s.stream.forEach(p => { const u = ((lt * p.sp + p.ph) % 1); p.m.position.set(-13 + 26 * u, p.y + Math.sin(u * 9 + p.ph * 6) * .4, p.z); p.m.scale.setScalar(p.sz * sm(1.5, 3, lt) * Math.sin(Math.PI * u) + 1e-4); });
  },
  cam(lt) { return orbit(lt, {c: V(0, 0, 0), r: 32, h: 2.6, a0: -.36, w: .03}); }});

// genome
const HOCON = [
  ['{', CSS.ink],
  ['  "llm_config": {"model_name": "gpt-5.4-nano"},', CSS.ink, 'gpt-5.4-nano'],
  ['  "tools": [', CSS.ink],
  ['    {"name": "Coordinator",', CSS.teal],
  ['     "tools": ["Researcher"]},', CSS.ink],
  ['    {"name": "Researcher",', CSS.gold],
  ['     "tools": ["CorpusSearch"],', CSS.ink],
  ['     "llm_config": {"model_name": "gpt-5.4-mini"}},', CSS.gold],
  ['    {"name": "CorpusSearch",', CSS.cyan],
  ['     "class": "esp.eval.corpus_tool.CorpusSearch"}', CSS.ink],
  ['  ]', CSS.ink],
  ['}', CSS.ink],
];
def('genome', {shift: .2,
  build(g, s) {
    s.code = panel(9.4, 6.6, (x, u, W2, H2) => {
      rrect(x, 0, 0, W2, H2, .3 * u, 'rgba(10,16,30,0.94)', 'rgba(120,160,255,0.45)', .03 * u);
      x.font = `600 ${.26 * u}px Inter`; x.fillStyle = CSS.mute; x.textBaseline = 'top'; x.fillText('2cc4 with its Researcher promoted · abridged', .4 * u, .32 * u);
      x.font = `${.27 * u}px 'DejaVu Sans Mono'`;
      HOCON.forEach(([t, col], i) => { x.fillStyle = col; x.fillText(t, .4 * u, (.95 + i * .43) * u); });
    });
    s.code.position.set(-4.4, .4, 0); g.add(s.code);
    s.N = [node(g, V(5.6, 3.2, 0), C.teal, {r: .7}), node(g, V(5.6, -.2, 0), C.gold, {r: .75})];
    s.tool = new THREE.Mesh(new THREE.BoxGeometry(1.1, 1.1, 1.1), glow(C.cyan, .9)); s.tool.position.set(5.6, -3.4, 0); g.add(s.tool);
    const em = glow(C.edge, .65); g.add(beam(V(5.6, 3.2, 0), V(5.6, -.2, 0), .05, em)); g.add(beam(V(5.6, -.2, 0), V(5.6, -3.4, 0), .05, em));
    s.f1 = flow(g, V(5.6, 3.2, 0), V(5.6, -.2, 0), {n: 2, speed: .5}); s.f2 = flow(g, V(5.6, -.2, 0), V(5.6, -3.4, 0), {n: 2, speed: .5, phase: .4});
    [['Coordinator', 3.2, CSS.teal, 'nano'], ['Researcher', -.2, CSS.gold, 'mini'], ['CorpusSearch', -3.4, CSS.cyan, 'coded tool']].forEach(([t, y, c, sub]) => { const L = label(t, {size: .42, color: c, weight: 700, sub, subScale: .7, align: 'left'}); L.position.set(6.9, y, 0); g.add(L); });
    // links from the code lines to the agents
    s.links = new THREE.Group(); g.add(s.links); const lm = new THREE.MeshBasicMaterial({color: C.edge, transparent: true, opacity: .5});
    [[3, 0], [5, 1], [8, 2]].forEach(([line, k]) => { const y = .4 + 6.6 / 2 - (.95 + line * .43 + .14) * 1.0; const a = V(-4.4 + 9.4 / 2 - .2, y, 0); const b = k < 2 ? s.N[k].p.clone().add(V(-.8, 0, 0)) : V(5.0, -3.4, 0); s.links.add(beam(a, b, .025, lm)); });
    s.hash = new THREE.Mesh(new THREE.CylinderGeometry(1.0, 1.0, .18, 6), glow(C.lilac, .8)); s.hash.rotation.x = Math.PI / 2; s.hash.position.set(-4.4, -4.7, 0); g.add(s.hash);
    s.hl = label('hash(structure, instructions, models)', {size: .36, mono: true, weight: 400, color: CSS.lilac}); s.hl.position.set(-4.4, -5.9, 0); g.add(s.hl);
    s.o1 = label('tools[0] = Arithmetic  →  0.00', {size: .38, mono: true, weight: 400, color: CSS.coral, align: 'left'}); s.o1.position.set(2.0, -5.4, 0); g.add(s.o1);
    s.o2 = label('tools[0] = Coordinator →  0.82', {size: .38, mono: true, weight: 400, color: CSS.teal, align: 'left'}); s.o2.position.set(2.0, -6.2, 0); g.add(s.o2);
  },
  update(lt, s) {
    s.code.scale.setScalar(.6 + .4 * eo((lt - s.c[0] + .3) / 1.2)); setOp(s.code, sm(s.c[0] - .3, s.c[0] + .8, lt));
    setOp(s.links, .9 * sm(s.c[0] + 2, s.c[0] + 3.5, lt)); spin(s.N, lt); s.tool.rotation.set(lt * .3, lt * .4, 0);
    s.f1(lt, sm(s.c[0] + 2, s.c[0] + 3, lt)); s.f2(lt, sm(s.c[0] + 2, s.c[0] + 3, lt));
    setOp(s.hash, sm(s.c[1], s.c[1] + .8, lt)); s.hash.rotation.z = lt * .8; setOp(s.hl, sm(s.c[1] + .3, s.c[1] + 1.1, lt));
    setOp(s.o1, sm(s.c[2] + 3.2, s.c[2] + 4, lt)); setOp(s.o2, sm(s.c[2] + 7.0, s.c[2] + 7.8, lt));
  },
  cam(lt) { return orbit(lt, {c: V(.8, -.4, 0), r: 27, h: 1.2, a0: -.08, w: .008}); }});

// world
def('world', {shift: .16,
  build(g, s) {
    const R = mulberry(21); const types = []; for (let i = 0; i < 24; i++) types.push(0); for (let i = 0; i < 40; i++) types.push(1); for (let i = 0; i < 60; i++) types.push(2);
    for (let i = types.length - 1; i > 0; i--) { const j = Math.floor(R() * (i + 1)); [types[i], types[j]] = [types[j], types[i]]; }
    const mats = [glow(C.cyan, .38), glow(C.gold, .32), glow(C.coral, .3)]; s.boxes = []; const cols = 12; let k = 0;
    for (let r = 0; r < 11 && k < 124; r++) for (let c = 0; c < cols && k < 124; c++) { const t = types[k++]; const x = (c - (cols - 1) / 2) * 1.25, z = (r - 5) * 1.25;
      const h = t === 0 ? 2.6 + R() * 1.6 : t === 1 ? 1.2 + R() * .9 : .45 + R() * .5; const m = new THREE.Mesh(BOX_UP, mats[t]); m.position.set(x, 0, z); m.scale.set(.86, 1e-4, .86); g.add(m); s.boxes.push({m, h, d: Math.hypot(x, z)}); }
    const grid = new THREE.GridHelper(44, 44, 0x2a3d66, 0x182640); grid.material.transparent = true; grid.material.opacity = .55; g.add(grid);
  },
  update(lt, s) { s.boxes.forEach(b => { const k = eo((lt - .4 - b.d * .12) / 1.1); b.m.scale.y = b.h * k + 1e-4; }); },
  cam(lt) { return orbit(lt, {c: V(0, .6, 0), r: 24, h: 13, a0: .65, w: .04, look: V(0, 1.2, 0)}); }});

// sets
def('sets', {shift: .2,
  build(g, s) {
    s.tiles = []; const add = (x, z, color, i, group) => { const m = new THREE.Mesh(BOX, glow(color, .5)); m.position.set(x, .12, z); m.scale.set(.5, 1e-4, .5); g.add(m); s.tiles.push({m, i, group, base: color}); };
    for (let i = 0; i < 60; i++) add(-6.6 + (i % 6) * .5, -2.6 + Math.floor(i / 6) * .55, C.gold, i, 'select');
    for (let i = 0; i < 200; i++) add(-2.0 + (i % 20) * .45, -2.6 + Math.floor(i / 20) * .55, C.teal, i, 'judge');
    s.tiles.forEach(t => t.m.scale.set(.4, 1e-4, .4));
    const div = new THREE.Mesh(new THREE.PlaneGeometry(.08, 6.4), new THREE.MeshBasicMaterial({color: C.white, transparent: true, opacity: .6})); div.position.set(-3.1, .02, 0); div.rotation.x = -Math.PI / 2; g.add(div);
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(6.4, 3.6), glass(C.blue, .08)); wall.position.set(-3.1, 1.8, 0); wall.rotation.y = Math.PI / 2; g.add(wall);
    const L1 = label('select · 60', {size: .5, color: CSS.gold, weight: 700}); L1.position.set(-5.35, 1.3, -3.7); g.add(L1);
    const L2 = label('judge · 200', {size: .5, color: CSS.teal, weight: 700}); L2.position.set(2.3, 1.3, -3.7); g.add(L2);
    const L3 = label('two halves of one company', {size: .42, color: CSS.mute}); L3.position.set(-3.1, 4.6, -1); g.add(L3);
    // 25 of the last 100 judge questions answer "not stated"
    const R = mulberry(3); const idx = [...Array(100).keys()].map(i => 100 + i).sort(() => R() - .5).slice(0, 25); s.ns = new Set(idx);
    s.nl = label('"not stated" · 25 of 100', {size: .48, color: CSS.lilac, weight: 700}); s.nl.material.depthTest = false; s.nl.renderOrder = 30; s.nl.position.set(2.6, .9, 3.7); g.add(s.nl);
  },
  update(lt, s) {
    s.tiles.forEach(t => { const k = eo((lt - s.c[0] - .4 - (t.group === 'select' ? t.i * .02 : .6 + t.i * .008)) / .6); t.m.scale.y = .22 * k + 1e-4; t.m.position.y = .11 * k;
      const hi = t.group === 'judge' && s.ns.has(t.i) ? sm(s.c[1] + 1, s.c[1] + 2, lt) : 0; t.m.material.color.setHex(hi > .5 ? C.lilac : t.base); t.m.material.emissive.setHex(hi > .5 ? C.lilac : t.base); t.m.position.y += hi * .5; });
    setOp(s.nl, sm(s.c[1] + 1.5, s.c[1] + 2.3, lt));
  },
  cam(lt) { return orbit(lt, {c: V(-.4, 0, 0), r: 18, h: 10.5, a0: .1, w: .01, look: V(-.4, .6, .2)}); }});

// loop
def('loop', {chap: 2, shift: .21,
  build(g, s) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(9, .05, 10, 240), glow(C.edge, .9)); ring.rotation.x = Math.PI / 2; g.add(ring);
    const st = [{a: -Math.PI / 2, n: 'A · measure seeds', c: C.cyan}, {a: Math.PI, n: 'B · predict outcomes', c: C.teal}, {a: Math.PI / 2, n: 'C · mutate + rank', c: C.lilac}, {a: 0, n: 'D · measure the elite', c: C.gold}];
    s.st = st.map((o, i) => { const p = V(Math.cos(o.a) * 9, 0, -Math.sin(o.a) * 9); const m = new THREE.Mesh(new THREE.OctahedronGeometry(.9, 0), glow(o.c, .15)); m.position.copy(p); g.add(m);
      const L = label(o.n, {size: .55, color: hex(o.c), weight: 700}); L.position.copy(p).add(V(0, o.a === Math.PI / 2 ? -1.7 : 1.8, 0)); g.add(L); return {...o, i, p, m, L}; });
    s.comet = []; for (let i = 0; i < 12; i++) { const m = new THREE.Mesh(SPH_LO, glow(C.cyan, 1.4)); g.add(m); s.comet.push(m); }
    const R = mulberry(5); const n = 1700; s.cn = n; s.cc = V(0, 5.2, -9); s.from = s.st[2].p.clone(); s.tgt = new Float32Array(n * 3); s.fit = new Float32Array(n);
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { let x, y, z; do { x = R() * 2 - 1; y = R() * 2 - 1; z = R() * 2 - 1; } while (x * x + y * y + z * z > 1); const f = R(); s.fit[i] = f;
      s.tgt.set([s.cc.x + x * 4.2, s.cc.y + y * 2.6, s.cc.z + z * 2.6], i * 3); const c = new THREE.Color(0x24407a).lerp(new THREE.Color(C.cyan), Math.pow(f, 1.6)); col.set([c.r, c.g, c.b], i * 3); }
    const gg = new THREE.BufferGeometry(); gg.setAttribute('position', new THREE.BufferAttribute(pos, 3)); gg.setAttribute('color', new THREE.BufferAttribute(col, 3));
    s.cloud = new THREE.Points(gg, new THREE.PointsMaterial({size: .24, map: dotTex, vertexColors: true, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending})); g.add(s.cloud);
    const idx = [...s.fit.keys()].sort((a, b) => s.fit[b] - s.fit[a]).slice(0, 3);
    s.elite = idx.map(i => { const m = new THREE.Mesh(SPH, glow(C.gold, 1.6)); m.scale.setScalar(1e-4); g.add(m); return {m, p0: V(s.tgt[i * 3], s.tgt[i * 3 + 1], s.tgt[i * 3 + 2])}; });
    s.cl = label('mutants ranked by the Predictor · 0 model calls', {size: .46, color: CSS.lilac}); s.cl.position.set(0, 8.9, -9); g.add(s.cl);
  },
  update(lt, s) {
    s.st.forEach(o => { const t0 = s.c[o.i + 1]; const k = sm(t0, t0 + .7, lt); o.m.material.emissiveIntensity = .15 + 1.3 * k; o.m.rotation.set(lt * .5, lt * .7, 0); setOp(o.L, .3 + .7 * k); });
    const th = -Math.PI / 2 - lt * .5; s.comet.forEach((m, i) => { const a = th + i * .045; m.position.set(Math.cos(a) * 9, 0, -Math.sin(a) * 9); m.scale.setScalar(.22 * (1 - i / 12)); });
    const tc = s.c[3]; const k = eo((lt - tc - .2) / 1.8); s.cloud.material.opacity = sm(tc, tc + .7, lt) * (1 - .75 * sm(s.c[4] + 1.5, s.c[4] + 3.5, lt));
    const pa = s.cloud.geometry.attributes.position; for (let i = 0; i < s.cn; i++) { const j = i * 3; const wob = Math.sin(lt * .8 + i) * .08; pa.array[j] = s.from.x + (s.tgt[j] - s.from.x) * k + wob; pa.array[j + 1] = s.from.y + (s.tgt[j + 1] - s.from.y) * k; pa.array[j + 2] = s.from.z + (s.tgt[j + 2] - s.from.z) * k + wob; } pa.needsUpdate = true;
    setOp(s.cl, sm(tc + 1.4, tc + 2.4, lt) * (1 - sm(s.c[4] + 2, s.c[4] + 3, lt)));
    const D = s.st[3].p, td = s.c[4]; s.elite.forEach((e, i) => { const a = sm(td - 1.4 + i * .25, td - .8 + i * .25, lt); const b = eio((lt - td - .2 - i * .2) / 1.6);
      const p = e.p0.clone().multiplyScalar(k).add(s.from.clone().multiplyScalar(1 - k)); const q = new THREE.Vector3().lerpVectors(p, D, b); q.y += Math.sin(Math.PI * b) * 2.5; e.m.position.copy(q); e.m.scale.setScalar(.34 * a * (1 - sm(td + 1.8, td + 2.6, lt)) + 1e-4); });
  },
  cam(lt) { const a = .12 + lt * .016; return {p: V(Math.sin(a) * 34, 15.5, Math.cos(a) * 34), l: V(0, 1.4, -1.2)}; }});

// operators
def('operators', {shift: .2,
  build(g, s) {
    const P = {P: V(0, 3.4, 0), W1: V(-3.4, -.6, 0), W2: V(0, -1.4, 1.2), W3: V(3.4, -.6, 0), N: V(-1.6, -4.2, .5), S: V(5.2, -2.6, 0)};
    s.P = P; s.n = {}; for (const [k, p] of Object.entries(P)) { s.n[k] = node(g, p, k === 'P' ? C.teal : C.cyan, {r: k === 'P' ? .8 : .6}); }
    s.L = {}; [['P', 'Coordinator'], ['W1', 'worker'], ['W2', 'worker'], ['W3', 'worker']].forEach(([k, t]) => { const L = label(t, {size: .38, color: CSS.mute}); L.position.copy(P[k]).add(V(0, k === 'P' ? 1.5 : -1.25, 0)); g.add(L); s.L[k] = L; });
    const mk = () => { const m = new THREE.Mesh(UNIT_BEAM(.05), new THREE.MeshBasicMaterial({color: C.edge, transparent: true, opacity: .8})); g.add(m); return m; };
    s.e = {PW1: mk(), PW2: mk(), PW3: mk(), PN: mk(), W2W1: mk(), PS: mk()};
    s.search = new THREE.Mesh(new THREE.TorusGeometry(1.15, .07, 10, 100), glow(C.gold, 1.6)); s.search.position.copy(P.W2); g.add(s.search);
    s.sl = label('search', {size: .34, color: CSS.gold}); s.sl.position.copy(P.W2).add(V(1.5, .6, 0)); g.add(s.sl);
    s.ml = label('mini', {size: .36, color: CSS.gold, weight: 700}); s.ml.position.copy(P.W1).add(V(-1.4, .7, 0)); g.add(s.ml);
    // the invalid mutant: a cycle
    s.bad = new THREE.Group(); s.bad.position.set(9.5, 1, -1); g.add(s.bad);
    const B = [V(0, 1.4, 0), V(-1.3, -.9, 0), V(1.3, -.9, 0)]; B.forEach(p => node(s.bad, p, C.coral, {r: .42, ring: false}));
    const bm = glow(C.coral, .9); [[0, 1], [1, 2], [2, 0]].forEach(([a, b]) => s.bad.add(beam(B[a], B[b], .05, bm)));
    s.badL = label('cycle · discarded', {size: .42, color: CSS.coral, weight: 700}); s.badL.position.set(9.5, -1.4, -1); g.add(s.badL);
    s.chips = [...Array(7).keys()].map(i => document.getElementById('op' + i));
  },
  update(lt, s) {
    const t0 = s.c[1] + .3, D = (s.c[2] - .4 - t0) / 7; const ph = i => clamp((lt - t0 - i * D) / (D * .7));
    const P = s.P, n = s.n;
    const show = (k, a) => { n[k].m.scale.setScalar(n[k].r * a + 1e-4); n[k].rg.scale.setScalar(n[k].r * a + 1e-4); };
    // add then remove the new agent
    const addN = ph(0) * (1 - ph(1)); show('N', addN);
    // split W3 into W3 and S, then merge back
    const sp = ph(3) * (1 - ph(4)); show('S', sp); n.S.m.position.lerpVectors(P.W3, P.S, sp); n.S.rg.position.copy(n.S.m.position);
    // rewire: P-W1 becomes W2-W1
    const rw = ph(2);
    const setE = (m, a, b, op) => { setBeam(m, a, b); m.material.opacity = .8 * op; m.visible = op > .01; };
    setE(s.e.PW1, P.P, P.W1, 1 - rw); setE(s.e.W2W1, P.W2, P.W1, rw); setE(s.e.PW2, P.P, P.W2, 1); setE(s.e.PW3, P.P, P.W3, 1);
    setE(s.e.PN, P.P, n.N.m.position, addN); setE(s.e.PS, P.P, n.S.m.position, sp);
    // toggle search on W2
    const ts = ph(5); s.search.scale.setScalar(.6 * ts + 1e-4); s.search.rotation.set(lt * .9, lt * .6, 0); setOp(s.sl, ts);
    // reassign W1's model
    const rm = ph(6); const col = new THREE.Color(C.cyan).lerp(new THREE.Color(C.gold), rm); n.W1.m.material.color.copy(col); n.W1.m.material.emissive.copy(col); n.W1.rg.material.color.copy(col); n.W1.rg.material.emissive.copy(col);
    n.W1.m.scale.setScalar(n.W1.r * (1 + .35 * rm)); setOp(s.ml, rm);
    spin(Object.values(n), lt);
    s.chips.forEach((c, i) => { if (c) c.classList.toggle('on', lt >= t0 + i * D && lt < t0 + (i + 1) * D); });
    const tb = s.c[2]; const kb = sm(tb, tb + .8, lt) * (1 - sm(tb + 3.0, tb + 4.0, lt)); setOp(s.bad, kb); s.bad.rotation.z = lt * .3; s.bad.scale.setScalar(1 - .5 * sm(tb + 2.6, tb + 4.0, lt) + 1e-4);
    setOp(s.badL, sm(tb + 1.2, tb + 2, lt));
  },
  cam(lt) { return orbit(lt, {c: V(2.2, -.4, 0), r: 25, h: 2, a0: -.1, w: .012}); }});

// predictor
def('predictor', {shift: .2,
  build(g, s) {
    const R = mulberry(13); s.fb = []; for (let i = 0; i < 13; i++) { const b = bar(g, -7.2 + i * .55, -3.5, .38, .38, 1.2 + R() * 4.2, [C.cyan, C.lilac, C.teal][i % 3], {ei: .55}); s.fb.push(b); }
    const fl = label('13 features', {size: .5, color: CSS.ink}); fl.position.set(-3.9, -4.6, 0); g.add(fl);
    const tree = (x, color, name) => { const grp = new THREE.Group(); grp.position.set(x, 0, 0); g.add(grp); const pts = [V(0, 2.8, 0), V(-1.4, 1, 0), V(1.4, 1, 0), V(-2.1, -.8, 0), V(-.7, -.8, 0), V(.7, -.8, 0), V(2.1, -.8, 0)];
      const em = glow(color, .7); [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]].forEach(([a, b]) => grp.add(beam(pts[a], pts[b], .05, em))); pts.forEach((p, i) => node(grp, p, color, {r: i ? .3 : .45, ring: false}));
      const L = label(name, {size: .55, color: hex(color), weight: 700, sub: 'gradient-boosted', subScale: .62}); L.position.set(0, -2.2, 0); grp.add(L); return grp; };
    s.t1 = tree(1.4, C.teal, 'accuracy'); s.t2 = tree(6.4, C.gold, 'tokens');
    s.flows = [flow(g, V(-3.9, 1, 0), V(1.4, 2.8, 0), {n: 4, speed: .5, color: C.teal}), flow(g, V(-3.9, 1, 0), V(6.4, 2.8, 0), {n: 4, speed: .45, color: C.gold, phase: .3})];
    s.no = label('fewer than 8 measurements → no fit', {size: .45, color: CSS.coral, weight: 700}); s.no.position.set(3.9, 5.2, 0); g.add(s.no);
  },
  update(lt, s) {
    s.fb.forEach((b, i) => b.set(eo((lt - s.c[0] - .2 - i * .06) / .9)));
    const kt = eo((lt - s.c[0] - .9) / 1.2); s.t1.scale.setScalar(kt + 1e-4); s.t2.scale.setScalar(kt + 1e-4);
    s.flows.forEach(f => f(lt, sm(s.c[0] + 1.5, s.c[0] + 2.5, lt)));
    const kn = sm(s.c[1] + .4, s.c[1] + 1.2, lt); setOp(s.no, kn); const dim = 1 - .45 * kn * (.5 + .5 * Math.sin(lt * 4)); setOp(s.t1, dim); setOp(s.t2, dim);
  },
  cam(lt) { return orbit(lt, {c: V(0, .5, 0), r: 28, h: 1.5, a0: -.1, w: .012}); }});

// split (what the surrogate predicts)
def('split', {chap: 2, shift: .2,
  build(g, s) {
    const bg = new THREE.BoxGeometry(2.1, 2.1, 2.1); const Y = 5.0;
    s.old = new THREE.Group(); s.old.position.set(0, Y, 0); g.add(s.old); s.old.add(new THREE.Mesh(bg, glass(C.coral, .18))); s.old.add(edgesOf(bg, C.coral));
    const ol = label('one model → fitness', {size: .5, color: CSS.coral}); ol.position.set(0, 2.0, 0); s.old.add(ol);
    const mk = (x, c, t, sub) => { const grp = new THREE.Group(); grp.position.set(x, Y, 0); g.add(grp); grp.add(new THREE.Mesh(bg, glass(c, .2))); grp.add(edgesOf(bg, c));
      const L = label(t, {size: .5, color: hex(c), sub, subScale: .7}); L.position.set(0, 2.2, 0); grp.add(L); return grp; };
    s.acc = mk(-4.2, C.teal, 'Predictor', 'accuracy'); s.tok = mk(4.2, C.gold, 'Predictor', 'tokens');
    s.cnt = new THREE.Group(); g.add(s.cnt); node(s.cnt, V(0, 7.4, 0), C.lilac, {r: .42, ei: 1.6}); const cl = label('agents · counted', {size: .42, color: CSS.lilac}); cl.position.set(0, 8.4, 0); s.cnt.add(cl);
    s.fit = new THREE.Group(); g.add(s.fit); node(s.fit, V(0, 2.2, 0), C.white, {r: .5, ei: 1.0, ring: false}); const fl = label('fitness = formula', {size: .44, mono: true, weight: 400, color: CSS.ink}); fl.position.set(0, 1.15, 0); s.fit.add(fl);
    s.links = new THREE.Group(); g.add(s.links); const em = glow(C.edge, .7); s.lf = [];
    [V(-4.2, Y - 1.1, 0), V(4.2, Y - 1.1, 0), V(0, 6.9, 0)].forEach(a => { s.links.add(beam(a, V(0, 2.2, 0), .04, em)); s.lf.push(flow(s.links, a, V(0, 2.2, 0), {n: 2, speed: .6})); });
    s.chart = new THREE.Group(); g.add(s.chart); const base = -3.2, k = 4.2; s.base = base; s.k = k;
    const zp = new THREE.Mesh(new THREE.BoxGeometry(14, .04, 2.6), new THREE.MeshBasicMaterial({color: 0x5a6780, transparent: true, opacity: .6})); zp.position.y = base; s.chart.add(zp);
    const B = [{x: -5.4, v: .63, c: C.teal, t: '+0.63', sub: null}, {x: -3.0, v: -.07, c: C.grey, t: '−0.07', sub: 'null'}, {x: 3.0, v: -.53, c: C.coral, t: '−0.53', sub: null}, {x: 5.4, v: -.17, c: C.grey, t: '−0.17', sub: 'null'}];
    s.B = B.map(b => { const m = new THREE.Mesh(BOX_UP, glow(b.c, b.c === C.grey ? .2 : .55)); m.position.set(b.x, base, 0); m.scale.set(1.5, 1e-4, 1.5); m.visible = false; s.chart.add(m);
      const L = label(b.t, {size: .6, mono: true, weight: 400, color: b.c === C.grey ? CSS.mute : hex(b.c), sub: b.sub, subScale: .6}); L.position.set(b.x, b.v > 0 ? base + b.v * k + .85 : base + b.v * k - (b.sub ? 1.1 : .7), 0); s.chart.add(L); return {...b, m, L}; });
    const ca = label('accuracy', {size: .5, weight: 700, color: CSS.teal}); ca.position.set(-4.2, base + .63 * k + 2.0, 0); s.chart.add(ca);
    const ct = label('token cost', {size: .5, weight: 700, color: CSS.coral}); ct.position.set(4.2, base + .9, 0); s.chart.add(ct); s.cl = [ca, ct];
  },
  update(lt, s) {
    const t1 = s.c[1], t2 = s.c[2]; const sp = eio((lt - t1) / 1.8);
    setOp(s.old, 1 - sm(t1 - .2, t1 + 1, lt)); s.old.scale.setScalar(1 - .4 * sp);
    s.acc.position.x = lerp(0, -4.2, sp); s.tok.position.x = lerp(0, 4.2, sp); s.acc.scale.setScalar(sp + 1e-4); s.tok.scale.setScalar(sp + 1e-4);
    setOp(s.cnt, sm(t1 + 2, t1 + 2.8, lt)); setOp(s.fit, sm(t1 + 2.4, t1 + 3.2, lt)); setOp(s.links, sm(t1 + 2.8, t1 + 3.6, lt)); s.lf.forEach(f => f(lt, sm(t1 + 2.8, t1 + 3.6, lt)));
    [s.acc, s.tok].forEach((b, i) => b.rotation.y = lt * .25 * (i ? -1 : 1));
    s.B.forEach((b, i) => { const k = eo((lt - t2 - .4 - i * .35) / 1.3); b.m.visible = k > .001; const h = Math.abs(b.v) * s.k * k + 1e-4; b.m.scale.y = h; b.m.position.y = s.base - (b.v < 0 ? h : 0); setOp(b.L, sm(t2 + 1 + i * .35, t2 + 1.7 + i * .35, lt)); });
    s.cl.forEach(c => setOp(c, sm(t2 + .4, t2 + 1.2, lt)));
    if (lt > t2 + 3) s.B[2].m.material.emissiveIntensity = .55 + .5 * Math.max(0, Math.sin((lt - t2 - 3) * 4));
  },
  cam(lt) { return orbit(lt, {c: V(0, 1.6, 0), r: 29, h: 1.2, a0: -.22, w: .02}); }});

// pareto
const FIRST = [[.9412, 179800, 5, 0], [.8824, 130026, 5, 0], [.8824, 136034, 5, 0], [.8824, 190889, 3, 0], [.8824, 172613, 5, 0], [.8824, 197194, 5, 0],
  [.8235, 124831, 2, 0], [.8235, 121335, 3, 0], [.8235, 158037, 3, 1], [.8235, 188858, 1, 1], [.8235, 192640, 4, 1], [.7647, 236725, 3, 0]];
def('pareto', {shift: .2,
  build(g, s) {
    const X = t => (t - 180000) / 10000, Y = a => (a - .85) * 45, Z = n => (n - 3) * 1.4;
    const dom = (a, b) => a[0] >= b[0] && a[1] <= b[1] && a[2] <= b[2] && (a[0] > b[0] || a[1] < b[1] || a[2] < b[2]);
    s.pts = FIRST.map((r, i) => { const front = !FIRST.some((o, j) => j !== i && dom(o, r)); const p = V(X(r[1]), Y(r[0]), Z(r[2]));
      const n = node(g, p, front ? C.gold : (r[3] ? C.violet : C.cyan), {r: .4 + r[2] * .05, ei: front ? 1.8 : .9}); return {n, front, p}; });
    const ax = new THREE.MeshBasicMaterial({color: 0x3a4a6a, transparent: true, opacity: .8});
    g.add(beam(V(-7, -4.6, -3.6), V(7, -4.6, -3.6), .03, ax)); g.add(beam(V(-7, -4.6, -3.6), V(-7, 5.5, -3.6), .03, ax)); g.add(beam(V(-7, -4.6, -3.6), V(-7, -4.6, 4), .03, ax));
    [['tokens →', V(4.5, -5.4, -3.6)], ['accuracy ↑', V(-7, 6.3, -3.6)], ['agents', V(-7.6, -4.9, 4.4)]].forEach(([t, p]) => { const L = label(t, {size: .42, color: CSS.mute}); L.position.copy(p); g.add(L); });
    const fr = s.pts.filter(p => p.front).sort((a, b) => a.p.x - b.p.x); s.fl = new THREE.Group(); g.add(s.fl); const fm = glow(C.gold, 1.2);
    for (let i = 0; i + 1 < fr.length; i++) s.fl.add(beam(fr[i].p, fr[i + 1].p, .04, fm));
    s.child = node(g, V(0, 6, 2), C.teal, {r: .5}); s.cb = new THREE.Group(); g.add(s.cb); const cm = new THREE.MeshBasicMaterial({color: C.teal, transparent: true, opacity: .7});
    fr.slice(0, 2).forEach(p => s.cb.add(beam(p.p, V(0, 6, 2), .03, cm)));
    s.cl = label('child', {size: .4, color: CSS.teal}); s.cl.position.set(0, 7.1, 2); g.add(s.cl);
  },
  update(lt, s) {
    s.pts.forEach((p, i) => { const k = eo((lt - s.c[0] - .2 - i * .08) / .7); p.n.m.scale.setScalar(p.n.r * k + 1e-4); p.n.rg.scale.setScalar(p.n.r * k + 1e-4); }); spin(s.pts.map(p => p.n), lt);
    setOp(s.fl, sm(s.c[0] + 1.6, s.c[0] + 2.6, lt)); const kc = eo((lt - s.c[1] - .8) / 1); s.child.m.scale.setScalar(.5 * kc + 1e-4); s.child.rg.scale.setScalar(.5 * kc + 1e-4); setOp(s.cb, sm(s.c[1] + .4, s.c[1] + 1.2, lt)); setOp(s.cl, kc);
  },
  cam(lt) { return orbit(lt, {c: V(0, 1, 0), r: 25, h: 3, a0: .45, w: -.02}); }});

// holdout
def('holdout', {shift: .2,
  build(g, s) {
    const R = mulberry(41); s.R = R; s.front = []; s.back = []; s.base = -3.5;
    for (let i = 0; i < 12; i++) { s.front.push(bar(g, -5.0 + i * 1.0, s.base, .7, .7, 1, C.cyan, {ei: .45, z: 1.5})); s.back.push(bar(g, -5.0 + i * 1.0, s.base, .7, .7, 1, C.violet, {ei: .45, z: -1.5})); }
    plate(g, s.base, 13.5, 5.6).position.x = .5;
    const lf = label('■ selection half', {size: .4, color: CSS.cyan, align: 'left'}); lf.position.set(-5.4, 4.3, 0); g.add(lf);
    const lb = label('■ the other half', {size: .4, color: CSS.violet, align: 'left'}); lb.position.set(-5.4, 3.6, 0); g.add(lb);
    // 200 splits: fixed random heights per split; the selection winner never tops the other half
    s.splits = []; for (let k = 0; k < 40; k++) { const f = [...Array(12)].map(() => 1 + R() * 5), b = [...Array(12)].map(() => 1 + R() * 5); const w = f.indexOf(Math.max(...f)); if (b[w] >= Math.max(...b)) b[w] = Math.max(...b) * .6; s.splits.push({f, b, w}); }
    s.rank = []; for (let k = 0; k < 40; k++) { const sp = s.splits[k]; const order = [...sp.b].sort((x, y) => y - x); s.rank.push(order.indexOf(sp.b[sp.w]) + 1); }
    s.rl = [...Array(12).keys()].map(r => { const L = label(`held-out rank ${r + 1} of 12`, {size: .5, color: CSS.gold, weight: 700}); L.position.set(3.2, 4.0, 0); g.add(L); L.visible = false; return L; });
    s.mean = label('mean held-out rank 7.2 of 12 · first in 0% of 200 splits', {size: .46, color: CSS.coral, weight: 700}); s.mean.position.set(1, 5.6, 0); g.add(s.mean);
  },
  update(lt, s) {
    const t0 = s.c[1] + .6, t1 = s.c[2] + 1; const kk = clamp(Math.floor((lt - t0) / .55), 0, 39); const sp = s.splits[kk]; const grow = eo((lt - s.c[0]) / 1.2);
    s.front.forEach((b, i) => { b.h = sp.f[i]; b.set(grow); const win = i === sp.w && lt > t0; b.m.material.color.setHex(win ? C.gold : C.cyan); b.m.material.emissive.setHex(win ? C.gold : C.cyan); });
    s.back.forEach((b, i) => { b.h = sp.b[i]; b.set(grow); const win = i === sp.w && lt > t0; b.m.material.color.setHex(win ? C.gold : C.violet); b.m.material.emissive.setHex(win ? C.gold : C.violet); });
    s.rl.forEach((L, r) => { L.visible = lt > t0 && lt < t1 && s.rank[kk] === r + 1; });
    setOp(s.mean, sm(s.c[1] + 4.5, s.c[1] + 5.3, lt));
  },
  cam(lt) { return orbit(lt, {c: V(1, .4, 0), r: 24, h: 6, a0: .3, w: -.01}); }});

// bank
def('bank', {shift: .2,
  build(g, s) {
    s.base = -4; plate(g, s.base, 18, 4);
    s.b = [bar(g, -6.2, s.base, 2, 2, 23 / 24 * 7, C.violet), bar(g, -3.6, s.base, 2, 2, 21 / 24 * 7, C.teal), bar(g, 3.6, s.base, 2, 2, .68 * 7, C.violet), bar(g, 6.2, s.base, 2, 2, .68 * 7, C.gold)];
    s.vl = [['23', CSS.violet], ['21', CSS.teal], ['68', CSS.violet], ['68', CSS.gold]].map(([t, c], i) => { const L = label(t, {size: .9, weight: 700, display: true, color: c}); g.add(L); return L; });
    s.nl = [['designer', -6.2], ['evolved', -3.6], ['designer', 3.6], ['+ stronger router', 6.2]].map(([t, x]) => { const L = label(t, {size: .38, color: CSS.ink}); L.position.set(x, s.base - .8, 1.4); g.add(L); return L; });
    s.g1 = label('held-out bank · of 24', {size: .48, color: CSS.mute, weight: 700}); s.g1.position.set(-4.9, s.base - 1.8, 1.4); g.add(s.g1);
    s.g2 = label('first paid run · of 100', {size: .48, color: CSS.mute, weight: 700}); s.g2.position.set(4.9, s.base - 1.8, 1.4); g.add(s.g2);
    s.x2 = label('2.1× the cost', {size: .5, color: CSS.coral, weight: 700}); s.x2.position.set(6.2, 3.6, 0); g.add(s.x2);
  },
  update(lt, s) {
    const T = [s.c[0] + .6, s.c[0] + 1.0, s.c[1] + .2, s.c[1] + .6];
    s.b.forEach((b, i) => { const v = b.set(eo((lt - T[i]) / 1.4)); s.vl[i].position.set(b.m.position.x, s.base + v + .9, 0); setOp(s.vl[i], sm(T[i] + .9, T[i] + 1.5, lt)); setOp(s.nl[i], sm(T[i], T[i] + .6, lt)); });
    setOp(s.g1, sm(T[0], T[0] + .6, lt)); setOp(s.g2, sm(T[2], T[2] + .6, lt)); setOp(s.x2, sm(s.c[1] + 2.4, s.c[1] + 3.2, lt));
  },
  cam(lt) { return orbit(lt, {c: V(0, -.6, 0), r: 25, h: 2, a0: -.15, w: .014}); }});

// stages
const STAGES = [['1', 'router promoted: tie', C.grey], ['2', 'six networks on select', C.blue], ['3', 'flat 155 · designer 135', C.teal], ['4', 'eight children of flat', C.blue],
  ['5', '2cc4 153 · designer 138', C.teal], ['6', 'new kinds: 77 · 68', C.lilac], ['7', '10 documents a search', C.cyan], ['8', '189 · 168', C.gold]];
def('stages', {shift: .2,
  build(g, s) {
    s.steps = STAGES.map(([n, t, c], i) => { const m = new THREE.Mesh(new THREE.BoxGeometry(1.6, .5, 3), glow(c, .15)); m.position.set(-7 + i * 1.75, -4.4 + i * 1.15, 0); g.add(m);
      const e = edgesOf(new THREE.BoxGeometry(1.6, .5, 3), c, .9); e.position.copy(m.position); g.add(e);
      const L = label(`${n} · ${t}`, {size: .42, weight: 700, color: hex(c), align: 'left'}); L.material.depthTest = false; L.renderOrder = 30; L.position.copy(m.position).add(V(1.15, .05, 1.5)); g.add(L); return {m, e, L, c}; });
    s.when = [s.c[0] + .8, s.c[1] - .6, s.c[1] + .4, s.c[2] - .8, s.c[2] + .4, s.c[2] + 2.6, 1e9, 1e9];
  },
  update(lt, s) {
    s.steps.forEach((st, i) => { const k = sm(s.when[i], s.when[i] + .8, lt); st.m.material.emissiveIntensity = .12 + .9 * k; setOp(st.L, .3 + .7 * k); st.m.position.y = -4.4 + i * 1.15 + .25 * k * Math.max(0, Math.sin((lt - s.when[i]) * 3)) * (1 - sm(s.when[i] + 1, s.when[i] + 2, lt)); });
  },
  cam(lt) { return orbit(lt, {c: V(1, -.2, 0), r: 28, h: 3, a0: .22, w: -.01}); }});

// cap
def('cap', {shift: .2,
  build(g, s) {
    const R = mulberry(19); s.docs = []; const tagged = new Set([0, 2, 3, 5, 6, 8, 9, 11]);
    for (let i = 0; i < 12; i++) { const a = -0.58 + i * .108; const p = V(Math.sin(a) * 11 + 1.0, -.4, -Math.cos(a) * 11 + 9);
      const m = panel(1.5, 2.1, (x, u, W2, H2) => { rrect(x, 0, 0, W2, H2, .12 * u, 'rgba(18,26,46,0.95)', 'rgba(140,170,230,0.5)', .03 * u); x.fillStyle = 'rgba(200,215,240,0.35)'; for (let l = 0; l < 7; l++) x.fillRect(.18 * u, (.3 + l * .22) * u, (.9 + R() * .3) * u, .06 * u);
        if (tagged.has(i)) { rrect(x, .14 * u, 1.72 * u, 1.2 * u, .26 * u, .08 * u, 'rgba(255,122,122,0.9)'); x.fillStyle = '#1a0a0a'; x.font = `700 ${.16 * u}px Inter`; x.textBaseline = 'middle'; x.fillText('mis-picked', .22 * u, 1.85 * u); } });
      m.position.copy(p); m.lookAt(1.0, -.4, 22); g.add(m); s.docs.push(m); }
    s.win = new THREE.Mesh(new THREE.BoxGeometry(1, 3, 1), glass(C.cyan, .14)); g.add(s.win); s.wine = edgesOf(new THREE.BoxGeometry(1, 3, 1), C.cyan, .9); g.add(s.wine);
    s.l3 = label('top 3', {size: .55, color: CSS.cyan, weight: 700}); g.add(s.l3); s.l10 = label('top 10', {size: .55, color: CSS.teal, weight: 700}); g.add(s.l10);
    s.res = label('designer: 69% → 84%', {size: .6, color: CSS.teal, weight: 700}); s.res.position.set(1.2, 3.6, 2); g.add(s.res);
  },
  update(lt, s) {
    s.docs.forEach((d, i) => setOp(d, sm(s.c[0] + .2 + i * .08, s.c[0] + .7 + i * .08, lt)));
    const n = lerp(3, 10, eio((lt - s.c[1] - .3) / 1.8));
    const a0 = -0.58, a1 = -0.58 + (n - 1) * .108; const mid = (a0 + a1) / 2; const r = 11; const w = 2 * r * Math.sin((a1 - a0) / 2) + 1.9;
    s.win.position.set(Math.sin(mid) * r + 1.0, -.4, -Math.cos(mid) * r + 9 + .2); s.win.rotation.y = -mid; s.win.scale.set(w, 1, 1); s.wine.position.copy(s.win.position); s.wine.rotation.copy(s.win.rotation); s.wine.scale.copy(s.win.scale);
    const vis = sm(s.c[0] + 1.2, s.c[0] + 2, lt); setOp(s.win, vis); setOp(s.wine, vis);
    s.l3.position.set(s.win.position.x, 1.9, s.win.position.z); s.l10.position.copy(s.l3.position); setOp(s.l3, vis * (1 - sm(s.c[1] + .2, s.c[1] + .8, lt))); setOp(s.l10, sm(s.c[1] + 1.4, s.c[1] + 2, lt));
    setOp(s.res, sm(s.c[1] + 2.4, s.c[1] + 3.2, lt));
  },
  cam(lt) { return orbit(lt, {c: V(.6, 0, 0), r: 22, h: 3, a0: -.05, w: .008, look: V(.6, -.2, 0)}); }});

// placement
def('placement', {shift: .2,
  build(g, s) {
    s.N = [node(g, V(-3, 3.4, 0), C.teal, {r: .8}), node(g, V(-3, -1.8, 0), C.cyan, {r: .85})]; const em = glow(C.edge, .65); g.add(beam(V(-3, 3.4, 0), V(-3, -1.8, 0), .05, em)); s.f = flow(g, V(-3, 3.4, 0), V(-3, -1.8, 0), {n: 3, speed: .5});
    const a = label('Coordinator', {size: .46, color: CSS.teal, weight: 700, sub: 'routes', subScale: .7, align: 'left'}); a.position.set(-1.6, 3.4, 0); g.add(a);
    const b = label('Researcher', {size: .46, color: CSS.cyan, weight: 700, sub: 'reads the documents and adds', subScale: .7, align: 'left'}); b.position.set(-1.6, -1.8, 0); g.add(b);
    s.orb = new THREE.Mesh(SPH, glow(C.gold, 2.2)); g.add(s.orb); s.ol = label('mini', {size: .4, color: CSS.gold, weight: 700}); g.add(s.ol);
    s.tie = label('stage 1: a tie', {size: .42, color: CSS.mute}); s.tie.position.set(-6.4, 3.4, 0); g.add(s.tie);
    s.base = -5; plate(g, s.base, 5, 3).position.x = 6.5; s.b0 = bar(g, 5.4, s.base, 1.6, 1.6, 52 / 60 * 7, C.cyan); s.b1 = bar(g, 7.6, s.base, 1.6, 1.6, 57 / 60 * 7, C.gold);
    s.v0 = label('52', {size: .8, weight: 700, display: true, color: CSS.cyan}); g.add(s.v0); s.v1 = label('57', {size: .8, weight: 700, display: true, color: CSS.gold}); g.add(s.v1);
    s.sel = label('of 60 on select', {size: .4, color: CSS.mute}); s.sel.position.set(6.5, s.base - .8, 1.2); g.add(s.sel);
  },
  update(lt, s) {
    spin(s.N, lt); s.f(lt);
    const t0 = s.c[0] + .8, t1 = s.c[0] + 3.0; const m = eio((lt - t1) / 1.6); const p = V(-3, 3.4, 0).lerp(V(-3, -1.8, 0), m).add(V(-1.4, .9 - Math.sin(Math.PI * m) * .4, .8));
    s.orb.position.copy(p); s.orb.scale.setScalar(.38 * sm(t0, t0 + .6, lt) + 1e-4); s.ol.position.copy(p).add(V(0, .75, 0)); setOp(s.ol, sm(t0, t0 + .6, lt)); setOp(s.tie, sm(t0 + .4, t0 + 1.2, lt) * (1 - sm(t1, t1 + .6, lt)));
    const up = sm(t1 + 1.2, t1 + 2, lt); const col = new THREE.Color(C.cyan).lerp(new THREE.Color(C.gold), up); s.N[1].m.material.color.copy(col); s.N[1].m.material.emissive.copy(col); s.N[1].m.material.emissiveIntensity = .9 + .6 * up;
    const v0 = s.b0.set(eo((lt - t0) / 1.2)); s.v0.position.set(5.4, s.base + v0 + .8, 0); setOp(s.v0, sm(t0 + .8, t0 + 1.3, lt));
    const v1 = s.b1.set(eo((lt - t1 - 1.4) / 1.2)); s.v1.position.set(7.6, s.base + v1 + .8, 0); setOp(s.v1, sm(t1 + 2.2, t1 + 2.7, lt)); setOp(s.sel, sm(t0, t0 + .6, lt));
  },
  cam(lt) { return orbit(lt, {c: V(1.2, -.6, 0), r: 24, h: 1.5, a0: -.1, w: .012}); }});

// q3 — the held-out head-to-head
def('q3', {shift: .2,
  build(g, s) {
    const D = [V(-5.6, 8.6, 0), V(-8, 6.2, 0), V(-5.6, 5.6, 1.2), V(-3.2, 6.2, 0)]; s.dn = D.map((p, i) => node(g, p, C.violet, {r: i ? .5 : .7, ei: 1.5}));
    const em = glow(C.edge, .6); s.fl = []; for (let i = 1; i < 4; i++) { g.add(beam(D[0], D[i], .04, em)); s.fl.push(flow(g, D[0], D[i], {n: 2, speed: .45, phase: i * .3})); }
    const E = [V(5.6, 8.6, 0), V(5.6, 5.8, 0)]; s.en = [node(g, E[0], C.teal, {r: .7, ei: 1.7}), node(g, E[1], C.gold, {r: .78, ei: 1.9})]; g.add(beam(E[0], E[1], .05, em)); s.fl.push(flow(g, E[0], E[1], {n: 3, speed: .55}));
    const a = label('Coordinator · nano', {size: .4, align: 'left', color: CSS.ink}); a.position.set(7.2, 8.6, 0); g.add(a);
    const b = label('Researcher · mini', {size: .4, align: 'left', color: CSS.gold}); b.position.set(7.3, 5.8, 0); g.add(b);
    const t1 = label("Designer's shape", {size: .62, weight: 700, color: CSS.violet, sub: '4 agents · all gpt-5.4-nano', subScale: .64}); t1.position.set(-5.6, 11.1, 0); g.add(t1);
    const t2 = label('2cc4', {size: .62, weight: 700, color: CSS.teal, sub: '2 agents · evolved', subScale: .64}); t2.position.set(5.6, 11.1, 0); g.add(t2);
    s.base = -6.6; plate(g, s.base);
    s.bars = [bar(g, -5.6, s.base, 3.4, 2.2, (84 - 60) * .25, C.violet), bar(g, 5.6, s.base, 3.4, 2.2, (94.5 - 60) * .25, C.teal)];
    s.vl = [label('84.0%', {size: 1.15, weight: 700, display: true, color: CSS.violet}), label('94.5%', {size: 1.15, weight: 700, display: true, color: CSS.teal})]; s.vl.forEach(l => g.add(l));
    s.ax = label('axis from 60%', {size: .34, color: CSS.mute}); s.ax.position.set(0, s.base - .75, 1.5); g.add(s.ax);
    s.pv = label('17 to 2 · p = 0.0007', {size: .55, mono: true, weight: 400, color: CSS.gold}); s.pv.position.set(0, 1.0, 0); g.add(s.pv);
  },
  update(lt, s) {
    spin(s.dn, lt); spin(s.en, lt); s.fl.forEach(f => f(lt));
    s.bars.forEach((b, i) => { const t0 = s.c[0] + 1.6 + i * .6; const v = b.set(eo((lt - t0) / 2.4)); s.vl[i].position.set(b.m.position.x, s.base + v + 1.1, 0); setOp(s.vl[i], sm(t0 + 1.8, t0 + 2.6, lt)); });
    setOp(s.ax, sm(s.c[0] + 1.6, s.c[0] + 2.4, lt)); setOp(s.pv, sm(s.c[1], s.c[1] + .8, lt));
  },
  cam(lt) { return orbit(lt, {c: V(0, 2.4, 0), r: 37, h: 1.5, a0: -.18, w: .012}); }});

// rep — replication with a model control
def('rep', {shift: .2,
  build(g, s) {
    s.base = -4.4; plate(g, s.base);
    const D = [{x: -5.4, acc: 90.0, c: C.teal, t: '90.0%', n: '2cc4', sub: ['Researcher on mini']}, {x: 0, acc: 77.5, c: C.lilac, t: '77.5%', n: 'Model control', sub: ["designer's shape,", 'specialists on mini']}, {x: 5.4, acc: 75.0, c: C.violet, t: '75.0%', n: "Designer's shape", sub: ['all nano']}];
    s.bars = D.map(d => { const b = bar(g, d.x, s.base, 3, 2, (d.acc - 50) * .24, d.c, {ei: .4}); const L = label(d.t, {size: 1.0, weight: 700, display: true, color: hex(d.c)}); g.add(L);
      const N = label(d.n, {size: .48, weight: 700, color: CSS.ink, sub: d.sub, subScale: .7}); N.position.set(d.x, s.base - 1.5, 1.2); g.add(N); return {...d, b, L, N}; });
    s.ax = label('axis from 50%', {size: .34, color: CSS.mute}); s.ax.position.set(-8.4, s.base + .5, 1.6); g.add(s.ax);
  },
  update(lt, s) {
    s.bars.forEach((d, i) => { const t0 = s.c[0] + 1.6 + i * .5; const v = d.b.set(eo((lt - t0) / 2.2)); d.L.position.set(d.x, s.base + v + 1.0, 0); setOp(d.L, sm(s.c[1] + i * .5, s.c[1] + .7 + i * .5, lt)); setOp(d.N, sm(t0 - .4, t0 + .4, lt)); });
    setOp(s.ax, sm(s.c[0] + 1.6, s.c[0] + 2.4, lt)); if (lt > s.c[2]) s.bars[0].b.m.material.emissiveIntensity = .4 + .35 * Math.max(0, Math.sin((lt - s.c[2]) * 3));
  },
  cam(lt) { return orbit(lt, {c: V(0, .4, 0), r: 29, h: 2, a0: .2, w: -.018}); }});

// cost
def('cost', {shift: .2,
  build(g, s) {
    s.base = -4.2; plate(g, s.base);
    s.b = [bar(g, -6.2, s.base, 2, 2, 3250 / 5153 * 7, C.teal), bar(g, -3.6, s.base, 2, 2, 7, C.violet), bar(g, 3.6, s.base, 2, 2, 7, C.coral), bar(g, 6.2, s.base, 2, 2, 135 / 263 * 7, C.violet)];
    s.vl = [['3,250', CSS.teal], ['5,153', CSS.violet], ['$0.00263', CSS.coral], ['$0.00135', CSS.violet]].map(([t, c]) => { const L = label(t, {size: .62, weight: 700, mono: true, color: c}); g.add(L); return L; });
    s.gl = [['tokens a question', -4.9], ['dollars a question', 4.9]].map(([t, x]) => { const L = label(t, {size: .48, color: CSS.ink, weight: 700}); L.position.set(x, s.base - 1.0, 1.4); g.add(L); return L; });
    s.who = [['2cc4', -6.2], ['designer', -3.6], ['2cc4', 3.6], ['designer', 6.2]].map(([t, x]) => { const L = label(t, {size: .36, color: CSS.mute}); L.position.set(x, s.base - 1.9, 1.4); g.add(L); return L; });
    s.x = label('−37%  ·  +94%', {size: .7, color: CSS.gold, weight: 700}); s.x.position.set(0, 4.8, 0); g.add(s.x);
  },
  update(lt, s) {
    const T = [s.c[0] + .4, s.c[0] + .8, s.c[1] + 1.2, s.c[1] + 1.6];
    s.b.forEach((b, i) => { const v = b.set(eo((lt - T[i]) / 1.3)); s.vl[i].position.set(b.m.position.x, s.base + v + .8, 0); setOp(s.vl[i], sm(T[i] + .9, T[i] + 1.4, lt)); setOp(s.who[i], sm(T[i], T[i] + .5, lt)); });
    setOp(s.gl[0], sm(T[0], T[0] + .6, lt)); setOp(s.gl[1], sm(T[2], T[2] + .6, lt)); setOp(s.x, sm(s.c[1] + 3, s.c[1] + 3.8, lt));
  },
  cam(lt) { return orbit(lt, {c: V(0, -.4, 0), r: 25, h: 2, a0: .15, w: -.012}); }});

// route — the frontier
def('route', {shift: .2,
  build(g, s) {
    const X = c => (c - .00195) * 9000, Y = a => (a - 89) * .82; s.X = X; s.Y = Y;
    const gm = new THREE.MeshBasicMaterial({color: 0x2a3a5c, transparent: true, opacity: .75});
    [.0015, .002, .0025].forEach(c => { const m = new THREE.Mesh(new THREE.BoxGeometry(.03, 12, .03), gm); m.position.set(X(c), Y(89.6), -1); g.add(m); const L = label('$' + c.toFixed(4), {size: .4, mono: true, weight: 400, color: CSS.mute}); L.position.set(X(c), -6.4, -1); g.add(L); });
    [85, 90, 95].forEach(a => { const m = new THREE.Mesh(new THREE.BoxGeometry(14.5, .03, .03), gm); m.position.set(.5, Y(a), -1); g.add(m); const L = label(a + '%', {size: .4, mono: true, weight: 400, color: CSS.mute}); L.position.set(-7.4, Y(a), -1); g.add(L); });
    const xl = label('cost per question →', {size: .44, color: CSS.ink}); xl.position.set(1.2, -7.4, -1); g.add(xl);
    const yl = label('accuracy ↑', {size: .44, color: CSS.ink}); yl.position.set(-6.6, 6.7, -1); g.add(yl);
    const P = [{n: "Designer's shape", c: .00135, a: 84.0, col: C.violet, lp: [.9, 0, 'left'], k: 0}, {n: '2cc4 alone', c: .00263, a: 94.5, col: C.teal, lp: [0, -1.55, 'center'], k: 0},
      {n: 'Router', c: .00186, a: 93.0, col: C.cyan, lp: [.15, -1.25, 'left'], k: 1}, {n: 'One-line rule', c: .00226, a: 94.5, col: C.gold, lp: [0, 1.5, 'center'], k: 2}];
    s.P = P.map((p, i) => { const pos = V(X(p.c), Y(p.a), 0); const n = node(g, pos, p.col, {r: .42, ei: 2.0}); const drop = new THREE.Mesh(new THREE.BoxGeometry(.025, 1, .025), new THREE.MeshBasicMaterial({color: p.col, transparent: true, opacity: .22})); drop.position.set(pos.x, (pos.y + Y(82.2)) / 2, 0); drop.scale.y = pos.y - Y(82.2); g.add(drop);
      const L = label(p.n, {size: .46, weight: 700, color: hex(p.col), sub: `${p.a.toFixed(1)}% · $${p.c.toFixed(5)}`, subScale: .72, align: p.lp[2]}); L.position.set(pos.x + p.lp[0], pos.y + p.lp[1], .3); g.add(L); return {...p, i, pos, n, drop, L}; });
    s.fr = new THREE.Group(); g.add(s.fr); const fm = glow(C.gold, 1.4); const order = [0, 2, 3]; for (let i = 0; i < 2; i++) s.fr.add(beam(s.P[order[i]].pos, s.P[order[i + 1]].pos, .05, fm));
    s.rnd = label('random routing, same share · 88.2%', {size: .38, color: CSS.mute, align: 'left'}); s.rnd.position.set(-6.5, Y(88.2) + .42, .3); g.add(s.rnd);
    s.rnd2 = new THREE.Mesh(new THREE.BoxGeometry(14.5, .035, .035), new THREE.MeshBasicMaterial({color: 0x8f9bb0, transparent: true, opacity: .55})); s.rnd2.position.set(.5, Y(88.2), .1); g.add(s.rnd2);
  },
  update(lt, s) {
    s.P.forEach(p => { const t = s.c[p.k] + (p.k === 0 ? 1 + p.i * .8 : .8); const k = eo((lt - t) / .9); p.n.m.scale.setScalar(.42 * k + 1e-4); p.n.rg.scale.setScalar(.42 * k + 1e-4); setOp(p.L, sm(t + .3, t + 1, lt)); setOp(p.drop, sm(t, t + .8, lt)); }); spin(s.P.map(p => p.n), lt);
    setOp(s.fr, sm(s.c[2] + 2.2, s.c[2] + 3.2, lt)); setOp(s.rnd, sm(s.c[1] + 5.5, s.c[1] + 6.3, lt)); setOp(s.rnd2, sm(s.c[1] + 5.5, s.c[1] + 6.3, lt));
  },
  cam(lt) { return orbit(lt, {c: V(.4, .3, 0), r: 26, h: 1, a0: .28, w: -.012}); }});

// agree — two runs per question
def('agree', {shift: .2,
  build(g, s) {
    const R = mulberry(77); const idx = [...Array(100).keys()].sort(() => R() - .5); s.dis = new Set(idx.slice(0, 18)); s.pairs = [];
    for (let i = 0; i < 100; i++) { const x = -5.4 + (i % 10) * 1.2, z = -5.4 + Math.floor(i / 10) * 1.2; const d = s.dis.has(i); const col = d ? C.coral : C.teal;
      const a = new THREE.Mesh(SPH_LO, glow(col, 1.4)), b = new THREE.Mesh(SPH_LO, glow(d ? C.gold : col, 1.4)); g.add(a); g.add(b); s.pairs.push({a, b, x, z, d, i}); }
    s.gate = new THREE.Mesh(new THREE.TorusGeometry(2.0, .08, 12, 100), glow(C.gold, 1.4)); s.gate.position.set(7.6, 2.5, 0); s.gate.rotation.y = Math.PI / 2; g.add(s.gate);
    s.gl = label('one-line rule', {size: .42, color: CSS.gold, weight: 700, sub: 'already sends these up', subScale: .72}); s.gl.position.set(7.6, -.8, 0); g.add(s.gl);
    s.l1 = label('82 agree', {size: .5, color: CSS.teal, weight: 700}); s.l1.position.set(-3.2, 3.4, -6); g.add(s.l1);
    s.l2 = label('18 disagree', {size: .5, color: CSS.coral, weight: 700}); s.l2.position.set(1.4, 3.4, -6); g.add(s.l2);
  },
  update(lt, s) {
    const t1 = s.c[1] + .6;
    s.pairs.forEach(p => { const k = eo((lt - s.c[0] - .3 - p.i * .02) / .6); const fly = p.d ? eio((lt - t1 - p.i * .01) / 1.6) : 0;
      const base = V(p.x, .4, p.z); const tgt = V(7.6, 2.5, (p.i % 6 - 2.5) * .5); const pos = base.clone().lerp(tgt, fly); pos.y += Math.sin(Math.PI * fly) * 2;
      p.a.position.copy(pos).add(V(-.18, 0, 0)); p.b.position.copy(pos).add(V(.18, p.d ? .25 : 0, 0)); const sc = .16 * k * (1 - .7 * sm(.85, 1, fly)) + 1e-4; p.a.scale.setScalar(sc); p.b.scale.setScalar(sc); });
    const kg = sm(t1 - .4, t1 + .4, lt); s.gate.scale.setScalar(kg + 1e-4); s.gate.rotation.x = lt * .5; setOp(s.gl, kg); setOp(s.l1, sm(s.c[0] + 2.4, s.c[0] + 3, lt)); setOp(s.l2, sm(s.c[0] + 2.8, s.c[0] + 3.4, lt));
  },
  cam(lt) { return orbit(lt, {c: V(.8, .4, 0), r: 23, h: 11, a0: .28, w: -.01, look: V(.8, .8, 0)}); }});

// autopsy — penalty arithmetic
def('autopsy', {shift: .2,
  build(g, s) {
    s.base = -3.6; plate(g, s.base, 18, 4);
    s.r1 = bar(g, -4.6, s.base, 1.5, 1.5, 4375 / 5000 * 6, C.gold); s.r2 = bar(g, -2.5, s.base, 1.5, 1.5, 3875 / 5000 * 6, C.gold);
    s.h1 = bar(g, 3.6, s.base, 1.5, 1.5, 2.4, C.cyan); s.h2 = bar(g, 5.7, s.base, 1.5, 1.5, 3.4, C.cyan);
    s.lb = [['4375 / h', -4.6], ['3875 / h', -2.5], ['hours', 3.6], ['hours', 5.7]].map(([t, x]) => { const L = label(t, {size: .34, mono: true, weight: 400, color: CSS.ink}); L.position.set(x, s.base - .8, 1.4); g.add(L); return L; });
    [['two contracts\' hourly rates', -3.55], ['each incident\'s hours late', 4.65]].forEach(([t, x]) => { const L = label(t, {size: .36, color: CSS.mute}); L.position.set(x, s.base - 1.6, 1.4); g.add(L); });
    s.ok = label('right: rate × hours, for each incident', {size: .48, color: CSS.teal, weight: 700}); s.ok.position.set(.6, 5.6, 0); g.add(s.ok);
    s.no = label('4375 − 3875 = 500', {size: .72, color: CSS.coral, weight: 700, mono: true}); s.no.position.set(-1.6, 3.7, 1); g.add(s.no);
    s.nl = label('never multiplied by the hours', {size: .4, color: CSS.coral}); s.nl.position.set(-1.6, 2.7, 1); g.add(s.nl);
    s.mul = label('×', {size: 1.3, color: CSS.teal, weight: 700}); s.mul.position.set(.6, .2, 0); g.add(s.mul);
    s.miss = label('missed documents: at most 2 of 60 wrong runs', {size: .4, color: CSS.mute}); s.miss.position.set(.6, -6.0, 1.4); g.add(s.miss);
  },
  update(lt, s) {
    const t0 = s.c[0] + .5; [s.r1, s.r2, s.h1, s.h2].forEach((b, i) => { b.set(eo((lt - t0 - i * .3) / 1.2)); setOp(s.lb[i], sm(t0 + i * .3, t0 + .6 + i * .3, lt)); });
    setOp(s.ok, sm(t0 + 1.6, t0 + 2.4, lt)); setOp(s.mul, sm(t0 + 1.6, t0 + 2.4, lt)); setOp(s.miss, sm(s.c[1] + .2, s.c[1] + 1, lt)); setOp(s.no, sm(s.c[1] + 2.2, s.c[1] + 3, lt)); setOp(s.nl, sm(s.c[1] + 2.8, s.c[1] + 3.6, lt));
  },
  cam(lt) { return orbit(lt, {c: V(.6, .4, 0), r: 25, h: 2, a0: -.1, w: .01}); }});

// defects — a wall of cards
const DEFECTS = [['Timeouts → wrong answers', 'now: unfinished, not wrong', C.coral], ['A rate limiter froze the loop', 'time.sleep in async · ruff ASYNC251', C.coral], ['The cost term had no gradient', 'every network clipped to one penalty', C.coral],
  ['INC-4429 tripped the quota guard', '"429" read anywhere in a reply', C.gold], ['Marks for listing numbers', 'replies of 0–20 held 90 of 200 answers', C.gold], ['Tokens counted twice', 'fixed on 10 October', C.lilac],
  ['The front man was alphabetical', 'flat_pair scored 0.00', C.cyan], ['The image could not run the service', 'apps/ and registries/ were missing', C.cyan], ['No headroom', 'one agent scored 1.00 on 13 questions', C.cyan]];
def('defects', {shift: .2,
  build(g, s) {
    s.cards = DEFECTS.map(([t, sub, col], i) => { const grp = new THREE.Group(); grp.position.set(-3.6 + (i % 3) * 4.9, 3.0 - Math.floor(i / 3) * 2.95, 0); g.add(grp);
      const front = panel(4.8, 2.7, (x, u, W2, H2) => { rrect(x, 0, 0, W2, H2, .2 * u, 'rgba(14,22,40,0.96)', hex(col), .05 * u); x.fillStyle = hex(col); x.fillRect(.25 * u, .3 * u, .5 * u, .08 * u);
        x.textBaseline = 'top'; x.fillStyle = '#dfe5ee'; x.font = `700 ${.3 * u}px Inter`; wrap(x, t, .25 * u, .6 * u, 4.3 * u, .38 * u); x.fillStyle = '#b6c1d3'; x.font = `400 ${.22 * u}px Inter`; wrap(x, sub, .25 * u, 1.75 * u, 4.3 * u, .3 * u); }, {front: true});
      const back = panel(4.8, 2.7, (x, u, W2, H2) => { rrect(x, 0, 0, W2, H2, .2 * u, 'rgba(20,28,48,0.9)', 'rgba(120,150,210,0.35)', .03 * u); x.fillStyle = 'rgba(140,160,200,0.4)'; x.font = `700 ${.5 * u}px 'DejaVu Sans Mono'`; x.textAlign = 'center'; x.textBaseline = 'middle'; x.fillText('?', W2 / 2, H2 / 2); }, {front: true});
      back.rotation.y = Math.PI; grp.add(front); grp.add(back); return {grp, i}; });
    const order = [0, 1, 2, 3, 4, 5, 6, 7, 8]; s.when = order.map(i => i < 3 ? s.c[1] + .4 + i * 1.6 : i < 5 ? s.c[2] + .4 + (i - 3) * 2.6 : i === 5 ? s.c[3] + .3 : s.c[3] + 1.6 + (i - 6) * .5);
  },
  update(lt, s) { s.cards.forEach(c => { const k = eio((lt - s.when[c.i]) / .8); c.grp.rotation.y = Math.PI * (1 - k) + Math.sin(lt * .4 + c.i) * .04; c.grp.position.z = Math.sin(Math.PI * k) * 1.2; setOp(c.grp, sm(s.c[0] - .4 + c.i * .05, s.c[0] + .4 + c.i * .05, lt)); }); },
  cam(lt) { return orbit(lt, {c: V(1.2, .1, 0), r: 23, h: 1, a0: -.05, w: .006}); }});
function wrap(x, text, X, Y, maxW, lh) { const words = text.split(' '); let line = '', y = Y; for (const w of words) { const t = line ? line + ' ' + w : w; if (x.measureText(t).width > maxW && line) { x.fillText(line, X, y); line = w; y += lh; } else line = t; } if (line) x.fillText(line, X, y); }

// audit — documents reread
def('audit', {shift: .2,
  build(g, s) {
    s.sheets = []; for (let i = 0; i < 12; i++) { const m = panel(2.4, 3.2, (x, u, W2, H2) => { rrect(x, 0, 0, W2, H2, .1 * u, 'rgba(18,26,46,0.95)', 'rgba(140,170,230,0.5)', .03 * u); x.fillStyle = 'rgba(200,215,240,0.35)'; for (let l = 0; l < 10; l++) x.fillRect(.2 * u, (.3 + l * .25) * u, (1.5 + ((i * 7 + l * 3) % 5) * .1) * u, .06 * u); });
      m.position.set(-4.2 + (i % 6) * 2.6, 2 - Math.floor(i / 6) * 3.8, 0); g.add(m); s.sheets.push(m); }
    s.ring = new THREE.Mesh(new THREE.TorusGeometry(2.2, .06, 10, 100), glow(C.cyan, 1.6)); s.ring.rotation.y = Math.PI / 2; g.add(s.ring);
    s.checks = s.sheets.map((m, i) => { const flag = [2, 5, 7, 10].includes(i); const L = label(flag ? '!' : '✓', {size: .8, weight: 700, color: flag ? CSS.coral : CSS.teal}); L.position.copy(m.position).add(V(.8, 1.1, .3)); g.add(L); const L2 = label('✓', {size: .8, weight: 700, color: CSS.teal}); L2.position.copy(L.position); g.add(L2); return {L, L2, flag}; });
    s.cl = label('corrected where they stand', {size: .5, color: CSS.gold, weight: 700}); s.cl.position.set(2.3, 5.6, 0); g.add(s.cl);
  },
  update(lt, s) {
    const t0 = s.c[0] + .4, t1 = s.c[0] + 4.6; const x = lerp(-6, 11, clamp((lt - t0) / (t1 - t0))); s.ring.position.set(x, 0, 0); setOp(s.ring, sm(t0, t0 + .4, lt) * (1 - sm(t1, t1 + .5, lt)));
    s.sheets.forEach((m, i) => setOp(m, sm(s.c[0] - .5 + i * .05, s.c[0] + .3 + i * .05, lt)));
    s.checks.forEach((c, i) => { const px = s.sheets[i].position.x; const seen = lt > t0 ? sm(px - .2, px + .6, x) : 0;
      if (!c.flag) { setOp(c.L, seen); setOp(c.L2, 0); return; }
      const fix = sm(s.c[1] + 1.8, s.c[1] + 2.6, lt); setOp(c.L, sm(s.c[1], s.c[1] + .6, lt) * (1 - fix)); setOp(c.L2, fix); });
    setOp(s.cl, sm(s.c[1] + 2.2, s.c[1] + 3, lt));
  },
  cam(lt) { return orbit(lt, {c: V(1.2, .2, 0), r: 23, h: 1.2, a0: .06, w: -.006}); }});

// q5 — the service
def('q5', {shift: .2,
  build(g, s) {
    const r1 = [['neuro-san server', 'cron · 0 * * * *'], ['optimizer agent', 'user_id: system'], ['two tools', 'RunWake · ReportFinding'], ['wake()', '≤ 3 candidates']];
    const r2 = [['preflight', 'refuses a bad key'], ['lease.json', 'no overlapping wakes'], ['state.json', 'atomic, per candidate'], ['cache', 'keyed by genome hash'], ['budget', "the day's spend, then stop"]];
    const c1 = [C.cyan, C.teal, C.lilac, C.gold], c2 = [C.coral, C.cyan, C.teal, C.lilac, C.gold];
    s.b1 = r1.map((r, i) => block(g, V(-7.8 + i * 5.2, 4.4, 0), 4.7, 2.3, .5, c1[i], r[0], r[1], {size: .5}));
    s.b2 = r2.map((r, i) => block(g, V(-8.2 + i * 4.1, -2.2, 0), 3.6, 2.2, .5, c2[i], r[0], r[1], {size: .44}));
    s.b3 = block(g, V(0, -7.6, 0), 9.5, 1.9, .5, C.blue, 'results/ · committed logs', 'every figure re-derived by a test', {size: .46});
    const em = glow(C.edge, .7), em2 = glow(C.edge, .7), em3 = glow(C.edge, .7); s.lk1 = new THREE.Group(); s.lk2 = new THREE.Group(); s.lk3 = new THREE.Group(); g.add(s.lk1, s.lk2, s.lk3); s.fl1 = []; s.fl2 = []; s.fl3 = [];
    for (let i = 0; i < 3; i++) { const a = V(-7.8 + i * 5.2 + 2.35, 4.4, 0), b = V(-7.8 + (i + 1) * 5.2 - 2.35, 4.4, 0); s.lk1.add(beam(a, b, .05, em)); s.fl1.push(flow(s.lk1, a, b, {n: 2, speed: .8})); }
    const w = V(7.8, 4.4 - 1.15, 0); for (let i = 0; i < 5; i++) { const b = V(-8.2 + i * 4.1, -2.2 + 1.1, 0); s.lk2.add(beam(w, b, .035, em2)); s.fl2.push(flow(s.lk2, w, b, {n: 2, speed: .5, phase: i * .17})); }
    for (let i = 0; i < 5; i++) { const a = V(-8.2 + i * 4.1, -2.2 - 1.1, 0), b = V(-3 + i * 1.5, -7.6 + .95, 0); s.lk3.add(beam(a, b, .03, em3)); s.fl3.push(flow(s.lk3, a, b, {n: 1, speed: .5, phase: i * .2})); }
  },
  update(lt, s) {
    const lit = (bk, t0) => { const k = sm(t0, t0 + .8, lt); bk.fill.material.opacity = .35 + .55 * k; bk.fill.material.emissiveIntensity = .04 + .22 * k; bk.e.material.opacity = .25 + .75 * k; setOp(bk.L, .25 + .75 * k); bk.grp.scale.setScalar(.92 + .08 * eo((lt - t0) / .8)); };
    const A = s.c[1], B = s.c[2]; s.b1.forEach((b, i) => lit(b, A + i * .9)); s.b2.forEach((b, i) => lit(b, B + i * 1.1)); lit(s.b3, B + 6.4);
    setOp(s.lk1, sm(A + .6, A + 3, lt)); s.fl1.forEach(f => f(lt, sm(A + .6, A + 3, lt))); setOp(s.lk2, sm(B - .4, B + 1, lt)); s.fl2.forEach(f => f(lt, sm(B - .4, B + 1, lt))); setOp(s.lk3, sm(B + 5.8, B + 6.8, lt)); s.fl3.forEach(f => f(lt, sm(B + 5.8, B + 6.8, lt)));
  },
  cam(lt) { return orbit(lt, {c: V(0, -.6, 0), r: 31, h: 2.2, a0: -.14, w: .01}); }});

// providers
def('providers', {shift: .2,
  build(g, s) {
    const P = [['Anthropic', 'claude-haiku', 'claude-sonnet', C.coral, -4], ['OpenAI', 'gpt-5.4-mini', 'gpt-5.5', C.teal, 1.2], ['Gemini', 'gemini-3.1-flash-lite', 'gemini-3.5-flash', C.cyan, 6.4]];
    s.p = P.map(([name, lo, hi, col, x]) => { const grp = new THREE.Group(); grp.position.set(x, 0, 0); g.add(grp); const ring = new THREE.Mesh(new THREE.TorusGeometry(2.3, .05, 10, 120), glow(col, 1.0)); ring.rotation.x = Math.PI / 2; ring.position.y = -1; grp.add(ring);
      const a = node(grp, V(0, -1, 0), col, {r: .45}), b = node(grp, V(0, 2.2, 0), col, {r: .7}); grp.add(beam(V(-.4, -1, 0), V(-.4, 2.2, 0), .03, glow(col, .6))); grp.add(beam(V(.4, -1, 0), V(.4, 2.2, 0), .03, glow(col, .6)));
      for (let k = 0; k < 4; k++) grp.add(beam(V(-.4, -.4 + k * .65, 0), V(.4, -.4 + k * .65, 0), .025, glow(col, .5)));
      const L = label(name, {size: .6, weight: 700, color: hex(col)}); L.position.set(0, 4.2, 0); grp.add(L);
      const Lh = label(hi, {size: .34, mono: true, weight: 400, color: CSS.ink}); Lh.position.set(0, 3.25, 0); grp.add(Lh); const Ll = label(lo, {size: .34, mono: true, weight: 400, color: CSS.mute}); Ll.position.set(0, -2.6, 0); grp.add(Ll);
      return {grp, a, b}; });
    s.h = label('the model is in the genome hash', {size: .5, color: CSS.lilac, weight: 700}); s.h.position.set(1.2, -4.6, 0); g.add(s.h);
    s.reset = label('new provider → new hashes → start from the seeds', {size: .44, color: CSS.gold}); s.reset.position.set(1.2, -5.6, 0); g.add(s.reset);
    s.orb = new THREE.Mesh(SPH, glow(C.white, 1.6)); g.add(s.orb);
  },
  update(lt, s) {
    s.p.forEach((p, i) => { const k = eo((lt - s.c[0] - .4 - i * .4) / 1); p.grp.scale.setScalar(k + 1e-4); spin([p.a, p.b], lt + i); });
    setOp(s.h, sm(s.c[1], s.c[1] + .8, lt)); setOp(s.reset, sm(s.c[1] + 1.6, s.c[1] + 2.4, lt));
    const m = eio((lt - s.c[1] - .6) / 2); const a = V(1.2, 2.2, 0), b = V(6.4, 2.2, 0); s.orb.position.lerpVectors(a, b, m); s.orb.position.y += Math.sin(Math.PI * m) * 2; s.orb.scale.setScalar(.3 * sm(s.c[1] + .4, s.c[1] + .8, lt) * (1 - sm(s.c[1] + 2.6, s.c[1] + 3, lt)) + 1e-4);
  },
  cam(lt) { return orbit(lt, {c: V(1, .2, 0), r: 25, h: 2.5, a0: -.08, w: .01}); }});

// ci — gates
def('ci', {shift: .19,
  build(g, s) {
    const names = ['pinned actions', 'pip-audit', 'ruff', 'pymarkdown', 'HOCON validator', 'pytest · cov ≥ 85%', 'figures = logs', 'offline search']; s.gates = [];
    names.forEach((n, i) => { const x = -8.4 + i * 2.4; const m = new THREE.Mesh(new THREE.TorusGeometry(1.25, .07, 12, 100), glow(C.grey, .3)); m.rotation.y = Math.PI / 2; m.position.set(x, 0, 0); g.add(m);
      const L = label(n, {size: .4, mono: true, weight: 400, color: CSS.ink}); L.position.set(x, i % 2 ? 2.05 : 3.0, 0); g.add(L);
      const tick = new THREE.Mesh(new THREE.BoxGeometry(.02, i % 2 ? .5 : 1.45, .02), new THREE.MeshBasicMaterial({color: 0x4a5a7a})); tick.position.set(x, i % 2 ? 1.5 : 1.98, 0); g.add(tick); s.gates.push({m, L, x, i}); });
    const rm = glow(C.edge, .5); [-.9, .9].forEach(z => { const r = new THREE.Mesh(new THREE.BoxGeometry(24, .05, .05), rm); r.position.set(0, -1.5, z); g.add(r); });
    s.orb = new THREE.Mesh(SPH, glow(C.white, 1.4)); s.orb.scale.setScalar(.42); g.add(s.orb);
    s.trail = []; for (let i = 0; i < 10; i++) { const m = new THREE.Mesh(SPH_LO, glow(C.cyan, 1.3)); g.add(m); s.trail.push(m); }
    s.hl = label('a claim that drifts from its evidence fails the build', {size: .46, color: CSS.gold, weight: 700}); s.hl.position.set(4, -3.6, 0); g.add(s.hl);
  },
  update(lt, s) {
    const x = lerp(-12.5, 11, eio((lt - s.c[0]) / (s.c[1] + 2.5 - s.c[0]))); s.orb.position.set(x, 0, 0); s.trail.forEach((m, i) => { m.position.set(x - (i + 1) * .32, 0, 0); m.scale.setScalar(.22 * (1 - i / 10) + 1e-4); });
    s.gates.forEach(gt => { const k = sm(gt.x - .4, gt.x + .4, x); const hot = gt.i === 6 ? sm(s.c[1], s.c[1] + .6, lt) * (.6 + .4 * Math.sin(lt * 5)) : 0; gt.m.material.color.setHex(k > .5 ? (hot > .1 ? C.gold : C.teal) : C.grey); gt.m.material.emissive.setHex(k > .5 ? (hot > .1 ? C.gold : C.teal) : C.grey); gt.m.material.emissiveIntensity = .3 + 1.1 * k + hot; setOp(gt.L, .5 + .5 * k); });
    setOp(s.hl, sm(s.c[1] + 1, s.c[1] + 1.8, lt));
  },
  cam(lt) { return orbit(lt, {c: V(0, -1.2, 0), r: 31, h: 6, a0: .36, w: -.014}); }});

// serve — the page and its guards
def('serve', {shift: .2,
  build(g, s) {
    s.page = panel(10.4, 6.6, (x, u, W2, H2) => {
      rrect(x, 0, 0, W2, H2, .25 * u, '#c3c9d4', 'rgba(120,160,255,0.6)', .04 * u); x.fillStyle = '#aeb5c2'; x.fillRect(0, 0, W2, .6 * u);
      ['#c45050', '#c9a046', '#2fa57f'].forEach((c, i) => { x.beginPath(); x.arc((.4 + i * .3) * u, .3 * u, .09 * u, 0, 7); x.fillStyle = c; x.fill(); });
      x.textBaseline = 'top'; x.fillStyle = '#0f172a'; x.font = `700 ${.36 * u}px Inter`; x.fillText('Talking to a measured agent network', .6 * u, 1.0 * u);
      x.font = `600 ${.22 * u}px Inter`; x.fillStyle = '#1e40af'; x.fillText('Ask a network', .6 * u, 1.65 * u); x.fillStyle = '#475569'; x.fillText('Measure networks', 2.6 * u, 1.65 * u);
      rrect(x, .6 * u, 2.2 * u, 9.2 * u, 1.0 * u, .12 * u, '#d9dde5', '#9aa3b2', .02 * u); x.fillStyle = '#0f172a'; x.font = `400 ${.22 * u}px Inter`; x.fillText('What is the total penalty owed for incident INC-4401?', .8 * u, 2.5 * u);
      rrect(x, .6 * u, 3.5 * u, 9.2 * u, 2.6 * u, .12 * u, '#111827'); x.font = `${.2 * u}px 'DejaVu Sans Mono'`; x.fillStyle = '#cfd6e0';
      ['answer:   The total penalty owed for incident INC-4401 is 4500. ...', 'expected: 4500', 'correct:  true', 'explains: every identifier it followed, and the arithmetic'].forEach((l, i) => x.fillText(l, .85 * u, (3.75 + i * .5) * u));
    });
    s.page.position.set(-.6, .9, 0); g.add(s.page);
    s.ends = [['web page', C.cyan], ['accelerator UI', C.teal], ['registry', C.lilac], ['Hugging Face Space', C.gold]].map(([t, c], i) => { const p = V(7.2, 3.6 - i * 2.4, 0); const n = node(g, p, c, {r: .5}); const L = label(t, {size: .38, color: hex(c), weight: 700}); L.position.copy(p).add(V(0, -1.05, 0)); g.add(L);
      const b = beam(V(4.6, .9, 0), p, .025, new THREE.MeshBasicMaterial({color: c, transparent: true, opacity: .55})); g.add(b); return {n, L, b}; });
    s.guards = ['caps per day', 'bearer token', 'X-Request-ID', '400 KB limit', 'keys masked'].map((t, i) => { const L = label(t, {size: .34, color: CSS.gold, weight: 700}); L.position.set(i < 3 ? -4.4 + i * 3.8 : -2.5 + (i - 3) * 3.8, i < 3 ? -3.5 : -4.3, .8); g.add(L); return L; });
    s.shield = new THREE.Mesh(new THREE.TorusGeometry(6.0, .05, 10, 160), glow(C.gold, 1.2)); s.shield.scale.set(1, .6, 1); s.shield.position.set(-.6, 1.1, -.6); g.add(s.shield);
  },
  update(lt, s) {
    setOp(s.page, sm(s.c[0] - .4, s.c[0] + .6, lt)); s.page.rotation.y = Math.sin(lt * .25) * .06;
    s.ends.forEach((e, i) => { const k = eo((lt - s.c[0] - .8 - i * .4) / .8); e.n.m.scale.setScalar(.5 * k + 1e-4); e.n.rg.scale.setScalar(.5 * k + 1e-4); setOp(e.L, k); setOp(e.b, k); }); spin(s.ends.map(e => e.n), lt);
    s.guards.forEach((L, i) => setOp(L, sm(s.c[1] + .3 + i * .5, s.c[1] + .9 + i * .5, lt))); setOp(s.shield, sm(s.c[1], s.c[1] + 1, lt)); s.shield.rotation.z = lt * .1;
  },
  cam(lt) { return orbit(lt, {c: V(.8, .4, 0), r: 24, h: 1, a0: -.05, w: .006}); }});

// timeline — 46 commits
const COMMITS = ['08-24', '08-24', '09-11', '09-13', '09-23', '09-23', '09-24', '09-25', '09-26', '09-26', '09-28', '09-28', '09-28', '09-30', '10-03', '10-03', '10-10', '10-10', '10-10', '10-10', '10-10', '10-10', '10-10', '10-10', '10-10', '10-10', '10-10', '10-10', '10-23', '10-23', '10-24', '10-24', '10-24', '10-24', '10-25', '10-25', '10-25', '10-25', '10-25', '10-25', '10-29', '10-29', '10-29', '10-29', '10-29', '10-29'];
const MILESTONES = {'08-24': 'the search', '09-11': 'held-out judging', '09-23': 'any provider', '09-28': 'the audit', '10-03': 'paid runs', '10-10': 'tokens counted once', '10-23': 'routing · replication · autopsy', '10-29': 'pinned CI'};
def('timeline', {shift: .2,
  build(g, s) {
    const day = d => { const [m, dd] = d.split('-').map(Number); return (m === 8 ? dd : m === 9 ? 31 + dd : 61 + dd) - 24; };
    const at = t => V(-6.5 + t * 15, Math.sin(t * Math.PI * 1.5) * 1.4, -Math.sin(t * Math.PI) * 3);
    const pos = d => at(day(d) / 66);
    const count = {}; s.dots = COMMITS.map((d, i) => { const k = (count[d] = (count[d] || 0) + 1) - 1; const p = pos(d).add(V(0, .3 + k * .36, 0)); const m = new THREE.Mesh(SPH_LO, glow(MILESTONES[d] && k === 0 ? C.gold : C.cyan, 1.6)); m.position.copy(p); g.add(m); return {m, p, i}; });
    const pts = []; for (let i = 0; i <= 120; i++) pts.push(at(i / 120));
    s.path = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 200, .04, 8, false), glow(C.edge, .7)); g.add(s.path);
    s.ml = Object.entries(MILESTONES).map(([d, t], i) => { const top = COMMITS.filter(c => c === d).length * .36 + .3; const L = label(t, {size: .36, color: CSS.gold, weight: 700, sub: d.replace('-', ' / '), subScale: .7}); L.position.copy(pos(d)).add(V(0, i % 2 ? -1.3 : top + 1.0, 0)); g.add(L); return {L, i}; });
  },
  update(lt, s) {
    const t0 = s.c[0], t1 = s.c[1] + 2.5; s.dots.forEach(d => { const k = eo((lt - t0 - d.i / 46 * (t1 - t0)) / .5); d.m.scale.setScalar(.15 * k + 1e-4); });
    setOp(s.path, sm(t0 - .5, t0 + .5, lt)); s.ml.forEach(m => setOp(m.L, sm(s.c[1] + m.i * .3, s.c[1] + .5 + m.i * .3, lt)));
  },
  cam(lt) { return orbit(lt, {c: V(1, 1.2, -1), r: 25, h: 2.5, a0: -.04, w: .008}); }});

// limits
def('limits', {shift: .2,
  build(g, s) {
    const pts = [0, 1, 2].map(i => { const a = Math.PI / 2 + i * 2 * Math.PI / 3; return V(Math.cos(a) * 4.4, Math.sin(a) * 4.4 - .6, 0); });
    s.pr = new THREE.Mesh(new THREE.OctahedronGeometry(1.15, 0), glow(C.teal, .9)); s.pr.position.copy(pts[0]); g.add(s.pr);
    s.fi = new THREE.Mesh(SPH, glow(C.gold, .75)); s.fi.position.copy(pts[1]); g.add(s.fi);
    s.px = new THREE.Mesh(new THREE.IcosahedronGeometry(1.35, 1), new THREE.MeshBasicMaterial({color: C.coral, wireframe: true, transparent: true, opacity: .75})); s.px.position.copy(pts[2]); g.add(s.px);
    const L = [label('Predictor', {size: .55, weight: 700, color: CSS.teal, sub: 'one per outcome · learned', subScale: .66}), label('Fitness', {size: .55, weight: 700, color: CSS.gold, sub: 'a fixed formula', subScale: .66}), label('Prescriptor', {size: .55, weight: 700, color: CSS.coral, sub: 'not learned here', subScale: .66})];
    L[0].position.copy(pts[0]).add(V(0, 2.2, 0)); L[1].position.copy(pts[1]).add(V(-.3, -2.2, 0)); L[2].position.copy(pts[2]).add(V(.3, -2.3, 0)); L.forEach(l => g.add(l));
    const em = glow(C.edge, .6); g.add(beam(pts[0], pts[1], .04, em)); const dm = new THREE.MeshBasicMaterial({color: C.coral, transparent: true, opacity: .35}); g.add(beam(pts[1], pts[2], .03, dm)); g.add(beam(pts[2], pts[0], .03, dm));
    s.f = flow(g, pts[0], pts[1], {n: 3, speed: .5}); s.ops = label('7 operators + Pareto selection', {size: .42, mono: true, weight: 400, color: CSS.ink}); s.ops.position.copy(pts[2]).add(V(.3, -3.45, 0)); g.add(s.ops);
  },
  update(lt, s) { s.g.rotation.y = Math.sin(lt * .25) * .3; s.pr.rotation.set(lt * .4, lt * .6, 0); s.px.rotation.set(lt * .2, lt * .3, 0); s.px.material.opacity = .45 + .35 * Math.abs(Math.sin(lt * 2.3)); s.f(lt); setOp(s.ops, sm(s.c[0] + 1.4, s.c[0] + 2.2, lt)); },
  cam(lt) { return orbit(lt, {c: V(0, 0, 0), r: 24, h: 1, a0: -.15, w: .02}); }});

// prior art
def('prior', {shift: .2,
  build(g, s) {
    const O = V(1.2, .9, 0); s.core = node(g, O, C.teal, {r: .8}); const cl = label('neuro-san-esp', {size: .48, color: CSS.teal, weight: 700}); cl.position.copy(O).add(V(0, 1.6, 0)); g.add(cl);
    const names = ['ADAS', 'AFlow', 'GPTSwarm', 'MaAS', 'AgentSquare', 'EvoMAS', 'Promptbreeder'];
    s.n = names.map((t, i) => { const a = i / names.length * Math.PI * 2 + .45; const p = V(O.x + Math.cos(a) * 5.4, O.y + Math.sin(a) * 3.3, Math.sin(a * 2) * 1.2); const hot = t === 'AgentSquare';
      const n = node(g, p, hot ? C.gold : C.violet, {r: hot ? .5 : .36, ring: hot}); const L = label(t, {size: hot ? .46 : .38, color: hot ? CSS.gold : CSS.lilac, weight: 700}); L.position.copy(p).add(V(0, .95, 0)); g.add(L); return {n, L, p, hot}; });
    const hot = s.n.find(n => n.hot); s.hb = beam(hot.p, O, .035, glow(C.gold, 1.2)); g.add(s.hb);
    s.hl = label('a performance predictor, first', {size: .34, color: CSS.gold}); s.hl.position.copy(hot.p).add(V(0, -.9, 0)); g.add(s.hl);
    s.rays = [['a fitness function for neuro-san', V(-2.6, -3.7, 1.5)], ['a service on a budget', V(1.2, -4.5, 1.8)], ['defects only running it found', V(5.2, -3.7, 1.5)]].map(([t, p]) => { const b = beam(V(O.x, O.y - .6, 0), p, .03, glow(C.teal, 1.2)); g.add(b); const L = label(t, {size: .36, color: CSS.teal, weight: 700}); L.position.copy(p).add(V(0, -.6, 0)); g.add(L); return {b, L}; });
  },
  update(lt, s) {
    s.n.forEach((n, i) => { const k = eo((lt - s.c[0] - .5 - i * .25) / .7); n.n.m.scale.setScalar(n.n.r * k + 1e-4); if (n.n.rg) n.n.rg.scale.setScalar(n.n.r * k + 1e-4); setOp(n.L, k); }); spin([s.core, ...s.n.map(n => n.n)], lt);
    setOp(s.hb, sm(s.c[0] + 3, s.c[0] + 3.8, lt)); setOp(s.hl, sm(s.c[0] + 3.4, s.c[0] + 4.2, lt));
    s.rays.forEach((r, i) => { const k = sm(s.c[1] + 2 + i * .8, s.c[1] + 2.8 + i * .8, lt); setOp(r.b, k); setOp(r.L, k); }); s.g.rotation.y = Math.sin(lt * .2) * .1;
  },
  cam(lt) { return orbit(lt, {c: V(1, -.2, 0), r: 24, h: 1.5, a0: .08, w: -.008}); }});

// next
def('next', {shift: .2,
  build(g, s) {
    const P = [['same-budget experiment', 'built · runs on GitHub Actions', C.gold], ['a learned Prescriptor', 'first, decide the context', C.teal], ['resample one agent', 'not the whole network', C.cyan], ['a second world', 'to test transfer', C.lilac]];
    s.p = P.map(([t, sub, c], i) => { const grp = new THREE.Group(); grp.position.set(-.8 + (i % 2) * 6.2, 2.7 - Math.floor(i / 2) * 5.6, 0); g.add(grp); const ring = new THREE.Mesh(new THREE.TorusGeometry(1.45, .08, 12, 100), glow(c, 1.3)); grp.add(ring);
      const inner = new THREE.Mesh(new THREE.CircleGeometry(1.35, 64), new THREE.MeshBasicMaterial({color: c, transparent: true, opacity: .08, side: THREE.DoubleSide})); grp.add(inner);
      const L = label(t, {size: .42, weight: 700, color: hex(c), sub, subScale: .66}); L.position.set(0, -2.25, 0); grp.add(L); return {grp, ring}; });
    s.prog = new THREE.Mesh(new THREE.TorusGeometry(1.8, .045, 8, 100, Math.PI * 1.2), glow(C.gold, 1.6)); s.p[0].grp.add(s.prog);
  },
  update(lt, s) {
    s.p.forEach((p, i) => { const t = i === 0 ? s.c[0] + .6 : s.c[1] + (i - 1) * .7; const k = eo((lt - t) / .9); p.grp.scale.setScalar(k + 1e-4); p.ring.rotation.z = lt * .3 * (i % 2 ? -1 : 1); });
    s.prog.rotation.z = -lt * 1.2;
  },
  cam(lt) { return orbit(lt, {c: V(2.2, -.2, 0), r: 23, h: 1.2, a0: -.12, w: .01}); }});

// numbers
def('numbers', {shift: 0, shade: 'full',
  build(g, s) {
    const R = mulberry(77); const n = 2600; const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { const u = R() * 2 - 1, th = R() * Math.PI * 2, rr = 12 + R() * 14; const q = Math.sqrt(1 - u * u); pos.set([Math.cos(th) * q * rr, u * rr * .7, Math.sin(th) * q * rr], i * 3); const c = new THREE.Color([C.cyan, C.teal, C.violet, C.gold][i % 4]); col.set([c.r, c.g, c.b], i * 3); }
    const gg = new THREE.BufferGeometry(); gg.setAttribute('position', new THREE.BufferAttribute(pos, 3)); gg.setAttribute('color', new THREE.BufferAttribute(col, 3));
    s.pts = new THREE.Points(gg, new THREE.PointsMaterial({size: .3, map: dotTex, vertexColors: true, transparent: true, opacity: .8, depthWrite: false, blending: THREE.AdditiveBlending})); g.add(s.pts);
  },
  update(lt, s) { s.pts.rotation.y = lt * .05; s.pts.rotation.x = .2; },
  cam(lt, s) { return {p: V(0, 2, lerp(40, 30, eio(lt / s.dur))), l: V(0, 0, 0)}; }});

// end
def('end', {shift: .2,
  build(g, s) {
    const A = V(0, 2.6, 0), B = V(0, -2.2, 0); s.n = [node(g, A, C.teal, {r: 1.0, ei: 1.2}), node(g, B, C.gold, {r: 1.1, ei: 1.3})]; const em = glow(C.edge, .7); g.add(beam(A, B, .07, em)); s.f = flow(g, A, B, {n: 4, speed: .5, size: .16});
    const a = label('Coordinator', {size: .55, weight: 700, color: CSS.teal, sub: 'gpt-5.4-nano', subScale: .7}); a.position.set(0, 5.1, 0); g.add(a);
    const b = label('Researcher', {size: .55, weight: 700, color: CSS.gold, sub: 'gpt-5.4-mini · 10 documents a search', subScale: .7}); b.position.set(0, -6.6, 0); g.add(b);
    const R = mulberry(31); s.docs = []; const dm = [glow(C.cyan, .35), glow(C.gold, .3), glow(C.coral, .3)];
    for (let i = 0; i < 124; i++) { const t = i < 24 ? 0 : i < 64 ? 1 : 2; const m = new THREE.Mesh(BOX, dm[t]); m.scale.set(.22, .3, .04); g.add(m); s.docs.push({m, a: R() * Math.PI * 2, r: 3.2 + R() * 1.8, y: (R() - .5) * 1.0, sp: .15 + R() * .2}); }
  },
  update(lt, s) { spin(s.n, lt); s.f(lt); s.docs.forEach(d => { const a = d.a + lt * d.sp; d.m.position.set(Math.cos(a) * d.r, -2.2 + d.y + Math.sin(a * 2) * .15, Math.sin(a) * d.r); d.m.rotation.y = -a; }); },
  cam(lt, s) { return orbit(lt, {c: V(0, -.6, 0), r: lerp(22, 26, eio(lt / s.dur)), h: 3, a0: .3, w: .025}); }});

// ---------- frame ----------
const missing = TL.scenes.filter(t => !SCENES.find(s => s.id === t.id)).map(t => t.id);
if (missing.length) throw new Error('scenes in timeline.js but not in film.js: ' + missing.join(', '));
window.TOTAL = TL.total; window.FPS = 30; window.SCENE_STARTS = SCENES.map(s => s.start); window.CHAPTER_STARTS = SCENES.filter((s, i) => i === 0 || s.chap !== SCENES[i - 1].chap).map(s => s.start);
const hud = document.getElementById('hud'), dip = document.getElementById('dip'), shade = document.getElementById('shade'), clock = document.getElementById('clock');
const segs = [...document.querySelectorAll('.seg')];
const chapRange = [1, 2, 3, 4, 5].map(c => { const ss = SCENES.filter(s => s.chap === c); return [ss[0].start, ss.at(-1).start + ss.at(-1).dur]; });
const fmt = (v, dec) => (dec ? v.toFixed(dec) : Math.round(v).toLocaleString('en-US'));
const mmss = t => `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(Math.floor(t % 60)).padStart(2, '0')}`;

// A scene's large textures are released when the film leaves it, so a browser holds about one
// scene's worth at 7680x4320; three.js uploads them again if the scene is drawn later.
let shown = null;
function release(group) { group.traverse(o => { const m = o.material; if (m && m.map && m.map.image && m.map.image.width > 512) m.map.dispose(); }); }
window.renderAt = t => {
  const s = SCENES.find(x => t >= x.start && t < x.start + x.dur) || SCENES.at(-1); const lt = t - s.start;
  if (shown && shown !== s) release(shown.g);
  shown = s;
  for (const o of SCENES) { o.g.visible = o === s; o.el.style.opacity = o === s ? String(1 - sm(s.dur - .6, s.dur - .05, lt)) : '0'; }
  for (const {e, t: t0} of s.items) { const p = eo((lt - t0) / .75); e.style.opacity = p; e.style.transform = `translateY(${(1 - p) * 40}px)`; }
  for (const {e, t: t0} of s.counts) { const p = eo((lt - t0) / 2.2); e.style.opacity = 1; e.textContent = fmt(parseFloat(e.dataset.count) * p, parseInt(e.dataset.dec || 0)); }
  shade.className = s.shade;
  s.update && s.update(lt, s);
  const cm = s.cam(lt, s); camera.position.copy(cm.p); camera.lookAt(cm.l);
  camera.setViewOffset(W, H, -W * s.shift, 0, W, H); camera.updateProjectionMatrix();
  let d = Math.max(1 - sm(0, .55, lt), sm(s.dur - .5, s.dur, lt));
  if (s === SCENES[0]) d = Math.max(1 - sm(0, 1.8, lt), sm(s.dur - .5, s.dur, lt));
  if (s === SCENES.at(-1)) d = Math.max(1 - sm(0, .55, lt), sm(s.dur - 2.2, s.dur - .15, lt));
  dip.style.opacity = d;
  const h1 = SCENES[1].start, hN = SCENES.find(x => x.id === 'numbers').start; hud.style.opacity = sm(h1 - .2, h1 + 1, t) * (1 - sm(hN - .4, hN + .4, t));
  segs.forEach((sg, i) => { const [a, b] = chapRange[i]; sg.querySelector('b').style.width = (clamp((t - a) / (b - a)) * 100) + '%'; sg.classList.toggle('on', t >= a && t < b); });
  clock.textContent = `${mmss(t)} / ${mmss(TL.total)}`;
  stars.rotation.y = t * .006; stars.rotation.x = Math.sin(t * .04) * .04;
  composer.render();
  return true;
};
if (QP.has('t')) window.renderAt(parseFloat(QP.get('t')));
window.ready = true;
