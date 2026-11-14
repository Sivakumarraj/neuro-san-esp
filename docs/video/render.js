'use strict';
// Render the film at 3840x2160, or 7680x4320 with --scale 2, at 30 fps.
//
//   node render.js all [--workers N] [--scale S] [--gl swiftshader|llvmpipe]
//       render every frame in 5-second chunks shared between N browser workers (default 3),
//       encode each finished chunk, then assemble: join the chunks, lay the music under the
//       narration, and write out/neuro-san-esp-<4k|8k>.mp4 and a 1080p copy. Anything already
//       rendered or encoded is kept, so an interrupted run picks up where it stopped.
//   node render.js assemble [--scale S]      only the last step
//   node render.js stills T [T ...] [--scale S] [--gl G]   PNG stills at T seconds, in out/
//
// Before `all`, write the narration and the timeline it sets (needs Kokoro, see README):
//   python3 narration.py out/narration.wav timeline.js
//
// film.html draws any frame from its time alone, which is what lets the workers take the film
// in pieces. --scale 2 keeps the layout and doubles the pixels. Browsers write JPEG frames and
// one encoder turns each finished chunk into video, so memory stays bounded at 7680x4320.
//
// --gl picks Chromium's WebGL backend. Both are software, need no GPU and draw the same frames.
// swiftshader ships with Chromium; llvmpipe is Mesa's, needs Mesa's EGL (libegl-mesa0 on Debian
// and Ubuntu), and is several times faster at 7680x4320.

const {chromium} = require('playwright');
const http = require('http');
const fs = require('fs');
const path = require('path');
const {spawn, spawnSync} = require('child_process');

const ROOT = __dirname;
const OUT = path.join(ROOT, 'out');
const FPS = 30;
const CHUNK = 150;
const W = 3840;
const H = 2160;
const PYTHON = process.env.PYTHON || 'python3';
const TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript'};
const BACKENDS = {
  swiftshader: {args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'], env: {}},
  llvmpipe: {args: ['--use-gl=angle', '--use-angle=gl-egl', '--ignore-gpu-blocklist', '--enable-gpu'],
    env: {EGL_PLATFORM: 'surfaceless', GALLIUM_DRIVER: 'llvmpipe'}},
};
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function option(args, name, fallback) {
  const i = args.indexOf(name);
  if (i < 0) return fallback;
  const value = args[i + 1];
  args.splice(i, 2);
  return value;
}

function serve() {
  const server = http.createServer((req, res) => {
    const file = path.join(ROOT, path.normalize(decodeURIComponent(req.url.split('?')[0])));
    if (!file.startsWith(ROOT + path.sep)) { res.writeHead(403); res.end(); return; }
    fs.readFile(file, (err, body) => {
      if (err) { res.writeHead(404); res.end(); return; }
      res.writeHead(200, {'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream'});
      res.end(body);
    });
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

async function openFilm(scale, gl) {
  const backend = BACKENDS[gl];
  if (!backend) throw new Error(`unknown --gl ${gl}; use swiftshader or llvmpipe`);
  const server = await serve();
  const browser = await chromium.launch({args: backend.args, env: {...process.env, ...backend.env}});
  const page = await browser.newPage({viewport: {width: W, height: H}, deviceScaleFactor: scale});
  page.on('pageerror', err => { console.error('page error:', err.message); process.exitCode = 1; });
  const url = `http://127.0.0.1:${server.address().port}/film.html?dpr=${scale}`;
  const load = async () => { await page.goto(url); await page.waitForFunction('window.ready === true', null, {timeout: 300000}); };
  await load();
  const info = await page.evaluate(() => ({total: window.TOTAL, cuts: window.SCENE_STARTS, chapters: window.CHAPTER_STARTS}));
  return {page, ...info, reload: load, close: async () => { await browser.close(); server.close(); }};
}

function run(cmd, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, {stdio: 'inherit'});
    child.on('close', code => (code === 0 ? resolve() : reject(new Error(`${cmd} exited with ${code}`))));
  });
}

function framesIn(file) {
  if (!fs.existsSync(file)) return 0;
  const probe = spawnSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-count_packets',
    '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', file], {encoding: 'utf8'});
  return parseInt(probe.stdout, 10) || 0;
}

const width = scale => W * scale;
const chunkFile = (scale, k) => path.join(OUT, `chunks-${width(scale)}`, `chunk-${String(k).padStart(4, '0')}.mp4`);
const frameDir = (scale, k) => path.join(OUT, `frames-${width(scale)}`, `chunk-${String(k).padStart(4, '0')}`);
const span = (count, k) => [k * CHUNK, Math.min(count, (k + 1) * CHUNK)];

// One worker: render chunks first, first + step, ... as numbered JPEG frames.
async function chunks(first, step, scale, gl) {
  const film = await openFilm(scale, gl);
  const count = Math.round(film.total * FPS);
  for (let k = first; k * CHUNK < count; k += step) {
    const [from, to] = span(count, k);
    if (framesIn(chunkFile(scale, k)) === to - from) continue;
    const dir = frameDir(scale, k);
    if (fs.existsSync(path.join(dir, '.done'))) continue;
    fs.mkdirSync(dir, {recursive: true});
    const started = Date.now();
    for (let f = from; f < to; f++) {
      const file = path.join(dir, `f${String(f - from).padStart(4, '0')}.jpg`);
      if (fs.existsSync(file)) continue;
      for (let attempt = 1; ; attempt++) {
        try {
          await film.page.evaluate(t => window.renderAt(t), f / FPS);
          const jpeg = await film.page.screenshot({type: 'jpeg', quality: 94});
          fs.writeFileSync(file + '.tmp', jpeg); fs.renameSync(file + '.tmp', file);
          break;
        } catch (err) {
          if (attempt === 3) throw err;
          console.error(`frame ${f}: ${err.message.split('\n')[0]}; reloading and retrying`);
          await sleep(2000); await film.reload();
        }
      }
    }
    fs.writeFileSync(path.join(dir, '.done'), String(to - from));
    console.log(`chunk ${k}: frames ${from}-${to - 1} rendered, ${((Date.now() - started) / 1000 / (to - from)).toFixed(2)} s a frame`);
  }
  await film.close();
}

// x264 at 7680x4320 keeps its look-ahead short so one encoder fits beside the browsers.
const encoder = scale => (scale > 1
  ? ['-c:v', 'libx264', '-preset', 'fast', '-crf', '17', '-threads', '2', '-x264-params', 'rc-lookahead=10', '-pix_fmt', 'yuv420p', '-profile:v', 'high']
  : ['-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level', '5.1']);

// Encode every chunk whose frames are complete; returns how many it encoded.
async function encodeReady(scale, count) {
  let encoded = 0;
  for (let k = 0; k * CHUNK < count; k++) {
    const [from, to] = span(count, k);
    const dir = frameDir(scale, k), done = chunkFile(scale, k);
    if (!fs.existsSync(path.join(dir, '.done'))) continue;
    if (framesIn(done) !== to - from) {
      fs.mkdirSync(path.dirname(done), {recursive: true});
      const partial = done.replace(/\.mp4$/, '.part.mp4');
      await run('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(FPS), '-i', path.join(dir, 'f%04d.jpg'), ...encoder(scale), partial]);
      if (framesIn(partial) !== to - from) throw new Error(`chunk ${k} encoded short`);
      fs.renameSync(partial, done);
      console.log(`chunk ${k}: encoded`);
      encoded++;
    }
    fs.rmSync(dir, {recursive: true, force: true});
  }
  return encoded;
}

async function assemble(scale, gl = 'swiftshader') {
  const film = await openFilm(1, gl);
  const {total, chapters} = film;
  await film.close();
  const count = Math.round(total * FPS);
  const parts = [];
  for (let k = 0; k * CHUNK < count; k++) {
    const [from, to] = span(count, k);
    if (framesIn(chunkFile(scale, k)) !== to - from) throw new Error(`chunk ${k} is missing; run all again`);
    parts.push(chunkFile(scale, k));
  }
  const list = path.join(OUT, `parts-${width(scale)}.txt`);
  fs.writeFileSync(list, parts.map(p => `file '${p}'`).join('\n') + '\n');
  const silent = path.join(OUT, `film-silent-${width(scale)}.mp4`);
  await run('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', silent]);

  const music = path.join(OUT, 'music.wav');
  await run(PYTHON, [path.join(ROOT, 'music.py'), music, String(total), chapters.slice(1).join(',')]);
  const voice = path.join(OUT, 'narration.wav');
  if (!fs.existsSync(voice)) throw new Error('out/narration.wav is missing: run python3 narration.py out/narration.wav timeline.js');
  // The voice ducks the music: the bed drops well under the words while someone is speaking.
  const mix = '[1:a]volume=0.5[bed];'
    + '[2:a]aresample=48000,pan=stereo|c0=c0|c1=c0,highpass=f=70,asplit=2[v][key];'
    + '[bed][key]sidechaincompress=threshold=0.02:ratio=8:attack=20:release=500[ducked];'
    + "[ducked][v]amix=inputs=2:weights='1 1.6':normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11[a]";
  const final = path.join(OUT, `neuro-san-esp-${scale > 1 ? '8k' : '4k'}.mp4`);
  await run('ffmpeg', ['-y', '-loglevel', 'error', '-i', silent, '-i', music, '-i', voice, '-filter_complex', mix,
    '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '320k', '-ar', '48000', '-shortest',
    '-movflags', '+faststart', final]);
  console.log(`wrote ${final}`);
  const small = path.join(OUT, 'neuro-san-esp-1080p.mp4');
  await run('ffmpeg', ['-y', '-loglevel', 'error', '-i', final, '-vf', 'scale=1920:1080:flags=lanczos',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart', small]);
  console.log(`wrote ${small}`);
}

async function all(workers, scale, gl) {
  fs.mkdirSync(OUT, {recursive: true});
  const film = await openFilm(1, gl);
  const count = Math.round(film.total * FPS);
  await film.close();
  let running = workers;
  const jobs = [];
  for (let k = 0; k < workers; k++) {
    jobs.push(run(process.execPath, [__filename, 'chunks', String(k), String(workers), '--scale', String(scale), '--gl', gl])
      .finally(() => { running--; }));
  }
  const failed = Promise.allSettled(jobs);
  while (true) {
    const n = await encodeReady(scale, count);
    if (n === 0 && running === 0) break;
    if (n === 0) await sleep(5000);
  }
  for (const r of await failed) if (r.status === 'rejected') throw r.reason;
  await encodeReady(scale, count);
  await assemble(scale, gl);
}

async function stills(times, scale, gl) {
  fs.mkdirSync(OUT, {recursive: true});
  const film = await openFilm(scale, gl);
  for (const t of times) {
    await film.page.evaluate(s => window.renderAt(s), t);
    await film.page.screenshot({path: path.join(OUT, `still-${t}${scale > 1 ? `-x${scale}` : ''}.png`)});
  }
  await film.close();
}

async function main() {
  const args = process.argv.slice(2);
  const scale = parseInt(option(args, '--scale', '1'), 10);
  const gl = option(args, '--gl', 'swiftshader');
  const workers = parseInt(option(args, '--workers', '3'), 10);
  const [mode, ...rest] = args;
  if (mode === 'all') return all(workers, scale, gl);
  if (mode === 'assemble') return assemble(scale, gl);
  if (mode === 'chunks' && rest.length === 2) return chunks(parseInt(rest[0], 10), parseInt(rest[1], 10), scale, gl);
  if (mode === 'stills' && rest.length) return stills(rest.map(Number), scale, gl);
  console.error('usage: node render.js all [--workers N] [--scale S] [--gl G] | assemble [--scale S]'
    + ' | stills T [T ...] [--scale S] [--gl G]');
  process.exitCode = 2;
}

main().catch(err => { console.error(err); process.exit(1); });
