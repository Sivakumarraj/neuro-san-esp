'use strict';
// Render the film at 3840x2160, 30 fps.
//
//   node render.js all [WORKERS]          every frame, split across WORKERS processes, then
//                                         joined, scored and written to out/ (default 3)
//   node render.js frames FROM TO OUT     one worker: frames [FROM, TO) piped into ffmpeg
//   node render.js stills T [T ...]       PNG stills at T seconds, written to out/
//
// Chromium draws WebGL in software (SwiftShader), so no GPU is needed and every machine
// renders the same frames. film.html draws any frame from its time alone, which is what
// lets the workers take the film in pieces.

const {chromium} = require('playwright');
const http = require('http');
const fs = require('fs');
const path = require('path');
const {spawn} = require('child_process');

const ROOT = __dirname;
const OUT = path.join(ROOT, 'out');
const FPS = 30;
const W = 3840;
const H = 2160;
const CHROME_ARGS = ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];
const TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript'};

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

async function openFilm() {
  const server = await serve();
  const browser = await chromium.launch({args: CHROME_ARGS});
  const page = await browser.newPage({viewport: {width: W, height: H}, deviceScaleFactor: 1});
  page.on('pageerror', err => { console.error('page error:', err.message); process.exitCode = 1; });
  await page.goto(`http://127.0.0.1:${server.address().port}/film.html`);
  await page.waitForFunction('window.ready === true', null, {timeout: 180000});
  const {total, cuts} = await page.evaluate(() => ({total: window.TOTAL, cuts: window.SCENE_STARTS}));
  return {page, total, cuts, close: async () => { await browser.close(); server.close(); }};
}

function run(cmd, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, {stdio: 'inherit'});
    child.on('close', code => (code === 0 ? resolve() : reject(new Error(`${cmd} exited with ${code}`))));
  });
}

async function frames(from, to, out) {
  const film = await openFilm();
  const ffmpeg = spawn('ffmpeg', [
    '-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p',
    '-profile:v', 'high', '-level', '5.1', out], {stdio: ['pipe', 'inherit', 'inherit']});
  const encoded = new Promise(resolve => ffmpeg.on('close', resolve));
  const started = Date.now();
  for (let f = from; f < to; f++) {
    await film.page.evaluate(t => window.renderAt(t), f / FPS);
    const jpeg = await film.page.screenshot({type: 'jpeg', quality: 94});
    if (!ffmpeg.stdin.write(jpeg)) await new Promise(resolve => ffmpeg.stdin.once('drain', resolve));
    const done = f - from + 1;
    if (done % 150 === 0 || f === to - 1) {
      const each = (Date.now() - started) / 1000 / done;
      console.log(`${path.basename(out)}: ${done}/${to - from} frames, ${each.toFixed(2)} s a frame`);
    }
  }
  ffmpeg.stdin.end();
  const code = await encoded;
  await film.close();
  if (code !== 0) throw new Error(`ffmpeg exited with ${code} for ${out}`);
}

async function stills(times) {
  fs.mkdirSync(OUT, {recursive: true});
  const film = await openFilm();
  for (const t of times) {
    await film.page.evaluate(s => window.renderAt(s), t);
    await film.page.screenshot({path: path.join(OUT, `still-${t}.png`)});
  }
  await film.close();
}

async function all(workers) {
  fs.mkdirSync(OUT, {recursive: true});
  const film = await openFilm();
  const {total, cuts} = film;
  await film.close();
  const count = Math.round(total * FPS);
  const per = Math.ceil(count / workers);
  const parts = [];
  const jobs = [];
  for (let k = 0; k * per < count; k++) {
    const part = path.join(OUT, `part-${k}.mp4`);
    parts.push(part);
    jobs.push(run(process.execPath, [__filename, 'frames', String(k * per), String(Math.min(count, (k + 1) * per)), part]));
  }
  await Promise.all(jobs);

  const list = path.join(OUT, 'parts.txt');
  fs.writeFileSync(list, parts.map(p => `file '${path.basename(p)}'`).join('\n') + '\n');
  const silent = path.join(OUT, 'film-silent.mp4');
  await run('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', silent]);

  const music = path.join(OUT, 'music.wav');
  await run('python3', [path.join(ROOT, 'music.py'), music, String(total), cuts.slice(1).join(',')]);

  const film4k = path.join(OUT, 'neuro-san-esp-4k.mp4');
  await run('ffmpeg', ['-y', '-loglevel', 'error', '-i', silent, '-i', music, '-map', '0:v', '-map', '1:a',
    '-c:v', 'copy', '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11', '-c:a', 'aac', '-b:a', '256k', '-ar', '48000',
    '-shortest', '-movflags', '+faststart', film4k]);
  const film1080 = path.join(OUT, 'neuro-san-esp-1080p.mp4');
  await run('ffmpeg', ['-y', '-loglevel', 'error', '-i', film4k, '-vf', 'scale=1920:1080:flags=lanczos',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'copy',
    '-movflags', '+faststart', film1080]);
  console.log(`wrote ${film4k}\nwrote ${film1080}`);
}

async function main() {
  const [mode, ...args] = process.argv.slice(2);
  if (mode === 'all') return all(parseInt(args[0] || '3', 10));
  if (mode === 'frames' && args.length === 3) return frames(parseInt(args[0], 10), parseInt(args[1], 10), args[2]);
  if (mode === 'stills' && args.length) return stills(args.map(Number));
  console.error('usage: node render.js all [WORKERS] | frames FROM TO OUT | stills T [T ...]');
  process.exitCode = 2;
}

main().catch(err => { console.error(err); process.exit(1); });
