// Render chosen frames of chosen scenes to PNG, bundling once.
//   node tools/stills.mjs s01_hook:420 s02_agent:0.5 ...
// A frame below 1 is a fraction of the scene's length.
import {bundle} from '@remotion/bundler';
import {renderStill, selectComposition} from '@remotion/renderer';
import path from 'node:path';
import fs from 'node:fs';

const browserExecutable = '/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell';
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const serveUrl = await bundle({entryPoint: path.join(root, 'src/index.ts')});
fs.mkdirSync(path.join(root, 'out/stills'), {recursive: true});

for (const arg of process.argv.slice(2)) {
  const [scene, at] = arg.split(':');
  const id = `Scene-${scene.replaceAll('_', '-')}`;
  const composition = await selectComposition({serveUrl, id, browserExecutable, chromiumOptions: {gl: 'swangle'}});
  const n = Number(at);
  const frame = n < 1 ? Math.floor(n * composition.durationInFrames) : n;
  const output = path.join(root, `out/stills/${scene}_${frame}.png`);
  await renderStill({serveUrl, composition, frame, output, browserExecutable, chromiumOptions: {gl: 'swangle'}});
  console.log(output);
}
