#!/bin/sh
# Render every scene to out/scenes/<id>.mp4, skipping ones already done, then
# join them, add the music bed and normalise loudness into out/neuro-san-esp-explained.mp4.
set -e
cd "$(dirname "$0")/.."
HS=/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell
mkdir -p out/scenes
for id in $(python3 -c "import json;print(' '.join(s['id'] for s in json.load(open('src/timing.json'))['scenes']))"); do
  comp="Scene-$(echo $id | tr '_' '-')"
  if [ -s "out/scenes/$id.mp4" ]; then echo "skip $id"; continue; fi
  echo "render $id $(date +%T)"
  npx remotion render src/index.ts "$comp" "out/scenes/$id.part.mp4" --browser-executable=$HS --gl=swangle --concurrency=4 --log=error
  mv "out/scenes/$id.part.mp4" "out/scenes/$id.mp4"
done
: > out/scenes/list.txt
for id in $(python3 -c "import json;print(' '.join(s['id'] for s in json.load(open('src/timing.json'))['scenes']))"); do
  echo "file '$id.mp4'" >> out/scenes/list.txt
done
ffmpeg -y -loglevel error -f concat -safe 0 -i out/scenes/list.txt -c copy out/joined.mp4
ffmpeg -y -loglevel error -i out/joined.mp4 -i public/audio/music.mp3 -filter_complex \
  "[0:a]loudnorm=I=-16:TP=-1.5:LRA=11[v];[1:a]volume=0.09[m];[v][m]amix=inputs=2:duration=first:normalize=0[a]" \
  -map 0:v -map "[a]" -c:v libx264 -preset slow -crf 21 -pix_fmt yuv420p -c:a aac -b:a 160k -movflags +faststart \
  out/neuro-san-esp-explained.mp4
echo "done $(date +%T)"
