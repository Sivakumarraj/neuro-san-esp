#!/bin/sh
# sheet.sh OUT a.png b.png c.png d.png  -> a 2x2 contact sheet at 1920x1080
out=$1; shift
ffmpeg -loglevel error -y -i "$1" -i "$2" -i "$3" -i "$4" -filter_complex \
 "[0]scale=960:540[a];[1]scale=960:540[b];[2]scale=960:540[c];[3]scale=960:540[d];[a][b][c][d]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0" "$out"
