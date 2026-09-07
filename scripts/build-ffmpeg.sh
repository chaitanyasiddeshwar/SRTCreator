#!/usr/bin/env bash
# Minimized, static, audio-only FFmpeg build for SRTCreator (LGPL scope).
#
# This is the OPTIMIZATION pass that replaces the bootstrap dev/shared libs with
# a lean static build linked straight into srt.exe. Run from an MSYS2 shell that
# has the MSVC environment loaded (so cl.exe is on PATH), then point CMake's
# FFMPEG_DIR at ../ffmpeg-min.
#
# Prereqs: MSYS2 (pacman -S make nasm), and VsDevCmd already applied in the shell.
set -euo pipefail

SRC="${1:-third_party/ffmpeg-src}"   # FFmpeg source checkout
OUT="$(pwd)/third_party/ffmpeg-min"  # install prefix (become the new FFMPEG_DIR)

cd "$SRC"

./configure \
  --toolchain=msvc \
  --prefix="$OUT" \
  --disable-everything --disable-programs --disable-doc \
  --enable-static --disable-shared \
  --enable-protocol=file \
  --enable-demuxer=matroska,mov,mpegts,avi,flv,ogg,wav,mp3,flac,aac,ac3,dts \
  --enable-decoder=aac,aac_latm,ac3,eac3,dca,truehd,mlp,mp3,mp2,opus,vorbis,flac,pcm_s16le,pcm_s24le,pcm_f32le,alac,wmav1,wmav2 \
  --enable-parser=aac,ac3,dca,flac,mpegaudio,opus,vorbis \
  --enable-filter=aresample \
  --disable-encoders --disable-muxers --disable-bsfs \
  --disable-network --disable-avdevice --disable-postproc --disable-swscale

make -j"$(nproc)"
make install

echo "Done. Set CMake FFMPEG_DIR (or third_party/ffmpeg) to: $OUT"
