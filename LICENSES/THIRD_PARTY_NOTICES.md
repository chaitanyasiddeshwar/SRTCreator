# Third-Party Notices — SRTCreator

SRTCreator ships and/or dynamically links the following third-party components.
Their licenses are reproduced or referenced below. Exact pinned versions are in
`dependencies.json`.

---

## whisper.cpp + ggml — MIT

ASR engine and tensor library (shipped as `whisper.dll`, `ggml.dll`,
`ggml-base.dll`, `ggml-cpu-*.dll`, and the optional GPU backends `ggml-cuda.dll` /
`ggml-vulkan.dll`).

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in the
Software without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the
following conditions: the above copyright notice and this permission notice shall
be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The full upstream license is bundled as `LICENSE-whisper.txt`.

---

## FFmpeg — LGPL-2.1-or-later

Demux/decode of any container/codec (shipped as `avformat-*.dll`, `avcodec-*.dll`,
`avutil-*.dll`, `swresample-*.dll`). We use the BtbN **LGPL** shared build — audio
path only, **no GPL components**.

FFmpeg is free software licensed under the GNU Lesser General Public License
(LGPL) version 2.1 or later. SRTCreator dynamically links the FFmpeg libraries;
the shipped `.dll` files are unmodified BtbN LGPL builds.

**Relink notice (LGPL §6):** because SRTCreator links FFmpeg dynamically, you may
replace the shipped FFmpeg DLLs with your own LGPL-compatible build of the same
soname (`avformat-63`, `avcodec-63`, `avutil-61`, `swresample-7`) and the
application will use them. FFmpeg source corresponding to the shipped build is
available from the FFmpeg project (https://ffmpeg.org) and the BtbN builds project
(https://github.com/BtbN/FFmpeg-Builds). The full LGPL text is bundled as
`LICENSE-ffmpeg.txt`.

---

## ONNX Runtime — MIT

Inference runtime for MDX-Net vocal isolation via the DirectML execution provider
(shipped as `onnxruntime.dll`).

Copyright (c) Microsoft Corporation. Licensed under the MIT License (same terms as
the whisper.cpp MIT block above).

---

## DirectML — Microsoft redistributable

GPU inference backend for ONNX Runtime on any Direct3D-12 GPU (shipped as
`DirectML.dll`). Redistributed under the Microsoft software license terms that
accompany the DirectML redistributable package
(https://www.nuget.org/packages/Microsoft.AI.DirectML).

---

## Microsoft Visual C++ Runtime — Microsoft redistributable

The base distribution ships the Visual C++ runtime app-locally (`MSVCP140.dll`,
`VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`, `VCOMP140.dll`) so end users do not need
to install the VC++ Redistributable separately.

Copyright (c) Microsoft Corporation. Redistributed under the Microsoft Visual C++
Redistributable terms, which permit app-local deployment of these runtime DLLs.

---

## NVIDIA CUDA Runtime — NVIDIA CUDA EULA (GPU pack only)

The optional NVIDIA GPU pack ships the CUDA runtime redistributables
(`cudart64_*.dll`, `cublas64_*.dll`, `cublasLt64_*.dll`) alongside `ggml-cuda.dll`.
These are redistributed under the NVIDIA CUDA Toolkit End User License Agreement,
which permits redistribution of the listed runtime libraries
(https://docs.nvidia.com/cuda/eula/). End users need only an NVIDIA driver; the
CUDA Toolkit is not required.

---

## Models (not bundled)

Whisper GGUF models and MDX-Net (UVR Kim_Vocal_2 etc.) weights are **downloaded on
first run**, not shipped in the distribution. They carry their own licenses from
their respective sources (Hugging Face / UVR).
