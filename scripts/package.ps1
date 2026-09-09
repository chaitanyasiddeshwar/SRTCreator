<#
.SYNOPSIS
    Assemble the tiered SRTCreator distribution (modular_proposal.md 3.3):

      SRTCreator-v<ver>-win64-base.zip        universal, runs on ANY win64
        srt.exe, srtgui.exe, whisper.dll, ggml(.dll/-base/-cpu-*),
        FFmpeg + onnxruntime + DirectML, VC++ runtime, docs + LICENSES.
        CPU transcription + DirectML vocal isolation. No GPU transcription.

      SRTCreator-v<ver>-win64-cuda-pack.zip   NVIDIA power pack (~500 MB)
        ggml-cuda.dll + cudart/cublas/cublasLt. Unzip next to srt.exe.

      SRTCreator-v<ver>-win64-vulkan-pack.zip universal GPU pack (~few MB)
        ggml-vulkan.dll. Unzip next to srt.exe. Any modern GPU, driver only.

    A GPU pack is additive: ggml_backend_load_all() detects the dropped-in DLL and
    the app auto-selects it (CUDA -> Vulkan -> CPU). Pull everything from build\,
    which already staged the exact runtime set. Requires a prior successful build.

.PARAMETER Version  Distribution version (default: dependencies.json app.version).
.PARAMETER Root     Repo root (default: parent of this script's dir).
#>
param(
    [string]$Version = "",
    [string]$Root    = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = "Stop"

if (-not $Version) {
    $Version = (Get-Content (Join-Path $Root "dependencies.json") -Raw | ConvertFrom-Json).app.version
}
$build = Join-Path $Root "build"
if (-not (Test-Path (Join-Path $build "srt.exe"))) { throw "build\srt.exe not found. Run build.bat first." }

$dist = Join-Path $Root "dist"
if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
New-Item -ItemType Directory -Force -Path $dist | Out-Null

# Copy files matching any of $patterns from $srcDir into $destDir (must find each).
function Copy-Set([string]$srcDir, [string[]]$patterns, [string]$destDir) {
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    foreach ($pat in $patterns) {
        $hits = Get-ChildItem -Path (Join-Path $srcDir $pat) -ErrorAction SilentlyContinue
        if (-not $hits) { throw "package: no match for '$pat' in $srcDir" }
        $hits | ForEach-Object { Copy-Item $_.FullName $destDir -Force }
    }
}
function Zip([string]$stageDir, [string]$zipName) {
    $zip = Join-Path $dist $zipName
    Compress-Archive -Path (Join-Path $stageDir '*') -DestinationPath $zip -Force
    "{0,-42} {1,7:N1} MB" -f $zipName, ((Get-Item $zip).Length/1MB)
}

# ---------------- base ----------------
$baseStage = Join-Path $dist "SRTCreator"
Copy-Set $build @("srt.exe","srtgui.exe") $baseStage
# whisper + ggml core + ALL cpu variants (no cuda/vulkan)
Copy-Set $build @("whisper.dll","ggml.dll","ggml-base.dll","ggml-cpu-*.dll") $baseStage
# FFmpeg (the four the exe links) + ONNX + DirectML
Copy-Set $build @("avformat-*.dll","avcodec-*.dll","avutil-*.dll","swresample-*.dll",
                  "onnxruntime.dll","DirectML.dll") $baseStage
# VC++ runtime (app-local, avoids requiring the VC++ Redist install)
$sys = Join-Path $env:WINDIR "System32"
Copy-Set $sys @("MSVCP140.dll","VCRUNTIME140.dll","VCRUNTIME140_1.dll","VCOMP140.dll") $baseStage
# docs + licenses
Copy-Item (Join-Path $Root "README.md") $baseStage -Force
Copy-Item (Join-Path $Root "dependencies.json") $baseStage -Force
Copy-Item (Join-Path $Root "LICENSES") (Join-Path $baseStage "LICENSES") -Recurse -Force
if (Test-Path (Join-Path $Root "third_party\whisper.cpp\LICENSE")) {
    Copy-Item (Join-Path $Root "third_party\whisper.cpp\LICENSE") (Join-Path $baseStage "LICENSES\LICENSE-whisper.txt") -Force
}
if (Test-Path (Join-Path $Root "third_party\ffmpeg\LICENSE.txt")) {
    Copy-Item (Join-Path $Root "third_party\ffmpeg\LICENSE.txt") (Join-Path $baseStage "LICENSES\LICENSE-ffmpeg.txt") -Force
}

# ---------------- CUDA pack ----------------
$cudaStage = Join-Path $dist "cuda-pack"
Copy-Set $build @("ggml-cuda.dll","cudart64_*.dll","cublas64_*.dll","cublasLt64_*.dll") $cudaStage
Set-Content (Join-Path $cudaStage "README-CUDA-PACK.txt") @"
SRTCreator NVIDIA (CUDA) GPU pack.
Unzip these files NEXT TO srt.exe / srtgui.exe (same folder as the base download).
The app detects ggml-cuda.dll on startup and uses your NVIDIA GPU automatically.
Requires an NVIDIA driver only (no CUDA Toolkit install).
"@

# ---------------- Vulkan pack ----------------
$vkStage = Join-Path $dist "vulkan-pack"
Copy-Set $build @("ggml-vulkan.dll") $vkStage
Set-Content (Join-Path $vkStage "README-VULKAN-PACK.txt") @"
SRTCreator Vulkan GPU pack (any modern GPU - NVIDIA / AMD / Intel).
Unzip ggml-vulkan.dll NEXT TO srt.exe / srtgui.exe (same folder as the base download).
The app detects it on startup and uses your GPU via the Vulkan driver (already
installed with any recent GPU driver). No SDK or extra runtime needed.
"@

# ---------------- zip ----------------
Write-Host ""
Write-Host "Packaged (dist\):"
Zip $baseStage ("SRTCreator-v{0}-win64-base.zip"        -f $Version)
Zip $cudaStage ("SRTCreator-v{0}-win64-cuda-pack.zip"   -f $Version)
Zip $vkStage   ("SRTCreator-v{0}-win64-vulkan-pack.zip" -f $Version)
