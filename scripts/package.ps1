# Package a self-contained SRTCreator distribution zip.
#   scripts\package.ps1                 # default version
#   scripts\package.ps1 -Version 0.2.0
#
# Bundles srt.exe + exactly the runtime DLLs it needs (FFmpeg, CUDA, VC++),
# plus README and license notices. Whisper models are NOT bundled - they are
# downloaded on first run. Requires a prior successful build (build\srt.exe).
param(
    [string]$Version = "0.1.0",
    [string]$Root    = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = "Stop"

$build = Join-Path $Root "build"
$exe   = Join-Path $build "srt.exe"
if (-not (Test-Path $exe)) { throw "srt.exe not found in build\. Run build.bat first." }

$cuda  = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0" }
$ffbin = Join-Path $Root "third_party\ffmpeg\bin"
$sys   = Join-Path $env:WINDIR "System32"

$dist  = Join-Path $Root "dist"
$stage = Join-Path $dist "SRTCreator"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null

# 1) The executables (CLI + GUI).
Copy-Item $exe $stage
$gui = Join-Path $build "srtgui.exe"
if (Test-Path $gui) { Copy-Item $gui $stage }

# 2) FFmpeg DLLs - exactly the four the exe links (not avfilter/avdevice/swscale).
foreach ($d in "avformat-63.dll","avcodec-63.dll","avutil-61.dll","swresample-7.dll") {
    Copy-Item (Join-Path $ffbin $d) $stage
}

# 3) CUDA runtime DLLs (CUDA 13 ships them in bin\x64; CUDA 12 in bin).
function Find-Cuda([string]$pat) {
    $hit = Get-ChildItem -Path (Join-Path $cuda "bin\x64"),(Join-Path $cuda "bin") -Filter $pat -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $hit) { throw "CUDA DLL not found: $pat (looked under $cuda)" }
    return $hit.FullName
}
foreach ($p in "cudart64_*.dll","cublas64_*.dll","cublasLt64_*.dll") {
    Copy-Item (Find-Cuda $p) $stage
}

# 4) Visual C++ runtime (app-local; avoids requiring the VC++ Redist install).
foreach ($d in "MSVCP140.dll","VCRUNTIME140.dll","VCRUNTIME140_1.dll","VCOMP140.dll") {
    Copy-Item (Join-Path $sys $d) $stage
}

# 4b) ONNX Runtime (DirectML) for vocal isolation.
$ortbin = Join-Path $Root "third_party\onnxruntime\bin"
foreach ($d in "onnxruntime.dll","DirectML.dll") {
    $p = Join-Path $ortbin $d
    if (Test-Path $p) { Copy-Item $p $stage }
}

# 5) Docs + licenses.
Copy-Item (Join-Path $Root "README.md") $stage
Copy-Item (Join-Path $Root "THIRD_PARTY_NOTICES.txt") $stage
Copy-Item (Join-Path $Root "third_party\whisper.cpp\LICENSE") (Join-Path $stage "LICENSE-whisper.txt")
Copy-Item (Join-Path $Root "third_party\ffmpeg\LICENSE.txt")   (Join-Path $stage "LICENSE-ffmpeg.txt")

# 6) Zip it.
$zip = Join-Path $dist ("SRTCreator-v{0}-win64-cuda.zip" -f $Version)
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $stage -DestinationPath $zip

Write-Host ""
Write-Host "Packaged: $zip"
Get-ChildItem $stage | Select-Object Name,@{n="KB";e={[math]::Round($_.Length/1KB)}} | Format-Table -AutoSize
Write-Host ("Zip size: {0:N1} MB" -f ((Get-Item $zip).Length/1MB))
