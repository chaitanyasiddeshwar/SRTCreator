<#
.SYNOPSIS
    Fetch the prebuilt third-party dependencies a clean checkout needs to build the
    app: FFmpeg (BtbN LGPL shared) and ONNX Runtime + DirectML. Stages each into
    third_party/{ffmpeg,onnxruntime}/{include,lib,bin} - the layout CMakeLists.txt
    consumes. Versions come from ../dependencies.json (override with the params).

    This is the single biggest prerequisite for building on a machine that does not
    already have the git-ignored third_party libs (CI, a new contributor). It does
    NOT fetch the whisper backend DLLs - those are produced by build-whisper-dlls.ps1
    and mirrored under third_party/whisper. See modular_proposal.md 12.1.

.NOTES
    Requires network access to github.com release CDNs and nuget.org. Run once per
    checkout (or when dependencies.json is bumped). Idempotent: -Force re-downloads.

.PARAMETER FfmpegUrl   Override the FFmpeg zip URL (default: BtbN latest LGPL shared).
.PARAMETER OrtVersion  ONNX Runtime (DirectML) version to fetch (default: from dependencies.json).
.PARAMETER DmlVersion  Microsoft.AI.DirectML version to fetch (default: from dependencies.json).
.PARAMETER Force       Re-download even if third_party/<dep> already looks populated.
#>
[CmdletBinding()]
param(
    [string] $FfmpegUrl  = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl-shared.zip",
    [string] $OrtVersion = "",
    [string] $DmlVersion = "",
    [switch] $Force
)
$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$RepoRoot = Split-Path -Parent $PSScriptRoot
$tp       = Join-Path $RepoRoot "third_party"
$manifest = Get-Content (Join-Path $RepoRoot "dependencies.json") -Raw | ConvertFrom-Json
if (-not $OrtVersion) { $OrtVersion = $manifest.onnxruntime.version }
if (-not $DmlVersion) { $DmlVersion = $manifest.directml.version }

function Info($m) { Write-Host "[fetch-deps] $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "[fetch-deps] ERROR: $m" -ForegroundColor Red; exit 1 }

# Download to a temp file, extract into a fresh temp dir, return the extract dir.
function Get-AndExtract([string]$url, [string]$name) {
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("srtdep-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    $zip = Join-Path $tmp "$name.zip"
    Info "downloading $name ..."
    Info "  $url"
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    Info "extracting $name ..."
    $out = Join-Path $tmp "x"
    Expand-Archive -Path $zip -DestinationPath $out -Force
    return $out
}

# Stage include/lib/bin from a source tree into third_party/<dep>, replacing it.
function Stage-Dep([string]$dep, [hashtable]$dirs) {
    $dest = Join-Path $tp $dep
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    foreach ($sub in 'include','lib','bin') {
        $src = $dirs[$sub]
        if (-not $src -or -not (Test-Path $src)) { Fail "$dep : missing '$sub' at '$src'" }
        $d = Join-Path $dest $sub
        New-Item -ItemType Directory -Force -Path $d | Out-Null
        Copy-Item (Join-Path $src '*') $d -Recurse -Force
    }
    Info "$dep staged -> $dest"
}

# --- FFmpeg (BtbN LGPL shared): zip root is ffmpeg-*/... with bin/include/lib ---
if ($Force -or -not (Test-Path (Join-Path $tp "ffmpeg\include"))) {
    $x = Get-AndExtract $FfmpegUrl "ffmpeg"
    $root = Get-ChildItem $x -Directory | Select-Object -First 1   # the single ffmpeg-* folder
    if (-not $root) { $root = Get-Item $x }
    Stage-Dep "ffmpeg" @{ include = (Join-Path $root.FullName "include");
                          lib     = (Join-Path $root.FullName "lib");
                          bin     = (Join-Path $root.FullName "bin") }
} else { Info "ffmpeg already present (use -Force to refresh)" }

# --- ONNX Runtime (DirectML EP) + DirectML runtime, from NuGet ------------------
# The DirectML build of ONNX Runtime is NOT published as a GitHub release asset;
# it ships as the NuGet package Microsoft.ML.OnnxRuntime.DirectML (onnxruntime.dll
# + headers + import lib), and DirectML.dll comes from the separate NuGet package
# Microsoft.AI.DirectML. A .nupkg is a plain zip, so we download and unpack both
# and stage the win-x64 native files into third_party/onnxruntime/{include,lib,bin}.
if ($Force -or -not (Test-Path (Join-Path $tp "onnxruntime\include"))) {
    # 1) ONNX Runtime DirectML package -> onnxruntime.dll/.lib + headers.
    $ortUrl = "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/$OrtVersion"
    $ox = Get-AndExtract $ortUrl "onnxruntime"
    $ortInc = Join-Path $ox "build\native\include"
    $ortNat = Join-Path $ox "runtimes\win-x64\native"
    if (-not (Test-Path (Join-Path $ortInc "onnxruntime_c_api.h"))) { Fail "onnxruntime: headers not found under $ortInc" }
    if (-not (Test-Path (Join-Path $ortNat "onnxruntime.dll")))     { Fail "onnxruntime: win-x64 native DLL not found under $ortNat" }

    # 2) DirectML runtime package -> DirectML.dll (win-x64 desktop variant).
    $dmlUrl = "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/$DmlVersion"
    $dx = Get-AndExtract $dmlUrl "directml"
    $dmlDll = Join-Path $dx "bin\x64-win\DirectML.dll"
    if (-not (Test-Path $dmlDll)) { Fail "directml: DirectML.dll not found at $dmlDll" }

    # Stage the merged set into third_party/onnxruntime/{include,lib,bin}.
    $dest = Join-Path $tp "onnxruntime"
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    foreach ($sub in 'include','lib','bin') { New-Item -ItemType Directory -Force -Path (Join-Path $dest $sub) | Out-Null }
    Copy-Item (Join-Path $ortInc '*.h')                    (Join-Path $dest 'include') -Force
    Copy-Item (Join-Path $ortNat 'onnxruntime.lib')        (Join-Path $dest 'lib')     -Force
    Copy-Item (Join-Path $ortNat 'onnxruntime.dll')        (Join-Path $dest 'bin')     -Force
    Copy-Item $dmlDll                                       (Join-Path $dest 'bin')     -Force
    Info "onnxruntime staged -> $dest (ORT $OrtVersion + DirectML $DmlVersion)"
} else { Info "onnxruntime already present (use -Force to refresh)" }

Info "done. third_party/ffmpeg and third_party/onnxruntime are ready for the app build."
Info "whisper backend DLLs are separate: run scripts/build-whisper-dlls.ps1 (build machine / CI)."
