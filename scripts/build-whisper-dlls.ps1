<#
.SYNOPSIS
    Deps build (rare - only on a whisper version bump). Compiles the vendored
    whisper.cpp submodule into a COHERENT set of GGML_BACKEND_DL plugin DLLs
    (CPU variants + CUDA + Vulkan) at ONE ggml version, and stages the DLLs +
    import lib + headers into third_party/whisper/{bin,lib,include}.

    The app build (CMakeLists.txt) then consumes third_party/whisper exactly like
    FFmpeg - import lib + headers, no add_subdirectory, no CUDA/nvcc/Vulkan. See
    modular_proposal.md sections 11-12.

.DESCRIPTION
    Building every backend from one submodule commit guarantees ABI coherence,
    which mixing prebuilt zips from different tags does NOT. At runtime ggml's
    ggml_backend_load_all() scans the exe dir, loads whatever backend DLLs are
    present, scores the CPU variants against the host, and auto-selects the best
    available device (CUDA -> Vulkan -> CPU).

.PARAMETER Backends
    Which GPU backends to compile in, comma-separated: cuda, vulkan. CPU (with
    all variants) is always built. Default: cuda,vulkan. Drop one to build on a
    machine that lacks its toolkit/SDK (e.g. -Backends cuda before the Vulkan SDK
    is installed, or -Backends vulkan on a non-NVIDIA build box).

.PARAMETER CudaArch
    CMAKE_CUDA_ARCHITECTURES value. Default 86 (Ampere / RTX 3080 Ti). Add more
    (e.g. "75;86;89") to ship kernels for more GPU generations at a size cost.

.PARAMETER WhisperSrc  Vendored submodule path.       Default third_party/whisper.cpp
.PARAMETER BuildDir    Out-of-tree CMake build dir.    Default build-whisper-dlls
.PARAMETER OutDir      Staging root for the app build. Default third_party/whisper
.PARAMETER Clean       Wipe BuildDir before configuring.

.NOTES
    Build-machine requirements (END USERS need NONE of these):
      - CUDA Toolkit (for ggml-cuda.dll)          -> CUDA_PATH / nvcc on PATH
      - Vulkan SDK  (for ggml-vulkan.dll)          -> VULKAN_SDK set, glslc on PATH
      - MSVC x64 build tools + the bundled Ninja (drives nvcc directly).

    Runs MSVC discovery via vswhere + Enter-VsDevShell so it works
    non-interactively (Clink does not hook PowerShell). See CLAUDE.md 4.1.

.EXAMPLE
    # First light before the Vulkan SDK is installed - CUDA + CPU only:
    powershell -ExecutionPolicy Bypass -File scripts\build-whisper-dlls.ps1 -Backends cuda

.EXAMPLE
    # Full coherent set once the Vulkan SDK is present:
    powershell -ExecutionPolicy Bypass -File scripts\build-whisper-dlls.ps1
#>
[CmdletBinding()]
param(
    [string]   $Backends   = "cuda,vulkan",
    [string]   $CudaArch   = "86",
    [string]   $WhisperSrc = "third_party/whisper.cpp",
    [string]   $BuildDir   = "build-whisper-dlls",
    [string]   $OutDir     = "third_party/whisper",
    [bool]     $BuildExamples = $true,
    [int]      $Jobs       = 0,   # ninja parallelism; 0 = ninja default (all cores).
                                  # Lower it (e.g. 6) to cut the odds of the flaky
                                  # ptxas crash below.
    [int]      $BuildRetries = 8, # CUDA's assembler (ptxas) can die with a
                                  # non-deterministic ACCESS_VIOLATION on the heavy
                                  # flash-attn kernels (a different kernel each run,
                                  # even with RAM to spare). ninja keeps every
                                  # object it already built, so we just resume: each
                                  # retry recompiles only the crashed kernel and the
                                  # passes converge. This many resume attempts.
    [switch]   $Clean
)

# NOTE: keep this at "Continue", NOT "Stop". Under Windows PowerShell 5.1 a native
# exe (cmake/ninja/nvcc) writing to stderr - even a harmless CMake deprecation
# WARNING - is wrapped as a NativeCommandError, and "Stop" would promote that into
# a terminating error that aborts the whole build before it starts. We gate real
# failures on $LASTEXITCODE after each native call and -ErrorAction Stop on the
# cmdlets that must succeed instead.
$ErrorActionPreference = "Continue"

# --- resolve repo root (this script lives in <root>/scripts) ---
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$backendList = @($Backends -split "," | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })
$wantCuda    = $backendList -contains "cuda"
$wantVulkan  = $backendList -contains "vulkan"

function Fail($msg) { Write-Host "[deps] ERROR: $msg" -ForegroundColor Red; exit 1 }
function Info($msg) { Write-Host "[deps] $msg" -ForegroundColor Cyan }

Info "backends: CPU(all variants)$(if($wantCuda){' + CUDA'})$(if($wantVulkan){' + Vulkan'})"

# --- sanity: submodule present ---
if (-not (Test-Path (Join-Path $RepoRoot "$WhisperSrc/CMakeLists.txt"))) {
    Fail "whisper submodule not found at $WhisperSrc. Run: git submodule update --init --recursive"
}

# --- MSVC via vswhere + Enter-VsDevShell ---
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { Fail "vswhere.exe not found - install VS Build Tools with the C++ x64 toolset." }
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { Fail "No VS installation with the MSVC C++ x64 toolset found." }
Info "MSVC: $vsPath"
Import-Module (Join-Path $vsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll") -ErrorAction Stop
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64" -ErrorAction Stop | Out-Null

# --- bundled Ninja (drives nvcc directly; avoids the VS-generator CUDA gotcha) ---
$ninjaDir = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
if (Test-Path (Join-Path $ninjaDir "ninja.exe")) { $env:PATH = "$ninjaDir;$env:PATH" }
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) { Fail "ninja not found (expected bundled under VS CMake component)." }

# --- CUDA toolkit ---
if ($wantCuda) {
    if (-not $env:CUDA_PATH) { $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0" }
    if (-not (Test-Path (Join-Path $env:CUDA_PATH "bin\nvcc.exe"))) {
        Fail "nvcc not found at $env:CUDA_PATH. Set CUDA_PATH, or pass -Backends without cuda."
    }
    $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
    Info "CUDA: $env:CUDA_PATH (arch $CudaArch)"
}

# --- Vulkan SDK ---
if ($wantVulkan) {
    if (-not $env:VULKAN_SDK) {
        Fail "VULKAN_SDK not set - install the LunarG Vulkan SDK (one-time), or pass -Backends cuda to skip Vulkan."
    }
    $env:PATH = "$env:VULKAN_SDK\Bin;$env:PATH"
    if (-not (Get-Command glslc -ErrorAction SilentlyContinue)) {
        Fail "glslc not found under VULKAN_SDK ($env:VULKAN_SDK). The SDK's shader compiler is required to build ggml-vulkan.dll."
    }
    Info "Vulkan SDK: $env:VULKAN_SDK"
}

# --- clean ---
if ($Clean -and (Test-Path $BuildDir)) { Info "clean: removing $BuildDir"; Remove-Item -Recurse -Force $BuildDir }

# --- configure ---
# GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS => each backend is a separate drop-in DLL
# and the CPU variants (haswell/icelake/...) are scored against the host at load.
$cmakeArgs = @(
    "-S", $WhisperSrc,
    "-B", $BuildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
    "-DGGML_BACKEND_DL=ON",
    "-DGGML_CPU_ALL_VARIANTS=ON",
    "-DGGML_NATIVE=OFF",
    "-DWHISPER_BUILD_TESTS=OFF",
    # whisper-cli.exe (an example) is our isolated MILESTONE-1 harness: it already
    # calls ggml_backend_load_all(), so it exercises CPU<->CUDA<->Vulkan selection
    # against the staged DLLs exactly as the app will, before the Phase 2b swap.
    "-DWHISPER_BUILD_EXAMPLES=$(if ($BuildExamples) { 'ON' } else { 'OFF' })"
)
if ($wantCuda)   { $cmakeArgs += @("-DGGML_CUDA=ON",   "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch") }
if ($wantVulkan) { $cmakeArgs += @("-DGGML_VULKAN=ON") }

Info "configuring..."
cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { Fail "cmake configure failed." }

Info "building (this compiles CUDA kernels - can take 20-40 min on a cold build)..."
$buildArgs = @("--build", $BuildDir, "--config", "Release")
if ($Jobs -gt 0) { Info "ninja parallelism capped at -j $Jobs"; $buildArgs += @("--", "-j", "$Jobs") }
$maxAttempts = [Math]::Max(1, $BuildRetries + 1)
for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    if ($attempt -gt 1) { Info "build attempt $attempt/$maxAttempts (resuming from cache; only crashed kernels recompile)..." }
    cmake @buildArgs
    if ($LASTEXITCODE -eq 0) { break }
    if ($attempt -eq $maxAttempts) { Fail "build failed after $maxAttempts attempts (last exit $LASTEXITCODE)." }
    Info "build step failed (exit $LASTEXITCODE) - likely a flaky ptxas crash; retrying resume..."
    Start-Sleep -Seconds 3
}

# --- stage into third_party/whisper/{bin,lib,include} (glob-based; robust to exact names) ---
$binOut = Join-Path $OutDir "bin"
$libOut = Join-Path $OutDir "lib"
$incOut = Join-Path $OutDir "include"
foreach ($d in @($binOut, $libOut, $incOut)) {
    if (Test-Path $d) { Remove-Item -Recurse -Force $d -ErrorAction Stop }
    New-Item -ItemType Directory -Force -Path $d -ErrorAction Stop | Out-Null
}

# DLLs: whisper.dll, ggml.dll, ggml-base.dll, ggml-cpu*.dll, ggml-cuda.dll, ggml-vulkan.dll
$dlls = Get-ChildItem -Path $BuildDir -Recurse -Filter *.dll
if (-not $dlls) { Fail "no DLLs found under $BuildDir - did the shared build produce plugins?" }
$dlls | ForEach-Object { Copy-Item $_.FullName $binOut -Force -ErrorAction Stop }

# Import libs: whisper.lib (+ any ggml*.lib the linker needs).
$libs = Get-ChildItem -Path $BuildDir -Recurse -Include whisper.lib, ggml*.lib
if (-not ($libs | Where-Object { $_.Name -eq "whisper.lib" })) {
    Fail "whisper.lib import library not found under $BuildDir."
}
$libs | ForEach-Object { Copy-Item $_.FullName $libOut -Force -ErrorAction Stop }

# Public headers: whisper.cpp/include + ggml/include.
Copy-Item (Join-Path $WhisperSrc "include\*.h")      $incOut -Force -ErrorAction Stop
Copy-Item (Join-Path $WhisperSrc "ggml\include\*.h") $incOut -Force -ErrorAction Stop

# --- report / manifest ---
Info "staged into $OutDir :"
Write-Host ("  bin/    " + (($dlls | ForEach-Object { $_.Name } | Sort-Object -Unique) -join ", "))
Write-Host ("  lib/    " + ((Get-ChildItem $libOut | ForEach-Object { $_.Name }) -join ", "))
Write-Host ("  include/" + (Get-ChildItem $incOut).Count + " headers")

# A tiny sanity check for the coherent-set expectation.
$dllNames = $dlls | ForEach-Object { $_.Name.ToLower() }
if ($wantCuda   -and -not ($dllNames -contains "ggml-cuda.dll"))   { Write-Host "[deps] WARN: ggml-cuda.dll missing from the staged set." -ForegroundColor Yellow }
if ($wantVulkan -and -not ($dllNames -contains "ggml-vulkan.dll")) { Write-Host "[deps] WARN: ggml-vulkan.dll missing from the staged set." -ForegroundColor Yellow }

Info "done. The app build now links third_party/whisper (import lib + headers) and ships bin/ next to the exe."
