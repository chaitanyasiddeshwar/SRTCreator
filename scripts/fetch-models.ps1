# Download whisper.cpp ggml models into the model store.
#   scripts\fetch-models.ps1                       # default: large-v3-turbo-q8_0
#   scripts\fetch-models.ps1 -Model large-v3       # a specific model
#   scripts\fetch-models.ps1 -Vad                  # also fetch the Silero VAD model
param(
    [string]$Model = "large-v3-turbo-q8_0",
    [string]$Dir   = "$env:LOCALAPPDATA\SRTCreator\models",
    [switch]$Vad
)

$base = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
New-Item -ItemType Directory -Force -Path $Dir | Out-Null

function Get-Model([string]$file) {
    $url = "$base/$file"
    $out = Join-Path $Dir $file
    Write-Host "[models] $file -> $out"
    curl.exe -L --fail --progress-bar -o $out $url
    if ($LASTEXITCODE -ne 0) { throw "download failed: $url" }
}

Get-Model "ggml-$Model.bin"
if ($Vad) { Get-Model "ggml-silero-v5.1.2.bin" }
Write-Host "[models] done."
