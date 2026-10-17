param([string]$Port = 'COM7', [switch]$Spectrum)
$ErrorActionPreference = 'Stop'
$taskWorkspace = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskPio = Join-Path $taskWorkspace '.venv-platformio/Scripts/pio.exe'
if (Test-Path -LiteralPath $taskPio) {
    $env:PLATFORMIO_CORE_DIR = Join-Path $taskWorkspace '.platformio-local'
} else {
    $taskPio = 'pio'
}
$taskEnvironment = if ($Spectrum) { 'esp32dev_spectrum' } else { 'esp32dev' }
Write-Host "Upload firmware B: $taskEnvironment -> $Port"
& $taskPio run -d $PSScriptRoot -e $taskEnvironment --target upload --upload-port $Port
if ($LASTEXITCODE -ne 0) { throw 'Firmware B upload failed.' }
