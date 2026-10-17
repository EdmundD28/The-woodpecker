param([string]$Compiler, [string]$Run)
$ErrorActionPreference = 'Stop'
$testPython = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $testPython)) { $testPython = 'python' }
$testArgs = @((Join-Path $PSScriptRoot 'test/verify_models.py'))
if ($Compiler) { $testArgs += @('--compiler', $Compiler) }
if ($Run) { $testArgs += @('--run', $Run) }
& $testPython @testArgs
if ($LASTEXITCODE -ne 0) { throw 'Firmware/model host verification failed.' }
