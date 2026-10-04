$ErrorActionPreference = 'Stop'
$testCompiler = Join-Path $PSScriptRoot '../../work/b-test-tools/ziglang/zig.exe'
if (!(Test-Path -LiteralPath $testCompiler)) {
    throw 'Local test compiler missing. See README for a C++17 compiler command.'
}
Push-Location $PSScriptRoot
try {
    & $testCompiler c++ -std=c++17 -I test/host test/host/test_firmware.cpp -o test/host/test_firmware.exe
    if ($LASTEXITCODE -ne 0) { throw 'Host test build failed.' }
    & ./test/host/test_firmware.exe
    if ($LASTEXITCODE -ne 0) { throw 'Host test assertions failed.' }
} finally {
    Pop-Location
}
