# Збирання PC Benchmark: іконка -> PyInstaller (.exe) -> Inno Setup (установник)
#   powershell -ExecutionPolicy Bypass -File build.ps1            # остання версія
#   powershell -ExecutionPolicy Bypass -File build.ps1 -Version 0.8
param([string]$Version = "0.8")
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$script = Join-Path $root "benchmark_v$Version.py"
$icon = Join-Path $root "assets\pcbench.ico"
if (-not (Test-Path $script)) { throw "Не знайдено $script" }

# 1. Іконка
if (-not (Test-Path $icon)) {
    New-Item -ItemType Directory -Force (Split-Path $icon) | Out-Null
    python (Join-Path $root "tools\make_icon.py") $icon
}

# 2. Програма (.exe) — тека з exe та бібліотеками, без консолі
python -m PyInstaller --noconfirm --clean --windowed --name "PC Benchmark" `
    --icon $icon --add-data "$icon;." `
    --distpath (Join-Path $root "build\dist") --workpath (Join-Path $root "build\work") `
    --specpath (Join-Path $root "build") $script
if ($LASTEXITCODE -ne 0) { throw "PyInstaller завершився з помилкою" }

# 3. Установник
$iscc = @("$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
          "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
          "$env:ProgramFiles\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { throw "Не знайдено Inno Setup 6 (ISCC.exe): https://jrsoftware.org/isdl.php" }
& $iscc "/DAppVersion=$Version" (Join-Path $root "installer\pc_benchmark.iss")
if ($LASTEXITCODE -ne 0) { throw "Inno Setup завершився з помилкою" }

Write-Host "Готово: $(Join-Path $root "build\installer\PC-Benchmark-Setup-$Version.exe")"
