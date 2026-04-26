$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$packageName = 'animotion3d_ai_director'
$packagePath = Join-Path $projectRoot $packageName
$distPath = Join-Path $projectRoot 'dist'
$zipPath = Join-Path $distPath "$packageName.zip"

if (-not (Test-Path $distPath)) {
    New-Item -ItemType Directory -Path $distPath | Out-Null
}

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Compress-Archive -Path $packagePath -DestinationPath $zipPath -Force
Write-Output "Created $zipPath"
