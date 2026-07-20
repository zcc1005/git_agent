[CmdletBinding()]
param(
    [ValidatePattern("^[a-zA-Z0-9_-]+$")][string]$StreamName = "main-monitor",
    [ValidateRange(1, 30)][int]$TimeoutSeconds = 8,
    [string]$SnapshotPath
)

. (Join-Path $PSScriptRoot "common.ps1")
Assert-SimulatorInstalled

$rtspUrl = "rtsp://127.0.0.1:8554/$StreamName"
$timeoutMicroseconds = $TimeoutSeconds * 1000000
if ([string]::IsNullOrWhiteSpace($SnapshotPath)) {
    $SnapshotPath = Join-Path $script:ProjectRoot "outputs\rtsp_simulator\verified-frame.jpg"
}
elseif (-not [System.IO.Path]::IsPathRooted($SnapshotPath)) {
    $SnapshotPath = Join-Path $script:ProjectRoot $SnapshotPath
}
$SnapshotPath = [System.IO.Path]::GetFullPath($SnapshotPath)
New-Item -ItemType Directory -Path (Split-Path -Parent $SnapshotPath) -Force | Out-Null

$arguments = @(
    "-hide_banner",
    "-loglevel", "error",
    "-rtsp_transport", "tcp",
    "-timeout", $timeoutMicroseconds,
    "-i", $rtspUrl,
    "-frames:v", "1",
    "-q:v", "2",
    "-y",
    $SnapshotPath
)
$previousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$probeOutput = & $script:FfmpegPath $arguments 2>&1
$probeExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousPreference
if ($probeExitCode -ne 0) {
    throw "RTSP verification failed: $rtspUrl`n$($probeOutput -join [Environment]::NewLine)"
}
if (-not (Test-Path -LiteralPath $SnapshotPath -PathType Leaf) -or
    (Get-Item -LiteralPath $SnapshotPath).Length -eq 0) {
    throw "RTSP connected but no verification frame was produced: $rtspUrl"
}

[ordered]@{
    ok = $true
    rtsp_url = $rtspUrl
    snapshot_path = $SnapshotPath
    snapshot_bytes = (Get-Item -LiteralPath $SnapshotPath).Length
} | ConvertTo-Json
