[CmdletBinding()]
param(
    [string]$OutputPath,
    [ValidateRange(5, 600)][int]$DurationSeconds = 20
)

. (Join-Path $PSScriptRoot "common.ps1")
Assert-SimulatorInstalled

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $script:ProjectRoot "outputs\rtsp_simulator\sample.mp4"
}
elseif (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $script:ProjectRoot $OutputPath
}
$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
New-Item -ItemType Directory -Path (Split-Path -Parent $OutputPath) -Force | Out-Null

$arguments = @(
    "-hide_banner",
    "-loglevel", "warning",
    "-y",
    "-f", "lavfi",
    "-i", "testsrc2=size=640x360:rate=25",
    "-t", $DurationSeconds,
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    $OutputPath
)
$previousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $script:FfmpegPath $arguments
$ffmpegExitCode = $LASTEXITCODE
$ErrorActionPreference = $previousPreference
if ($ffmpegExitCode -ne 0) {
    throw "Failed to create the sample video. FFmpeg exit code: $ffmpegExitCode"
}

[ordered]@{
    ok = $true
    video_path = $OutputPath
    duration_seconds = $DurationSeconds
} | ConvertTo-Json
