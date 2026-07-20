[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$VideoPath,
    [ValidatePattern("^[a-zA-Z0-9_-]+$")][string]$StreamName = "main-monitor"
)

. (Join-Path $PSScriptRoot "common.ps1")
Assert-SimulatorInstalled

if (-not [System.IO.Path]::IsPathRooted($VideoPath)) {
    $VideoPath = Join-Path $script:ProjectRoot $VideoPath
}
$VideoPath = [System.IO.Path]::GetFullPath($VideoPath)
if (-not (Test-Path -LiteralPath $VideoPath -PathType Leaf)) {
    throw "The RTSP simulator input video does not exist: $VideoPath"
}

New-Item -ItemType Directory -Path $script:RuntimeRoot -Force | Out-Null
$serverPidFile = Join-Path $script:RuntimeRoot "mediamtx.pid"
$publisherPidFile = Join-Path $script:RuntimeRoot "publisher-$StreamName.pid"
$serverLog = Join-Path $script:MediaMtxRoot "mediamtx.log"

$serverProcess = Read-OwnedProcess -PidFile $serverPidFile -ExpectedExecutable $script:MediaMtxPath
if ($null -eq $serverProcess) {
    if (Test-TcpPort -HostName "127.0.0.1" -Port 8554) {
        throw "Port 8554 is already in use by a process not owned by this simulator."
    }
    $serverArguments = '"{0}"' -f $script:MediaMtxConfigPath
    $serverProcess = Start-HiddenProcess `
        -FilePath $script:MediaMtxPath `
        -Arguments $serverArguments `
        -WorkingDirectory $script:MediaMtxRoot
    Set-Content -LiteralPath $serverPidFile -Value $serverProcess.Id -Encoding Ascii
}
Wait-TcpPort -HostName "127.0.0.1" -Port 8554 -TimeoutSeconds 15

$existingPublisher = Read-OwnedProcess -PidFile $publisherPidFile -ExpectedExecutable $script:FfmpegPath
if ($null -ne $existingPublisher) {
    throw "Publisher $StreamName is already running with PID $($existingPublisher.Id)."
}

$rtspUrl = "rtsp://127.0.0.1:8554/$StreamName"
$publisherArguments = (
    '-hide_banner -loglevel error -nostdin -re -stream_loop -1 ' +
    '-i "{0}" -map 0:v:0 -an -c:v libx264 -preset veryfast ' +
    '-tune zerolatency -pix_fmt yuv420p -f rtsp ' +
    '-rtsp_transport tcp {1}'
) -f $VideoPath, $rtspUrl
$publisherProcess = Start-HiddenProcess `
    -FilePath $script:FfmpegPath `
    -Arguments $publisherArguments `
    -WorkingDirectory $script:ProjectRoot
Set-Content -LiteralPath $publisherPidFile -Value $publisherProcess.Id -Encoding Ascii

Start-Sleep -Seconds 2
$publisherProcess.Refresh()
if ($publisherProcess.HasExited) {
    throw "The FFmpeg publisher failed to start. Check $serverLog."
}

[ordered]@{
    ok = $true
    rtsp_url = $rtspUrl
    video_path = $VideoPath
    mediamtx_pid = $serverProcess.Id
    publisher_pid = $publisherProcess.Id
    runtime_directory = $script:RuntimeRoot
    mediamtx_log = $serverLog
} | ConvertTo-Json
