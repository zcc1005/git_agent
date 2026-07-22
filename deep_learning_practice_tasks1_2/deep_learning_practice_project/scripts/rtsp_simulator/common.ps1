$ErrorActionPreference = "Stop"

$script:ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$script:SimulatorRoot = Join-Path $script:ProjectRoot ".local\rtsp_simulator"
$script:BinRoot = Join-Path $script:SimulatorRoot "bin"
$script:MediaMtxRoot = Join-Path $script:SimulatorRoot "mediamtx"
$script:RuntimeRoot = Join-Path $script:SimulatorRoot "runtime"
$script:FfmpegPath = Join-Path $script:BinRoot "ffmpeg.exe"
$script:MediaMtxPath = Join-Path $script:MediaMtxRoot "mediamtx.exe"
$script:MediaMtxConfigPath = Join-Path $PSScriptRoot "mediamtx.yml"

function Assert-SimulatorInstalled {
    $missing = @()
    foreach ($path in @($script:FfmpegPath, $script:MediaMtxPath, $script:MediaMtxConfigPath)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            $missing += $path
        }
    }
    if ($missing.Count -gt 0) {
        throw (
            "RTSP simulator dependencies are missing. Run " +
            "scripts\rtsp_simulator\setup.ps1 first. Missing: " +
            ($missing -join ", ")
        )
    }
}

function Test-TcpPort {
    param(
        [Parameter(Mandatory = $true)][string]$HostName,
        [Parameter(Mandatory = $true)][int]$Port,
        [int]$TimeoutMilliseconds = 500
    )

    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $result = $client.BeginConnect($HostName, $Port, $null, $null)
        if (-not $result.AsyncWaitHandle.WaitOne($TimeoutMilliseconds, $false)) {
            return $false
        }
        $client.EndConnect($result)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Close()
    }
}

function Wait-TcpPort {
    param(
        [Parameter(Mandatory = $true)][string]$HostName,
        [Parameter(Mandatory = $true)][int]$Port,
        [int]$TimeoutSeconds = 15
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-TcpPort -HostName $HostName -Port $Port) {
            return
        }
        Start-Sleep -Milliseconds 250
    }
    throw "Timed out waiting for $HostName`:$Port."
}

function Start-HiddenProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string]$Arguments,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = $Arguments
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $true
    $startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "Failed to start process: $FilePath"
    }
    return $process
}

function Read-OwnedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$PidFile,
        [Parameter(Mandatory = $true)][string]$ExpectedExecutable
    )

    if (-not (Test-Path -LiteralPath $PidFile -PathType Leaf)) {
        return $null
    }
    $rawPid = (Get-Content -LiteralPath $PidFile -Raw).Trim()
    if ($rawPid -notmatch "^[0-9]+$") {
        Remove-Item -LiteralPath $PidFile -Force
        return $null
    }
    $process = Get-Process -Id ([int]$rawPid) -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        Remove-Item -LiteralPath $PidFile -Force
        return $null
    }
    try {
        $actualPath = [System.IO.Path]::GetFullPath($process.Path)
        $expectedPath = [System.IO.Path]::GetFullPath($ExpectedExecutable)
        if (-not $actualPath.Equals($expectedPath, [StringComparison]::OrdinalIgnoreCase)) {
            return $null
        }
    }
    catch {
        return $null
    }
    return $process
}
