[CmdletBinding()]
param(
    [ValidatePattern("^[a-zA-Z0-9_-]+$")][string]$StreamName = "main-monitor"
)

. (Join-Path $PSScriptRoot "common.ps1")

$results = @()
$targets = @(
    @{
        Name = "publisher-$StreamName"
        PidFile = (Join-Path $script:RuntimeRoot "publisher-$StreamName.pid")
        Executable = $script:FfmpegPath
    },
    @{
        Name = "mediamtx"
        PidFile = (Join-Path $script:RuntimeRoot "mediamtx.pid")
        Executable = $script:MediaMtxPath
    }
)

foreach ($target in $targets) {
    $process = Read-OwnedProcess -PidFile $target.PidFile -ExpectedExecutable $target.Executable
    if ($null -ne $process) {
        Stop-Process -Id $process.Id
        $process.WaitForExit(5000) | Out-Null
        $results += [ordered]@{ name = $target.Name; stopped = $true; pid = $process.Id }
    }
    else {
        $results += [ordered]@{ name = $target.Name; stopped = $false; pid = $null }
    }
    if (Test-Path -LiteralPath $target.PidFile) {
        Remove-Item -LiteralPath $target.PidFile -Force
    }
}

[ordered]@{
    ok = $true
    processes = $results
} | ConvertTo-Json -Depth 4
