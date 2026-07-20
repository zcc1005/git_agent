[CmdletBinding()]
param(
    [string]$PythonPath
)

. (Join-Path $PSScriptRoot "common.ps1")

New-Item -ItemType Directory -Path $script:BinRoot, $script:MediaMtxRoot -Force | Out-Null

function Resolve-ProjectPython {
    if (-not [string]::IsNullOrWhiteSpace($PythonPath)) {
        if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
            throw "PythonPath does not exist: $PythonPath"
        }
        return [System.IO.Path]::GetFullPath($PythonPath)
    }

    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE "anaconda3\envs\dl_practice\python.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\envs\dl_practice\python.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\python.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return [System.IO.Path]::GetFullPath($candidate)
        }
    }
    throw "Python was not found. Pass -PythonPath with the project environment python.exe."
}

if (-not (Test-Path -LiteralPath $script:FfmpegPath -PathType Leaf)) {
    $python = Resolve-ProjectPython
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $ffmpegSource = & $python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>$null
    $importExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($importExitCode -ne 0) {
        throw "imageio-ffmpeg is missing. Install the project requirements first."
    }
    $ffmpegSource = ($ffmpegSource | Select-Object -Last 1).Trim()
    if (-not (Test-Path -LiteralPath $ffmpegSource -PathType Leaf)) {
        throw "imageio-ffmpeg returned an invalid executable path: $ffmpegSource"
    }
    Copy-Item -LiteralPath $ffmpegSource -Destination $script:FfmpegPath -Force
}

if (-not (Test-Path -LiteralPath $script:MediaMtxPath -PathType Leaf)) {
    throw (
        "MediaMTX is missing. Download the Windows amd64 archive from " +
        "https://github.com/bluenviron/mediamtx/releases and extract it to " +
        $script:MediaMtxRoot
    )
}

Assert-SimulatorInstalled

[ordered]@{
    ok = $true
    install_root = $script:SimulatorRoot
    ffmpeg = $script:FfmpegPath
    mediamtx = $script:MediaMtxPath
    mediamtx_config = $script:MediaMtxConfigPath
} | ConvertTo-Json
