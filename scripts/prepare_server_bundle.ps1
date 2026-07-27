[CmdletBinding()]
param(
    [string]$OutputDirectory = "",
    [switch]$BuildImage
)

$ErrorActionPreference = "Stop"

$repositoryRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..")
)
$bundleRoot = if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    Join-Path $repositoryRoot "server_bundle"
} elseif ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    [System.IO.Path]::GetFullPath($OutputDirectory)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $OutputDirectory))
}

$projectRoot = Join-Path $repositoryRoot `
    "deep_learning_practice_tasks1_2\deep_learning_practice_project"
$sourceEnv = Join-Path $projectRoot ".env"
$sourceVideo = Join-Path $projectRoot "data\monitor\monitor.mp4"
$sourceCompose = Join-Path $repositoryRoot "compose.yaml"
$sourceGpuCompose = Join-Path $repositoryRoot "compose.gpu.yaml"
$sourceMediaMtx = Join-Path $repositoryRoot "deploy\mediamtx.yml"
$sourceDeploymentGuide = Join-Path $repositoryRoot "DEPLOYMENT.md"
$sourceUploadChecklist = Join-Path $repositoryRoot "deploy\SERVER_UPLOAD_FILES.md"
$sourceImageArchive = Join-Path $repositoryRoot "belt-agent.tar"

foreach ($requiredPath in @(
    $sourceEnv,
    $sourceVideo,
    $sourceCompose,
    $sourceGpuCompose,
    $sourceMediaMtx,
    $sourceDeploymentGuide,
    $sourceUploadChecklist
)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required deployment file is missing: $requiredPath"
    }
}

function Read-DotEnv {
    param([Parameter(Mandatory = $true)][string]$Path)

    $values = @{}
    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
            continue
        }
        $key, $value = $trimmed.Split("=", 2)
        $values[$key.Trim()] = $value.Trim()
    }
    return $values
}

$environment = Read-DotEnv -Path $sourceEnv
$provider = [string]$environment["LLM_PROVIDER"]
if ([string]::IsNullOrWhiteSpace($provider)) {
    $provider = "c4ai"
}
if ($provider -ne "c4ai") {
    throw "The project is not configured for c4ai. Current LLM_PROVIDER: $provider"
}

$apiKey = [string]$environment["LLM_C4AI_API_KEY"]
if ([string]::IsNullOrWhiteSpace($apiKey)) {
    $apiKey = [string]$environment["LLM_API_KEY"]
}
if ([string]::IsNullOrWhiteSpace($apiKey)) {
    throw "Neither LLM_C4AI_API_KEY nor LLM_API_KEY is configured in the project .env."
}

$baseUrl = [string]$environment["LLM_C4AI_BASE_URL"]
if ([string]::IsNullOrWhiteSpace($baseUrl)) {
    $baseUrl = "https://c4ai.ccccltd.cn/api/compatible/v1"
}
$model = [string]$environment["LLM_C4AI_MODEL"]
if ([string]::IsNullOrWhiteSpace($model)) {
    $model = "jiaorong-deepseek-v4-pro"
}
$timeout = [string]$environment["LLM_TIMEOUT_SECONDS"]
if ([string]::IsNullOrWhiteSpace($timeout)) {
    $timeout = "60"
}
$maxTokens = [string]$environment["LLM_MAX_TOKENS"]
if ([string]::IsNullOrWhiteSpace($maxTokens)) {
    $maxTokens = "1600"
}
$jsonMode = [string]$environment["LLM_JSON_MODE"]
if ([string]::IsNullOrWhiteSpace($jsonMode)) {
    $jsonMode = "true"
}
$plannerMode = [string]$environment["LLM_PLANNER_MODE"]
if ([string]::IsNullOrWhiteSpace($plannerMode)) {
    $plannerMode = "hybrid"
}

New-Item -ItemType Directory -Path $bundleRoot -Force | Out-Null
$bundleDeploy = Join-Path $bundleRoot "deploy"
New-Item -ItemType Directory -Path $bundleDeploy -Force | Out-Null

Copy-Item -LiteralPath $sourceCompose -Destination (Join-Path $bundleRoot "compose.yaml") -Force
Copy-Item -LiteralPath $sourceGpuCompose `
    -Destination (Join-Path $bundleRoot "compose.gpu.yaml") -Force
Copy-Item -LiteralPath $sourceMediaMtx -Destination (Join-Path $bundleDeploy "mediamtx.yml") -Force
Copy-Item -LiteralPath $sourceVideo -Destination (Join-Path $bundleRoot "demo.mp4") -Force
Copy-Item -LiteralPath $sourceDeploymentGuide -Destination (Join-Path $bundleRoot "README.md") -Force
Copy-Item -LiteralPath $sourceUploadChecklist `
    -Destination (Join-Path $bundleRoot "UPLOAD_CHECKLIST.md") -Force

$appEnvLines = @(
    "# Server runtime secrets. Do not commit or bake this file into an image.",
    "LLM_PROVIDER=c4ai",
    "LLM_C4AI_API_KEY=$apiKey",
    "LLM_C4AI_BASE_URL=$baseUrl",
    "LLM_C4AI_MODEL=$model",
    "LLM_TIMEOUT_SECONDS=$timeout",
    "LLM_MAX_TOKENS=$maxTokens",
    "LLM_JSON_MODE=$jsonMode",
    "LLM_PLANNER_MODE=$plannerMode",
    "MAIN_MONITOR_RTSP_URL=rtsp://rtsp:8554/main-monitor"
)
$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines(
    (Join-Path $bundleRoot "app.env"),
    $appEnvLines,
    $utf8WithoutBom
)

if ($BuildImage) {
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    if ($null -eq $docker) {
        throw "Docker CLI is not installed; belt-agent.tar could not be built."
    }
    Push-Location $repositoryRoot
    try {
        & $docker.Source build `
            --build-arg BUILDKIT_INLINE_CACHE=1 `
            --file (Join-Path $repositoryRoot "Dockerfile.gpu") `
            --tag belt-agent:latest `
            $repositoryRoot
        if ($LASTEXITCODE -ne 0) {
            throw "docker build failed with exit code $LASTEXITCODE"
        }
        & $docker.Source save -o $sourceImageArchive belt-agent:latest
        if ($LASTEXITCODE -ne 0) {
            throw "docker save failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

$imageIncluded = Test-Path -LiteralPath $sourceImageArchive -PathType Leaf
if ($imageIncluded) {
    Copy-Item -LiteralPath $sourceImageArchive `
        -Destination (Join-Path $bundleRoot "belt-agent.tar") -Force
}

$result = [ordered]@{
    ok = $true
    bundle_directory = $bundleRoot
    app_env = "prepared"
    compose = "prepared"
    mediamtx_config = "prepared"
    demo_video = "prepared"
    demo_video_bytes = (Get-Item -LiteralPath (Join-Path $bundleRoot "demo.mp4")).Length
    image_archive = if ($imageIncluded) { "prepared" } else { "missing_docker_build_required" }
}
$result | ConvertTo-Json
