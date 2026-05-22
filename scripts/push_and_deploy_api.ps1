param(
  [string]$Region = "ap-south-1",
  [string]$AccountId = "053849129634",
  [string]$Repository = "rag-api-service",
  [string]$Cluster = "rag-cluster",
  [string]$Service = "rag-api-service-call",
  [string]$Tag = "latest",
  [string[]]$SetEnv = @(),
  [switch]$SkipBuild,
  [switch]$WaitStable
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$imageUri = "$AccountId.dkr.ecr.$Region.amazonaws.com/$Repository`:$Tag"

if (-not $SkipBuild) {
  docker build -f api.Dockerfile -t "$Repository`:$Tag" .
}

aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin "$AccountId.dkr.ecr.$Region.amazonaws.com"
docker tag "$Repository`:$Tag" $imageUri
docker push $imageUri

$deployArgs = @(
  "-ExecutionPolicy", "Bypass",
  "-File", "scripts/deploy_ecs_api.ps1",
  "-Region", $Region,
  "-Cluster", $Cluster,
  "-Service", $Service,
  "-Image", $imageUri
)

if ($WaitStable) {
  $deployArgs += "-WaitStable"
}

foreach ($kv in $SetEnv) {
  $deployArgs += @("-SetEnv", $kv)
}

powershell @deployArgs
