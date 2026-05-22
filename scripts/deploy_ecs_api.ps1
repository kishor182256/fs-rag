param(
  [string]$Region = "ap-south-1",
  [string]$Cluster = "rag-cluster",
  [string]$Service = "rag-api-service-call",
  [string]$TaskDefFile = "infra/ecs-api-task-definition.json",
  [string]$Image = "",
  [string[]]$SetEnv = @(),
  [switch]$WaitStable
)

$ErrorActionPreference = "Stop"

$pythonExe = "python"
$venvPython = Join-Path $PSScriptRoot "..\\.venv\\Scripts\\python.exe"
if (Test-Path $venvPython) {
  $pythonExe = (Resolve-Path $venvPython).Path
}

$cmd = @(
  "scripts/deploy_ecs_api.py",
  "--region", $Region,
  "--cluster", $Cluster,
  "--service", $Service,
  "--task-def-file", $TaskDefFile
)

if ($Image -ne "") {
  $cmd += @("--image", $Image)
}

foreach ($kv in $SetEnv) {
  $cmd += @("--set-env", $kv)
}

if ($WaitStable) {
  $cmd += "--wait-stable"
}

& $pythonExe @cmd
