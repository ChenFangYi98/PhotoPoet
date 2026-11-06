param(
    [string]$Python = '',
    [switch]$PrepareOnly,
    [switch]$SmokeTest,
    [switch]$EvalOnly,
    [string]$Resume = '',
    [string]$Device = '',
    [string]$SaveDir = '',
    [string]$Checkpoint = '',
    [string]$RunName = '',
    [int]$MaxSteps = 100,
    [double]$PoetryRatio = 0.5
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $mindchenPython = Join-Path $env:USERPROFILE 'anaconda3\envs\mindchen\python.exe'
    $Python = if (Test-Path -LiteralPath $mindchenPython) { $mindchenPython } else { (Get-Command python).Source }
}
$grpoArguments = @('-X', 'utf8', '-u', (Join-Path $PSScriptRoot 'train_grpo.py'))
if ($Resume) {
    $grpoArguments += @('--resume', $Resume)
} else {
    if ($PSBoundParameters.ContainsKey('MaxSteps')) { $grpoArguments += @('--max_steps', "$MaxSteps") }
    if ($PSBoundParameters.ContainsKey('PoetryRatio')) { $grpoArguments += @('--poetry_ratio', "$PoetryRatio") }
}
if ($PrepareOnly) { $grpoArguments += '--prepare_only' }
if ($SmokeTest) { $grpoArguments += '--smoke_test' }
if ($EvalOnly) { $grpoArguments += '--eval_only' }
if ($Device) { $grpoArguments += @('--device', $Device) }
if ($SaveDir) { $grpoArguments += @('--save_dir', $SaveDir) }
if ($Checkpoint) { $grpoArguments += @('--checkpoint', $Checkpoint) }
if ($RunName) { $grpoArguments += @('--run_name', $RunName) }
Push-Location -LiteralPath $projectRoot
try {
    & $Python @grpoArguments
    if ($LASTEXITCODE -ne 0) { throw "GRPO exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
