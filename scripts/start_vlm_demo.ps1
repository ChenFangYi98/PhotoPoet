param(
    [string]$Checkpoint = '',
    [string]$LoraPath = '',
    [switch]$MindChenIdentity,
    [int]$Port = 8888,
    [string]$Python = (Get-Command python -ErrorAction Stop).Source
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Checkpoint) {
    $latest = Get-ChildItem -LiteralPath (Join-Path $projectRoot 'out') -Filter '*sft*.pth' |
        Where-Object { $_.Name -notmatch '^(smoke_)|_resume\.pth$' } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) { throw 'No SFT weights found in out/. Pass -Checkpoint explicitly.' }
    $Checkpoint = $latest.FullName
}
$Checkpoint = (Resolve-Path -LiteralPath $Checkpoint).Path
if (-not (Test-Path -LiteralPath $Python)) { throw "Python does not exist: $Python" }
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. If the demo is running, open http://127.0.0.1:$Port"
}
$logDir = Join-Path $projectRoot 'out\web_demo'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$serverArgs = @(
    '-X', 'utf8', '-u', ('"' + (Join-Path $PSScriptRoot 'web_demo_vlm.py') + '"'),
    '--checkpoint', ('"' + $Checkpoint + '"'),
    '--server_name', '127.0.0.1', '--server_port', $Port,
    '--max_new_tokens', '512'
)
if ($LoraPath) {
    $LoraPath = (Resolve-Path -LiteralPath $LoraPath).Path
    $serverArgs += @('--lora_path', ('"' + $LoraPath + '"'))
}
if ($MindChenIdentity) { $serverArgs += '--mindchen_identity' }
$process = Start-Process -FilePath $Python -ArgumentList $serverArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'stdout.log') -RedirectStandardError (Join-Path $logDir 'stderr.log')
Write-Output "PID: $($process.Id)"
Write-Output "Checkpoint: $Checkpoint"
Write-Output "Open after startup: http://127.0.0.1:$Port"
Write-Output "Logs: $logDir"
