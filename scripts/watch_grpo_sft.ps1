param([Parameter(Mandatory = $true)][string]$RunDir)

$ErrorActionPreference = 'Stop'
$run = Get-Content -LiteralPath (Join-Path $RunDir 'run.json') -Raw -Encoding utf8 | ConvertFrom-Json
$Host.UI.RawUI.WindowTitle = "MindChen-V SFT Training - PID $($run.process_id)"
$Host.UI.RawUI.BackgroundColor = 'Black'
$Host.UI.RawUI.ForegroundColor = 'Gray'
Clear-Host
Write-Host "MindChen-V SFT | Training PID: $($run.process_id)" -ForegroundColor Green
Write-Host "Dataset: $($run.dataset)"
Write-Host "Output:  $($run.output_weight)"
Write-Host 'Live logs below. Closing this viewer leaves background training running.'
Write-Host ''

function Open-LogReader([string]$Path) {
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    return [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8)
}

$stdoutReader = Open-LogReader $run.stdout
$stderrReader = Open-LogReader $run.stderr
try {
    while ($true) {
        while ($null -ne ($logLine = $stdoutReader.ReadLine())) { Write-Host $logLine }
        while ($null -ne ($errorLine = $stderrReader.ReadLine())) { Write-Host $errorLine -ForegroundColor Yellow }
        if (-not (Get-Process -Id $run.process_id -ErrorAction SilentlyContinue)) {
            Write-Host ''
            Write-Host 'Training process has exited. Check the logs above for completion or errors.' -ForegroundColor Yellow
            break
        }
        Start-Sleep -Seconds 1
    }
} finally {
    $stdoutReader.Dispose()
    $stderrReader.Dispose()
}
