param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$DataPath,
    [string]$EvalDataPath = '',
    [int]$Steps = 100,
    [ValidateSet('llm', 'vlm')][string]$Kind = 'llm',
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$entryName = if ($Kind -eq 'vlm') { 'trainer/train_lora_vlm.py' } else { 'trainer/train_lora_sft.py' }
$entry = Join-Path $root $entryName
$arguments = @($entry, '--checkpoint', $Checkpoint, '--data_path', $DataPath, '--max_steps', $Steps, '--export_merged')
if ($EvalDataPath) { $arguments += @('--eval_data_path', $EvalDataPath) }
& $Python @arguments
exit $LASTEXITCODE
