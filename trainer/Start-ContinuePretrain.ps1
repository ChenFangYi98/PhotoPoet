param(
    [switch]$SmokeTest,
    [switch]$PrepareOnly,
    [string]$Resume = ""
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = (Get-Command python -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python environment not found: $pythonExe"
}
$env:PYTHONIOENCODING = "utf-8"
$trainingArgs = @((Join-Path $PSScriptRoot "train_continue_pretrain.py"))
if ($Resume) {
    $trainingArgs += @("--resume", $Resume)
} else {
    $trainingArgs += @(
        "--init_weight_path", (Join-Path $projectRoot "out\pretrain_768_moe.pth"),
        "--data_path", (Join-Path $projectRoot "dataset\pretrain_t2t_with_poetry_gujilab.jsonl"),
        "--save_dir", (Join-Path $projectRoot "out\continue_pretrain"),
        "--save_weight", "pretrain_poetry_gujilab",
        "--epochs", "1", "--batch_size", "16", "--accumulation_steps", "4",
        "--learning_rate", "5e-5", "--min_lr_ratio", "0.1", "--warmup_ratio", "0.03",
        "--weight_decay", "0.01", "--grad_clip", "1.0",
        "--hidden_size", "768", "--num_hidden_layers", "8", "--use_moe", "1",
        "--max_seq_len", "340", "--device", "cuda:0", "--dtype", "bfloat16",
        "--num_workers", "0", "--log_interval", "50", "--save_interval", "1000",
        "--eval_interval", "1000", "--validation_samples", "256", "--validation_batch_size", "8",
        "--seed", "42"
    )
}
if ($SmokeTest) { $trainingArgs += "--smoke_test" }
if ($PrepareOnly) { $trainingArgs += "--prepare_only" }
Push-Location -LiteralPath $projectRoot
try {
    & $pythonExe @trainingArgs
    if ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
