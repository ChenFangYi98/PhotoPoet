param(
    [switch]$SmokeTest,
    [switch]$PrepareOnly,
    [string]$Resume = ""
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = (Get-Command python -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath $pythonExe)) { throw "Python environment not found: $pythonExe" }
$env:PYTHONIOENCODING = "utf-8"
$trainingArgs = @((Join-Path $PSScriptRoot "train_full_sft.py"))
if ($Resume) {
    $trainingArgs += @("--resume", $Resume)
} else {
    $trainingArgs += @(
        "--init_weight_path", (Join-Path $projectRoot "out\continue_pretrain\poetry_gujilab_20261022_190751_427063\pretrain_poetry_gujilab_768_moe_best.pth"),
        "--train_data_path", (Join-Path $projectRoot "dataset\sft_t2t_with_poetry\sft_t2t_with_poetry_train.jsonl"),
        "--valid_data_path", (Join-Path $projectRoot "dataset\sft_t2t_with_poetry\sft_t2t_with_poetry_valid.jsonl"),
        "--save_dir", (Join-Path $projectRoot "out\full_sft"), "--save_weight", "full_sft_poetry",
        "--epochs", "1", "--batch_size", "8", "--accumulation_steps", "8",
        "--learning_rate", "1e-5", "--warmup_ratio", "0.03", "--min_lr_ratio", "0.1",
        "--weight_decay", "0.01", "--grad_clip", "1.0",
        "--hidden_size", "768", "--num_hidden_layers", "8", "--use_moe", "1",
        "--max_seq_len", "1024", "--device", "cuda:0", "--dtype", "bfloat16", "--export_dtype", "float32",
        "--num_workers", "0", "--log_interval", "50", "--save_interval", "1000", "--eval_interval", "1000",
        "--validation_samples", "1024", "--validation_batch_size", "8", "--seed", "42"
    )
}
if ($SmokeTest) { $trainingArgs += "--smoke_test" }
if ($PrepareOnly) { $trainingArgs += "--prepare_only" }
Push-Location -LiteralPath $projectRoot
try {
    & $pythonExe @trainingArgs
    if ($LASTEXITCODE -ne 0) { throw "SFT exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
