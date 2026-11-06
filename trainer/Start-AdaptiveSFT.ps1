param(
    [ValidateSet('uniform','static','dynamic')][string]$WeightMode = 'dynamic',
    [int]$MaxUpdates = 0,
    [int]$Epochs = 1,
    [string]$RunName = '',
    [string]$InitWeightPath = '',
    [int]$StopAfterUpdates = 0,
    [switch]$PrepareOnly,
    [switch]$SmokeTest,
    [switch]$VerifySourceSha256,
    [ValidateSet(0,1)][int]$NormTopkProb = 0,
    [ValidateSet('legacy','straight_through')][string]$MoeRouterGradient = 'legacy',
    [string]$Resume = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = (Get-Command python -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath $pythonExe)) { throw "Python not found: $pythonExe" }
$env:PYTHONIOENCODING = 'utf-8'
# All paths follow the project directory; never overwrite old SFT runs.
$trainingArgs = @((Join-Path $PSScriptRoot 'train_adaptive_sft.py'))
if ($Resume) {
    $trainingArgs += @('--resume', $Resume)
} else {
    $trainingArgs += @(
        '--manifest', (Join-Path $projectRoot 'dataset\sft_t2t_with_poetry_clean_v1\adaptive_v3_ready\manifest.json'),
        '--init_weight_path', (Join-Path $projectRoot 'out\continue_pretrain\poetry_gujilab_20261022_190751_427063\pretrain_poetry_gujilab_768_moe_best.pth'),
        '--output_dir', (Join-Path $projectRoot 'out\adaptive_sft'),
        '--weight_mode', $WeightMode, '--max_updates', "$MaxUpdates", '--epochs', "$Epochs",
        '--examples_per_update', '64', '--window_batch_size', '2', '--router_batch_size', '8',
        '--max_seq_len', '2048', '--context_overlap', '512', '--context_policy', 'full_context',
        '--norm_topk_prob', "$NormTopkProb", '--moe_router_gradient', $MoeRouterGradient,
        '--learning_rate', '1e-5', '--router_learning_rate', '3e-4', '--warmup_ratio', '0.03', '--min_lr_ratio', '0.1',
        '--weight_decay', '0.01', '--grad_clip', '1', '--fact_boost', '0.6', '--max_sample_weight', '1.8',
        '--router_loss_weight', '0.05', '--router_warmup_updates', '100', '--router_min_accuracy', '0.75',
        '--validation_examples', '128', '--generation_per_source', '32', '--generation_tokens', '32',
        '--open_generation_examples', '4', '--open_generation_samples', '2', '--open_generation_tokens', '128',
        '--clarification_validation_examples', '16',
        '--preservation_tolerance', '0.15', '--eval_interval', '1000', '--save_interval', '1000', '--log_interval', '50',
        '--early_stopping_patience', '0', '--hidden_size', '768', '--num_hidden_layers', '8', '--use_moe', '1',
        '--dtype', 'bfloat16', '--device', 'cuda:0', '--seed', '42'
    )
}
if ($InitWeightPath) {
    if ($Resume) { throw 'InitWeightPath starts a new run; do not combine it with Resume.' }
    # An explicit model-only warm start creates a fresh optimizer and validation baseline.
    $trainingArgs += @('--init_weight_path', $InitWeightPath)
}
# An optional bounded run uses the same batch and learning-rate budget as training.
# StopAfterUpdates is an absolute optimizer update, including when resuming.
if ($RunName -and -not $Resume) { $trainingArgs += @('--run_name', $RunName) }
if ($StopAfterUpdates -gt 0) { $trainingArgs += @('--stop_after_updates', "$StopAfterUpdates") }
if ($PrepareOnly) { $trainingArgs += '--prepare_only' }
if ($SmokeTest) { $trainingArgs += '--smoke_test' }
if ($VerifySourceSha256 -and -not $Resume) { $trainingArgs += '--verify_source_sha256' }
Push-Location -LiteralPath $projectRoot
try {
    & $pythonExe @trainingArgs
    if ($LASTEXITCODE -ne 0) { throw "Adaptive SFT exited with code $LASTEXITCODE" }
} finally { Pop-Location }
