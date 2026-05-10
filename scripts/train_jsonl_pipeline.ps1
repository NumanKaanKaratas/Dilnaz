param(
    [string]$TrainFile = ".\TrainDatas\Test1.jsonl",
    [string]$EvalFile = ".\TrainDatas\TestCümleler.jsonl",
    [string]$DilOutputDir = ".\checkpoints\Dil",
    [string]$WriterOutputDir = ".\checkpoints\Dil",
    [string]$NazOutputDir = ".\checkpoints\Naz",
    [ValidateSet("streaming", "resident")]
    [string]$DataMode = "streaming",
    [int]$DilSteps = 30000,
    [int]$WriterSteps = 30000,
    [int]$NazSteps = 30000,
    [int]$DilBatchSize = 64,
    [int]$WriterBatchSize = 64,
    [int]$NazBatchSize = 8,
    [int]$SequenceLength = 258,
    [int]$LogEvery = 50,
    [int]$EvalEvery = 500,
    [int]$CheckpointEvery = 5000,
    [ValidateSet("off", "default", "reduce-overhead", "max-autotune")]
    [string]$CompileMode = "reduce-overhead",
    [switch]$Bf16
)

$ErrorActionPreference = "Stop"

$bf16Args = @()
if ($Bf16) {
    $bf16Args = @("--bf16")
}

function Invoke-CheckedPython {
    python @args
    if ($LASTEXITCODE -ne 0) {
        throw "python failed with exit code $LASTEXITCODE"
    }
}

Invoke-CheckedPython .\dilnaz\train\train_dil.py `
    --train-file $TrainFile `
    --eval-file $EvalFile `
    --output-dir $DilOutputDir `
    --data-mode $DataMode `
    --max-steps $DilSteps `
    --batch-size $DilBatchSize `
    --eval-batch-size $DilBatchSize `
    --writer-loss-weight 0.0 `
    --log-every $LogEvery `
    --eval-every $EvalEvery `
    --checkpoint-every $CheckpointEvery `
    --compile-mode $CompileMode `
    @bf16Args

Invoke-CheckedPython .\dilnaz\train\train_dil_writer.py `
    --train-file $TrainFile `
    --eval-file $EvalFile `
    --checkpoint "$DilOutputDir\checkpoint.pt" `
    --output-dir $WriterOutputDir `
    --data-mode $DataMode `
    --max-steps $WriterSteps `
    --batch-size $WriterBatchSize `
    --eval-batch-size $WriterBatchSize `
    --log-every $LogEvery `
    --eval-every $EvalEvery `
    --checkpoint-every $CheckpointEvery `
    --compile-mode $CompileMode `
    @bf16Args

Invoke-CheckedPython .\dilnaz\train\train_naz.py `
    --train-file $TrainFile `
    --eval-file $EvalFile `
    --dil-checkpoint-dir $WriterOutputDir `
    --output-dir $NazOutputDir `
    --data-mode $DataMode `
    --max-steps $NazSteps `
    --batch-size $NazBatchSize `
    --eval-batch-size $NazBatchSize `
    --sequence-length $SequenceLength `
    --log-every $LogEvery `
    --eval-every $EvalEvery `
    --checkpoint-every $CheckpointEvery `
    --compile-mode $CompileMode `
    @bf16Args
