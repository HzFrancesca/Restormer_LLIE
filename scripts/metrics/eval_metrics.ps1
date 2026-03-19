<#
.SYNOPSIS
    批量指标评价工具 (Batch Metrics Evaluation Tool)
    
.DESCRIPTION
    本脚本用于对指定的增强结果目录进行批量质量指标评估。
    它会调用 Python 脚本计算 PSNR, SSIM, LPIPS, NIQE, MUSIQ, BRISQUE 等指标。
    评价结果将自动按子文件夹分类存放在 results\metrics 目录下。

.PARAMETER ResultsDir
    包含模型推理结果的根目录（文件夹内应包含各权重的子文件夹，或直接包含图像）。
    例如: .\results\Restormer_128_2_60k_HTA

.PARAMETER GtDir
    与之匹配的地面真值（Ground Truth）图像目录。
    默认: .\datasets\LOL-v2\Real_captured\Test\Normal\

.PARAMETER UseGpu
    是否使用 GPU 进行评价计算（推荐开启，MUSIQ 和 LPIPS 在 GPU 上速度更快）。
    默认: $true

.PARAMETER ImgExt
    评估图像的后缀名。
    默认: png

.EXAMPLE
    运行评价（默认路径）:
    .\scripts\metrics\eval_metrics.ps1 -ResultsDir ".\results\MyExperiment"

.EXAMPLE
    自定义 GT 路径并禁用 GPU:
    .\scripts\metrics\eval_metrics.ps1 -ResultsDir ".\results\MyExperiment" -GtDir "C:\Data\GT" -UseGpu $false
#>

param (
    [Parameter(Mandatory=$true)]
    [string]$ResultsDir,

    [Parameter(Mandatory=$false)]
    [string]$GtDir = ".\datasets\LOL-v2\Real_captured\Test\Normal\",

    [Parameter(Mandatory=$false)]
    [bool]$UseGpu = $true,

    [Parameter(Mandatory=$false)]
    [string]$ImgExt = "png"
)

# 确保在项目根目录下运行（检测必要的 python 脚本位置）
$PSScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$BatchEvalPy = Join-Path $PSScriptDir "batch_eval_metrics.py"

if (-not (Test-Path $BatchEvalPy)) {
    Write-Host "Error: Could not find $BatchEvalPy" -ForegroundColor Red
    exit 1
}

Write-Host "`n" + ("=" * 80) -ForegroundColor Magenta
Write-Host ">>> Starting Batch Metric Calculation" -ForegroundColor Magenta
Write-Host ">>> Results Dir: $ResultsDir" -ForegroundColor Yellow
Write-Host ">>> GT Dir:      $GtDir" -ForegroundColor Yellow
Write-Host ">>> Use GPU:     $UseGpu" -ForegroundColor Yellow
Write-Host ("=" * 80) -ForegroundColor Magenta

# 构建参数并运行 Python 脚本
$GpuFlag = if ($UseGpu) { "--use_gpu" } else { "" }

# 使用相对于项目根目录的路径运行，或者直接调用绝对路径
python $BatchEvalPy --results_dir "$ResultsDir" --gt_dir "$GtDir" --img_ext "$ImgExt" $GpuFlag

if ($LASTEXITCODE -eq 0) {
    $FolderName = Split-Path $ResultsDir -Leaf
    Write-Host "`nEvaluation Completed!" -ForegroundColor Green
    Write-Host "Detailed reports are saved in: results\metrics\$FolderName" -ForegroundColor Cyan
} else {
    Write-Host "`nError: Metric calculation failed." -ForegroundColor Red
}
