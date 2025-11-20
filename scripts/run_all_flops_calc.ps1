# PowerShell 脚本：运行所有 FLOPs 计算方法并生成对比结果
# 使用方法: .\run_all_flops_calc.ps1
# 使用自定义输入尺寸: .\run_all_flops_calc.ps1 -Batch 1 -Channels 3 -Height 256 -Width 256

# 参数定义
param(
    [int]$Batch = 1,
    [int]$Channels = 3,
    [int]$Height = 256,
    [int]$Width = 256
)

# 设置错误处理
$ErrorActionPreference = "Stop"

# 获取脚本所在目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# 构建输入尺寸显示字符串
$InputSizeDisplay = "($Batch, $Channels, $Height, $Width)"

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Restormer FLOPs 和参数量计算 - 四种方法对比" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "输入尺寸: $InputSizeDisplay" -ForegroundColor White
Write-Host ""

# 激活 conda 环境（如果需要）
Write-Host "[1/5] 激活 conda 环境 dp311..." -ForegroundColor Yellow
$condaPath = "C:\Users\$env:USERNAME\anaconda3\Scripts\conda.exe"
if (Test-Path $condaPath) {
    & $condaPath activate dp311
    Write-Host "✓ Conda 环境已激活" -ForegroundColor Green
} else {
    Write-Host "! 未找到 conda，尝试直接使用 python..." -ForegroundColor Yellow
}
Write-Host ""

# 定义结果文件路径
$ResultsDir = $ScriptDir
$ComparisonFile = Join-Path $ResultsDir "results_comparison.txt"
$SummaryFile = Join-Path $ResultsDir "results_summary.txt"

# 初始化结果数组
$results = @()

# 1. 运行 thop 计算
Write-Host "[2/5] 运行 thop 计算..." -ForegroundColor Yellow
try {
    $thopScript = Join-Path $ScriptDir "calc_flops_thop.py"
    python $thopScript --input-size $Batch $Channels $Height $Width
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ thop 计算完成" -ForegroundColor Green
        $results += "thop"
    } else {
        Write-Host "✗ thop 计算失败" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ thop 计算出错: $_" -ForegroundColor Red
}
Write-Host ""

# 2. 运行 fvcore 计算
Write-Host "[3/5] 运行 fvcore 计算..." -ForegroundColor Yellow
try {
    $fvcoreScript = Join-Path $ScriptDir "calc_flops_fvcore.py"
    python $fvcoreScript --input-size $Batch $Channels $Height $Width
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ fvcore 计算完成" -ForegroundColor Green
        $results += "fvcore"
    } else {
        Write-Host "✗ fvcore 计算失败" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ fvcore 计算出错: $_" -ForegroundColor Red
}
Write-Host ""

# 3. 运行 torchinfo 计算
Write-Host "[4/5] 运行 torchinfo 计算..." -ForegroundColor Yellow
try {
    $torchinfoScript = Join-Path $ScriptDir "calc_flops_torchinfo.py"
    python $torchinfoScript --input-size $Batch $Channels $Height $Width
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ torchinfo 计算完成" -ForegroundColor Green
        $results += "torchinfo"
    } else {
        Write-Host "✗ torchinfo 计算失败" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ torchinfo 计算出错: $_" -ForegroundColor Red
}
Write-Host ""

# 4. 运行 torchprofile 计算
Write-Host "[5/5] 运行 torchprofile 计算..." -ForegroundColor Yellow
try {
    $torchprofileScript = Join-Path $ScriptDir "calc_flops_torchprofile.py"
    python $torchprofileScript --input-size $Batch $Channels $Height $Width
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ torchprofile 计算完成" -ForegroundColor Green
        $results += "torchprofile"
    } else {
        Write-Host "✗ torchprofile 计算失败" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ torchprofile 计算出错: $_" -ForegroundColor Red
}
Write-Host ""

# 5. 生成对比结果
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "生成对比结果..." -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# 创建对比文件
$comparisonContent = @"
============================================================
Restormer FLOPs 和参数量计算 - 四种方法对比
============================================================
模型配置:
  - 输入尺寸: $InputSizeDisplay
  - dim: 48
  - num_blocks: [4, 6, 6, 8]
  - num_refinement_blocks: 4
  - heads: [1, 2, 4, 8]
  - ffn_expansion_factor: 2.66
  - LayerNorm_type: WithBias

计算时间: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
============================================================


"@

# 读取各个方法的结果
$methodFiles = @{
    "thop" = Join-Path $ResultsDir "results_thop.txt"
    "fvcore" = Join-Path $ResultsDir "results_fvcore.txt"
    "torchinfo" = Join-Path $ResultsDir "results_torchinfo.txt"
    "torchprofile" = Join-Path $ResultsDir "results_torchprofile.txt"
}

foreach ($method in $results) {
    $file = $methodFiles[$method]
    if (Test-Path $file) {
        $content = Get-Content $file -Raw -Encoding UTF8
        $comparisonContent += $content + "`n`n"
    }
}

# 添加对比总结
$comparisonContent += @"
============================================================
对比总结
============================================================

"@

# 解析并对比结果
$parsedResults = @{}
foreach ($method in $results) {
    $file = $methodFiles[$method]
    if (Test-Path $file) {
        $content = Get-Content $file -Encoding UTF8
        foreach ($line in $content) {
            # 匹配参数量（支持逗号分隔的数字和小数点）
            # 同时支持 "参数量" 和 "总参数量"
            if ($line -match "(总)?参数量.*\(([0-9,\.]+)\)") {
                $numStr = $matches[2] -replace ",", "" -replace "\.0*$", ""
                $parsedResults[$method + "_params"] = [long]$numStr
            }
            # 匹配FLOPs（支持逗号分隔的数字和小数点）
            if ($line -match "FLOPs.*\(([0-9,\.]+)\)") {
                $numStr = $matches[1] -replace ",", "" -replace "\.0*$", ""
                $parsedResults[$method + "_flops"] = [long]$numStr
            }
        }
    }
}

# 格式化对比表格
$comparisonContent += "方法对比:`n"
$comparisonContent += "-" * 60 + "`n"
$comparisonContent += "{0,-15} {1,20} {2,20}`n" -f "方法", "参数量", "FLOPs"
$comparisonContent += "-" * 60 + "`n"

foreach ($method in $results) {
    $params = if ($parsedResults.ContainsKey($method + "_params")) { "{0:N0}" -f $parsedResults[$method + "_params"] } else { "N/A" }
    $flops = if ($parsedResults.ContainsKey($method + "_flops")) { "{0:N0}" -f $parsedResults[$method + "_flops"] } else { "N/A" }
    $comparisonContent += "{0,-15} {1,20} {2,20}`n" -f $method, $params, $flops
}

$comparisonContent += "-" * 60 + "`n`n"

# 计算差异
if ($results.Count -ge 2) {
    $comparisonContent += "差异分析:`n"
    $comparisonContent += "-" * 60 + "`n"
    
    # 参数量差异
    $paramValues = @()
    foreach ($method in $results) {
        if ($parsedResults.ContainsKey($method + "_params")) {
            $paramValues += $parsedResults[$method + "_params"]
        }
    }
    
    if ($paramValues.Count -ge 2) {
        $maxParams = ($paramValues | Measure-Object -Maximum).Maximum
        $minParams = ($paramValues | Measure-Object -Minimum).Minimum
        $avgParams = ($paramValues | Measure-Object -Average).Average
        
        $comparisonContent += "参数量统计:`n"
        $comparisonContent += "  最大值: {0:N0}`n" -f $maxParams
        $comparisonContent += "  最小值: {0:N0}`n" -f $minParams
        $comparisonContent += "  平均值: {0:N0}`n" -f $avgParams
        if ($minParams -gt 0) {
            $paramDiff = (($maxParams - $minParams) / $minParams) * 100
            $comparisonContent += "  差异率: {0:F2}%`n" -f $paramDiff
        }
        $comparisonContent += "`n"
    }
    
    # FLOPs 差异
    $flopsValues = @()
    foreach ($method in $results) {
        if ($parsedResults.ContainsKey($method + "_flops")) {
            $flopsValues += $parsedResults[$method + "_flops"]
        }
    }
    
    if ($flopsValues.Count -ge 2) {
        $maxFlops = ($flopsValues | Measure-Object -Maximum).Maximum
        $minFlops = ($flopsValues | Measure-Object -Minimum).Minimum
        $avgFlops = ($flopsValues | Measure-Object -Average).Average
        
        $comparisonContent += "FLOPs 统计:`n"
        $comparisonContent += "  最大值: {0:N0}`n" -f $maxFlops
        $comparisonContent += "  最小值: {0:N0}`n" -f $minFlops
        $comparisonContent += "  平均值: {0:N0}`n" -f $avgFlops
        if ($minFlops -gt 0) {
            $flopsDiff = (($maxFlops - $minFlops) / $minFlops) * 100
            $comparisonContent += "  差异率: {0:F2}%`n" -f $flopsDiff
        }
    }
}

$comparisonContent += "`n" + "=" * 60 + "`n"
$comparisonContent += "所有计算完成！`n"
$comparisonContent += "=" * 60 + "`n"

# 保存对比结果
$comparisonContent | Out-File -FilePath $ComparisonFile -Encoding UTF8
Write-Host "✓ 对比结果已保存到: $ComparisonFile" -ForegroundColor Green

# 创建简洁的摘要文件
$summaryContent = @"
Restormer 模型复杂度计算摘要
============================================================
输入尺寸: $InputSizeDisplay
计算时间: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

"@

foreach ($method in $results) {
    $params = if ($parsedResults.ContainsKey($method + "_params")) { "{0:N0}" -f $parsedResults[$method + "_params"] } else { "N/A" }
    $flops = if ($parsedResults.ContainsKey($method + "_flops")) { "{0:N0}" -f $parsedResults[$method + "_flops"] } else { "N/A" }
    $summaryContent += "$method : 参数量 = $params, FLOPs = $flops`n"
}

$summaryContent | Out-File -FilePath $SummaryFile -Encoding UTF8
Write-Host "✓ 摘要结果已保存到: $SummaryFile" -ForegroundColor Green

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "所有任务完成！" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "结果文件:" -ForegroundColor Yellow
Write-Host "  - 详细对比: $ComparisonFile" -ForegroundColor White
Write-Host "  - 简要摘要: $SummaryFile" -ForegroundColor White
Write-Host "  - thop 结果: $(Join-Path $ResultsDir 'results_thop.txt')" -ForegroundColor White
Write-Host "  - fvcore 结果: $(Join-Path $ResultsDir 'results_fvcore.txt')" -ForegroundColor White
Write-Host "  - torchinfo 结果: $(Join-Path $ResultsDir 'results_torchinfo.txt')" -ForegroundColor White
Write-Host "  - torchprofile 结果: $(Join-Path $ResultsDir 'results_torchprofile.txt')" -ForegroundColor White
Write-Host ""
