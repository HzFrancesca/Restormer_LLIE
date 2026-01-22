# 可视化三种注意力机制的第一层 Encoder Block 0
# 包括: Patch Embedding, Block Input, Q, K, V, Attention Map, Attention*V

$IMAGES = @(
    "low00729.png",
    "low00323.png",
    "low00736.png"
)
$EXPERIMENTS_DIR = "experiments"
$OUTPUT_DIR = "visualization_first_block_jet"

# 三种注意力机制的 checkpoint 路径
$CHECKPOINTS = @{
    "HTA"  = "$EXPERIMENTS_DIR\LowLight_Restormer_128_2_60k_HTA\net_g_44000.pth"
    "WTA"  = "$EXPERIMENTS_DIR\LowLight_Restormer_128_2_60k_WTA\net_g_44000.pth"
    "MDTA" = "$EXPERIMENTS_DIR\LowLight_Restormer_128_2_60k_MDTA\net_g_44000.pth"
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Visualizing First Block (Encoder Level 1 Block 0)" -ForegroundColor Cyan
Write-Host "Images: $($IMAGES.Count) files" -ForegroundColor Yellow
Write-Host "Output: $OUTPUT_DIR" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

foreach ($IMAGE in $IMAGES) {
    Write-Host "========================================" -ForegroundColor Magenta
    Write-Host "Processing Image: $IMAGE" -ForegroundColor Magenta
    Write-Host "========================================" -ForegroundColor Magenta
    Write-Host ""
    
    foreach ($attn_type in @("MDTA", "HTA", "WTA")) {
        $checkpoint = $CHECKPOINTS[$attn_type]
        
        Write-Host "  Processing: $attn_type" -ForegroundColor Green
        Write-Host "  Checkpoint: $checkpoint" -ForegroundColor Gray
        
        if (-not (Test-Path $checkpoint)) {
            Write-Host "    [WARNING] Checkpoint not found: $checkpoint" -ForegroundColor Red
            continue
        }
        
        # 运行可视化
        python scripts/visualize_first_block_JET.py `
            --image $IMAGE `
            --checkpoint $checkpoint `
            --attn_type $attn_type `
            --no_resize `
            --output_dir $OUTPUT_DIR
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "    [SUCCESS] $attn_type completed" -ForegroundColor Green
        } else {
            Write-Host "    [ERROR] $attn_type failed" -ForegroundColor Red
        }
        Write-Host ""
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "All visualizations completed!" -ForegroundColor Green
Write-Host "Results saved to: $OUTPUT_DIR\" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
