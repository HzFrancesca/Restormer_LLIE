# 运行所有库的MACs/FLOPs测试
# 测试fvcore, thop, torchprofile, torchinfo四个库返回的具体是什么

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "运行所有库的MACs/FLOPs验证测试" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# 测试1: fvcore
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host "测试 1/4: fvcore" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow
python scripts/test_fvcore_macs_flops.py
Write-Host ""

# 测试2: thop
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host "测试 2/4: thop" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow
python scripts/test_thop_macs_flops.py
Write-Host ""

# 测试3: torchprofile
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host "测试 3/4: torchprofile" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow
python scripts/test_torchprofile_macs_flops.py
Write-Host ""

# 测试4: torchinfo详细测试
Write-Host "================================================================================" -ForegroundColor Yellow
Write-Host "测试 4/4: torchinfo (详细)" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Yellow
python scripts/test_torchinfo_detailed.py
Write-Host ""

Write-Host "================================================================================" -ForegroundColor Green
Write-Host "所有测试完成！" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
