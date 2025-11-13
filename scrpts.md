```
$env:CUDA_VISIBLE_DEVICES = "2"   # 只用第 0 块卡
python basicsr\train.py -opt e:\2024_HZF\Restormer_LLIE\LLIE\Options\LowLight_Restormer_128_2.yml


set CUDA_VISIBLE_DEVICES=2
python basicsr\train.py -opt e:\2024_HZF\Restormer_LLIE\LLIE\Options\LowLight_Restormer_128_2.yml


python LLIE/test.py  --input_dir datasets/LOL-v2/Real_captured/Test/Low/ --result_dir results/LowLight_Restormer_128_2_60k/   --weights experiments/LowLight_Restormer_128_2_60k_MDTA/models/net_g_60000.pth   --opt LLIE/Options/LowLight_Restormer_128_2_60k.yml

python LLIE/metrics_cal.py  -dirA datasets/LOL-v2/Real_captured/Test/Normal -dirB results/LowLight_Restormer_128_2_60k -type png --use_gpu

```
