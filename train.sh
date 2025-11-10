#!/usr/bin/env bash

CONFIG=$1

python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt $CONFIG --launcher pytorch



$env:CUDA_VISIBLE_DEVICES = "2"   # 只用第 0 块卡
python basicsr\train.py -opt e:\2024_HZF\Restormer_LLIE\LLIE\Options\LowLight_Restormer_128_2.yml


set CUDA_VISIBLE_DEVICES=2
python basicsr\train.py -opt e:\2024_HZF\Restormer_LLIE\LLIE\Options\LowLight_Restormer_128_2.yml