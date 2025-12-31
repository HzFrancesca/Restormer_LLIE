# 项目结构

```
├── basicsr/                    # 核心框架
│   ├── data/                   # 数据集类 (*_dataset.py)
│   ├── metrics/                # 指标实现 (PSNR, SSIM, NIQE, FID)
│   ├── models/
│   │   ├── archs/              # 网络架构 (*_arch.py)
│   │   │   ├── restormer_arch.py           # 基础 Restormer
│   │   │   ├── dino_guided_restormer_arch.py  # DINO 引导变体
│   │   │   └── extra_attention_*.py        # 替代注意力机制
│   │   ├── losses/             # 损失函数
│   │   ├── base_model.py       # 基础模型类
│   │   └── image_restoration_model.py  # 训练/验证逻辑
│   ├── utils/                  # 工具函数 (日志、配置、图像处理)
│   ├── train.py                # 训练入口
│   └── test.py                 # 测试入口
│
├── LLIE/                       # 低光照图像增强任务
│   ├── Options/                # YAML 配置文件
│   ├── test.py                 # 任务专用测试脚本
│   └── utils.py                # 任务工具函数
│
├── datasets/                   # 数据集存储
│   └── LOL-v2/                 # LOL-v2 数据集
│       ├── Real_captured/      # 真实拍摄的图像对
│       └── Synthetic/          # 合成图像对
│
├── scripts/                    # 工具脚本
│   ├── flops/                  # FLOPs 计算工具
│   └── metrics/                # 指标计算脚本
│
└── md/                         # 文档
```

## 模块注册模式
框架通过 `__init__.py` 文件实现动态模块注册：
- **架构**: `basicsr/models/archs/` 中以 `_arch.py` 结尾的文件
- **模型**: `basicsr/models/` 中以 `_model.py` 结尾的文件
- **数据集**: `basicsr/data/` 中以 `_dataset.py` 结尾的文件

类通过 YAML 配置中的 `type` 字段实例化（如 `type: Restormer`）。

## 关键约定
- 架构类放在 `basicsr/models/archs/<name>_arch.py`
- 模型类放在 `basicsr/models/<name>_model.py`
- 数据集类放在 `basicsr/data/<name>_dataset.py`
- 配置文件放在 `LLIE/Options/`
- 训练输出自动保存到 `experiments/`
