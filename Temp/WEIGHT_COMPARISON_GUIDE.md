# 权重文件比较工具使用指南

## 工具说明

提供了两个版本的权重比较工具：

1. **`compare_weights_simple.py`** - 简化版（推荐）
   - 无需额外依赖
   - 只需要 PyTorch
   - 输出清晰简洁

2. **`compare_weights.py`** - 完整版
   - 支持彩色输出（需要 colorama）
   - 功能更丰富
   - 界面更美观

## 安装依赖

### 简化版（推荐）

```bash
# 无需额外安装，只要有 PyTorch 即可
```

### 完整版（可选）

```bash
pip install colorama
```

## 使用方法

### 1. 比较两个权重文件

```bash
# 使用简化版
python compare_weights_simple.py model1.pth model2.pth

# 使用完整版
python compare_weights.py --file1 model1.pth --file2 model2.pth
```

**示例输出：**

```
================================================================================================
比较 模型1 vs 模型2
================================================================================================

==================================================
【1. 键名比较】
==================================================
模型1 键数量: 150
模型2 键数量: 155
共同键数量: 145

仅在 模型1 中存在的键 (5 个):
  - encoder.attn.qkv_dwconv.weight [shape: (144, 48, 3, 3)]
  - encoder.attn.qkv_dwconv.bias [shape: (144,)]
  ...

仅在 模型2 中存在的键 (10 个):
  - encoder.attn.mdc_q.weight [shape: (48, 48, 3, 3)]
  - encoder.attn.mdc_k.weight [shape: (48, 48, 3, 3)]
  ...

==================================================
【2. 形状比较】
==================================================
[OK] 所有共同键的形状都匹配!
形状匹配的键: 145 个

==================================================
【3. 数值比较】
==================================================
完全相同的键: 0 个
极小差异的键: 0 个
明显差异的键: 145 个

================================================================================================
【比较总结】
================================================================================================
[×] 键名不同 (总差异: 15 个)
[OK] 所有共同键的形状都匹配
[~] 数值有差异 (145 个有明显差异)

================================================================================================
[结论] 两个权重文件的结构不同！
       原因: 键名不同
================================================================================================
```

### 2. 查看单个权重文件信息

```bash
# 简化版
python compare_weights_simple.py model.pth

# 完整版
python compare_weights.py --file1 model.pth
```

**示例输出：**

```
================================================================================================
模型1 权重信息
================================================================================================

总键数量: 150

序号   键名                                                                     形状                      参数量
--------------------------------------------------------------------------------------------------------------------
1      patch_embed.proj.weight                                                  (48, 3, 3, 3)             1,296
2      encoder_level1.0.norm1.body.weight                                       (48,)                        48
3      encoder_level1.0.attn.qkv.weight                                         (144, 48, 1, 1)           6,912
4      encoder_level1.0.attn.temperature                                        (1, 1, 1)                     1
...
--------------------------------------------------------------------------------------------------------------------

总参数数量: 26,124,320
张量数量: 150
```

### 3. 指定模型名称

```bash
# 简化版
python compare_weights_simple.py mdta.pth hta.pth --name1 "MDTA模型" --name2 "HTA模型"

# 完整版
python compare_weights.py --file1 mdta.pth --file2 hta.pth --name1 "MDTA模型" --name2 "HTA模型"
```

## 实际应用场景

### 场景1: 验证 MDTA 和 HTA 权重兼容性

```bash
python compare_weights_simple.py experiments/LowLight_Restormer/models/net_g_100000.pth experiments/LowLight_Restormer_HTA/models/net_g_100000.pth --name1 "MDTA权重" --name2 "HTA权重"
```

这将显示：

- ✓ 哪些层的权重可以共享（qkv, project_out, temperature）
- ✗ 哪些层是特有的（MDTA的qkv_dwconv，HTA的mdc_q/k/v）
- 是否可以使用 `strict=False` 加载

### 场景2: 检查训练前后的权重变化

```bash
python compare_weights_simple.py pretrained_model.pth experiments/train/models/net_g_latest.pth --name1 "预训练权重" --name2 "训练后权重"
```

### 场景3: 验证 EMA 权重

```bash
# 修改脚本以比较同一文件中的 params 和 params_ema
```

## 输出说明

### 键名比较

- **共同键**: 两个模型都有的参数
- **仅在模型1**: 只有模型1有的参数（例如MDTA的qkv_dwconv）
- **仅在模型2**: 只有模型2有的参数（例如HTA的mdc_q/k/v）

### 形状比较

- **形状匹配**: 共同键的张量形状是否一致
- **形状不匹配**: 如果有，会列出详细信息

### 数值比较

- **完全相同**: torch.equal() 返回 True
- **极小差异**: 最大差异 < 1e-6（通常是数值精度问题）
- **有明显差异**: 权重值真的不同（正常训练情况）

### 结论

- **结构完全相同**: 可以使用 `strict=False` 互相加载
- **结构不同**: 需要修改模型或权重文件

## 理解 MDTA vs HTA 的权重差异

### 共享的权重（可复用）

```
✓ attn.qkv.weight          - 生成Q,K,V的初始投影
✓ attn.project_out.weight  - 注意力输出的投影
✓ attn.temperature         - 温度参数
```

### MDTA 特有的权重

```
✗ attn.qkv_dwconv.weight   - Depthwise卷积
✗ attn.qkv_dwconv.bias     - 偏置
```

### HTA 特有的权重

```
✗ attn.mdc_q.weight        - Q的多尺度卷积
✗ attn.mdc_k.weight        - K的多尺度卷积  
✗ attn.mdc_v.*             - V的多尺度卷积模块
```

## 配置 strict_load_g

根据比较结果，在配置文件中设置：

```yaml
# 如果结构完全相同
path:
  pretrain_network_g: path/to/weights.pth
  strict_load_g: true

# 如果结构不同但想加载共同部分
path:
  pretrain_network_g: path/to/weights.pth
  strict_load_g: false  # 只加载匹配的权重
```

## 常见问题

### Q1: 为什么 MDTA 权重可以加载到 HTA 模型？

**A:** 因为它们有共同的核心权重（qkv, project_out），当使用 `strict=False` 时，PyTorch 会：

1. 加载所有匹配的权重
2. 忽略文件中多余的权重（如qkv_dwconv）
3. 对缺失的权重使用默认初始化（如mdc_q/k/v）

### Q2: 这样做性能会下降吗？

**A:** 初始时可能略有下降，因为 HTA 特有的层是随机初始化的。但这些层会在训练中快速学习，通常几千次迭代后就能恢复性能。

### Q3: 反过来可以吗？（HTA → MDTA）

**A:** 可以！同样使用 `strict=False`，会加载共同的权重，MDTA特有的层随机初始化。

### Q4: 如何确认权重加载成功？

**A:** 查看训练日志，会显示：

```
Loading: params_ema does not exist, use params.
Current net - loaded net:
  attn.mdc_q.weight
  attn.mdc_k.weight
  ...
Loaded net - current net:
  attn.qkv_dwconv.weight
  ...
```

## 高级用法

### 导出权重键名到文件

```bash
python compare_weights_simple.py model.pth > weights_info.txt
```

### 批量比较多个权重文件

创建脚本 `batch_compare.sh`：

```bash
#!/bin/bash
for file in experiments/*/models/net_g_*.pth; do
    echo "=== Comparing with $file ==="
    python compare_weights_simple.py base_model.pth "$file"
    echo ""
done
```

## 参考

- PyTorch 权重加载: <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict>
- MDTA 实现: `basicsr/models/archs/restormer_arch.py`
- HTA 实现: `basicsr/models/archs/extra_attention.py`
