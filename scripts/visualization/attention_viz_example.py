import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

class MDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, return_internals=False):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q_vis, k_vis, v_vis = q.clone(), k.clone(), v.clone()

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        attn_v_raw = attn @ v  # Before rearrange
        out_before_proj = rearrange(
            attn_v_raw, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out_before_proj)

        if return_internals:
            return {
                "q": q_vis, "k": k_vis, "v": v_vis,
                "attn": attn, 
                "attn_v_before_rearrange": attn_v_raw,
                "attn_v_after_rearrange": out_before_proj,
                "output": out
            }
        return out


class HTA(nn.Module): # WxW Attention
    def __init__(self, dim, num_heads, bias):
        super(HTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, return_internals=False):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q_vis, k_vis, v_vis = q.clone(), k.clone(), v.clone()

        # HTA: 在每个宽度位置 w 上，对所有 (c*h) 特征做 attention
        # 这意味着：每列像素独立处理，列内的所有行和通道会相互混合
        q = rearrange(q, "b (head c) h w -> b head w (c h)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head w (c h)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head w (c h)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1)

        # attn: [b, head, w, w] - 每个宽度位置与其他宽度位置的关系
        # v: [b, head, w, (c*h)] - 每个宽度位置的特征向量
        # out: [b, head, w, (c*h)] - 加权后的特征
        attn_v_raw = attn @ v  # Before rearrange
        out_before_proj = rearrange(
            attn_v_raw, "b head w (c h) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )

        out = self.project_out(out_before_proj)

        if return_internals:
            return {
                "q": q_vis, "k": k_vis, "v": v_vis,
                "attn": attn, 
                "attn_v_before_rearrange": attn_v_raw,
                "attn_v_after_rearrange": out_before_proj,
                "output": out
            }
        return out


class WTA(nn.Module): # HxH Attention
    def __init__(self, dim, num_heads, bias=False):
        super(WTA, self).__init__()
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.num_heads = num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, return_internals=False):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q_vis, k_vis, v_vis = q.clone(), k.clone(), v.clone()

        v1 = rearrange(v, "b (head c) h w -> b head h (c w)", head=self.num_heads)
        q1 = rearrange(q, "b (head c) h w -> b head h (c w)", head=self.num_heads)
        k1 = rearrange(k, "b (head c) h w -> b head h (c w)", head=self.num_heads)

        q1 = F.normalize(q1, dim=-1)
        k1 = F.normalize(k1, dim=-1)
        
        attn = (q1 @ k1.transpose(-2, -1) * self.temperature).softmax(dim=-1)

        attn_v_raw = attn @ v1  # Before rearrange
        out_before_proj = rearrange(
            attn_v_raw, "b head h (c w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        out = self.project_out(out_before_proj)

        if return_internals:
            return {
                "q": q_vis, "k": k_vis, "v": v_vis,
                "attn": attn, 
                "attn_v_before_rearrange": attn_v_raw,
                "attn_v_after_rearrange": out_before_proj,
                "output": out
            }
        return out

# ==========================================
# 2. 辅助工具：Mock数字“4”与可视化保存
# ==========================================

def create_mock_digit_4(width=600, height=400):
    """
    使用 PIL 在黑色背景上绘制一个白色的数字 4。
    手动绘制线条以确保没有字体依赖问题。
    """
    # 1. 创建黑色背景
    img_pil = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img_pil)
    
    # 2. 设置线条参数
    line_width = max(1, min(width, height) // 10) # 笔画宽度
    
    # 3. 绘制数字 4 的三个主要部分
    # 坐标系：(0,0)在左上角
    
    # (A) 斜线：从上方中间偏右 -> 到中间偏左
    p1 = (width * 0.70, height * 0.10)
    p2 = (width * 0.20, height * 0.60)
    draw.line([p1, p2], fill=(255, 255, 255), width=line_width)
    
    # (B) 横线：穿过中间
    p3 = (width * 0.10, height * 0.60)
    p4 = (width * 0.90, height * 0.60)
    draw.line([p3, p4], fill=(255, 255, 255), width=line_width)
    
    # (C) 竖线：右侧垂直柱
    p5 = (width * 0.70, height * 0.10)
    p6 = (width * 0.70, height * 0.90)
    draw.line([p5, p6], fill=(255, 255, 255), width=line_width)
    
    # 4. 转换为 Tensor [1, 3, H, W]
    img = np.array(img_pil).astype(np.float32) / 255.0
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0)

def save_tensor_as_image(tensor, path, normalize=True, cmap='viridis', title=None, original_shape=None):
    """保存张量为图片，并在图片上添加标题和shape信息。支持 [C, H, W] 或 [H, W]"""
    original_tensor = tensor
    if tensor.dim() == 4: 
        tensor = tensor[0] # remove batch
    
    # 记录原始shape用于显示
    if original_shape is None:
        original_shape = list(original_tensor.shape)
    
    # 如果是多通道特征图 (C > 3)，取平均值变成热力图
    is_rgb = False
    if tensor.dim() == 3:
        if tensor.shape[0] == 3:
            # RGB Image
            is_rgb = True
            img_np = tensor.detach().cpu().numpy().transpose(1, 2, 0)
            if normalize:
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        else:
            # Feature map: average across channels
            tensor = tensor.mean(dim=0)
            img_np = tensor.detach().cpu().numpy()
            if normalize:
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    else:
        img_np = tensor.detach().cpu().numpy()
        if normalize:
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    # 使用 matplotlib 创建带标题的图片
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if is_rgb:
        ax.imshow(img_np)
    else:
        im = ax.imshow(img_np, cmap=cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.axis('off')
    
    # 添加标题和shape信息
    if title:
        shape_str = f"Shape: {original_shape}"
        stats_str = f"Min: {original_tensor.min():.4f}, Max: {original_tensor.max():.4f}, Mean: {original_tensor.mean():.4f}"
        full_title = f"{title}\n{shape_str}\n{stats_str}"
        ax.set_title(full_title, fontsize=10, pad=20)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def visualize_attention_block(model_name, internals, output_dir):
    """专门处理特定Attention模块的可视化逻辑"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 打印调试信息
    print(f"\n{model_name} - Shape Analysis:")
    print(f"  Q shape: {internals['q'].shape}")
    print(f"  V shape: {internals['v'].shape}")
    print(f"  Attention shape: {internals['attn'].shape}")
    print(f"  Attn*V (before rearrange) shape: {internals['attn_v_before_rearrange'].shape}")
    print(f"  Attn*V (after rearrange) shape: {internals['attn_v_after_rearrange'].shape}")
    print(f"  Output shape: {internals['output'].shape}")
    print(f"  Attn*V (before) stats: min={internals['attn_v_before_rearrange'].min():.4f}, max={internals['attn_v_before_rearrange'].max():.4f}, mean={internals['attn_v_before_rearrange'].mean():.4f}")
    print(f"  Attn*V (after) stats: min={internals['attn_v_after_rearrange'].min():.4f}, max={internals['attn_v_after_rearrange'].max():.4f}, mean={internals['attn_v_after_rearrange'].mean():.4f}")
    print(f"  Output stats: min={internals['output'].min():.4f}, max={internals['output'].max():.4f}, mean={internals['output'].mean():.4f}")
    
    # 1. 保存 Q, K, V (Feature Maps)
    save_tensor_as_image(
        internals['q'], 
        f"{output_dir}/1_Q.png",
        title=f"{model_name} - Query (Q)",
        original_shape=list(internals['q'].shape)
    )
    save_tensor_as_image(
        internals['k'], 
        f"{output_dir}/2_K.png",
        title=f"{model_name} - Key (K)",
        original_shape=list(internals['k'].shape)
    )
    save_tensor_as_image(
        internals['v'], 
        f"{output_dir}/3_V.png",
        title=f"{model_name} - Value (V)",
        original_shape=list(internals['v'].shape)
    )
    
    # 2. 保存 Attention Map
    attn = internals['attn'][0, 0] # 取 Batch 0, Head 0
    attn_full_shape = list(internals['attn'].shape)
    
    # 不同的注意力机制，Attention Map的物理意义不同，这里都用 inferno 色图展示
    save_tensor_as_image(
        attn, 
        f"{output_dir}/4_Attention_Map_Head0.png", 
        cmap='inferno',
        title=f"{model_name} - Attention Map (Head 0)",
        original_shape=attn_full_shape
    )
    
    # 3. 保存 Attention * V (Before Rearrange)
    save_tensor_as_image(
        internals['attn_v_before_rearrange'], 
        f"{output_dir}/5_Attn_times_V_before_rearrange.png",
        title=f"{model_name} - Attn @ V (Before Rearrange)",
        original_shape=list(internals['attn_v_before_rearrange'].shape)
    )
    
    # 4. 保存 Attention * V (After Rearrange, Before Projection)
    save_tensor_as_image(
        internals['attn_v_after_rearrange'], 
        f"{output_dir}/6_Attn_times_V_after_rearrange.png",
        title=f"{model_name} - Attn @ V (After Rearrange)",
        original_shape=list(internals['attn_v_after_rearrange'].shape)
    )
    
    # 5. 保存 Final Output (After Projection)
    save_tensor_as_image(
        internals['output'], 
        f"{output_dir}/7_Final_Output_after_proj.png",
        title=f"{model_name} - Final Output (After Projection)",
        original_shape=list(internals['output'].shape)
    )

# ==========================================
# 3. 主程序执行
# ==========================================

def main():
    # 配置
    DIM = 32        
    HEADS = 2
    WIDTH = 600
    HEIGHT = 400
    ROOT_DIR = "./attention_digit_viz"
    
    # 1. 准备数据：数字 4
    x = create_mock_digit_4(width=WIDTH, height=HEIGHT) # [1, 3, 400, 600]
    
    # 将输入通道映射到模型维度 (3 -> DIM)
    input_proj = nn.Conv2d(3, DIM, 1)
    # 初始化权重以保证初始特征不太混乱
    nn.init.xavier_normal_(input_proj.weight)
    
    x_in = input_proj(x)
    
    # 保存原始输入
    if not os.path.exists(ROOT_DIR): os.makedirs(ROOT_DIR)
    save_tensor_as_image(
        x, 
        f"{ROOT_DIR}/0_Input_Digit.png", 
        normalize=False,
        title="Input Image - Digit '4'",
        original_shape=list(x.shape)
    )
    
    # 2. 初始化模型
    models = {
        "MDTA (Channel)": MDTA(dim=DIM, num_heads=HEADS, bias=True),
        "HTA (Width)":    HTA(dim=DIM, num_heads=HEADS, bias=True),
        "WTA (Height)":   WTA(dim=DIM, num_heads=HEADS, bias=True)
    }
    
    # 3. 推理与保存
    print(f"开始生成可视化，输入为数字 '4'，结果保存至 {ROOT_DIR} ...")
    
    for name, model in models.items():
        print(f"Processing {name}...")
        safe_name = name.split(" ")[0] # MDTA, HTA, WTA
        
        # Forward
        internals = model(x_in, return_internals=True)
        
        # Save
        visualize_attention_block(safe_name, internals, f"{ROOT_DIR}/{safe_name}")

    print("完成！")
    print(f"查看 {ROOT_DIR}/0_Input_Digit.png 确认输入图像。")

if __name__ == "__main__":
    main()