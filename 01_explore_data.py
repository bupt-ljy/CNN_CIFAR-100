"""
================================================================================
第1步：认识 CIFAR-10 数据集 + 理解张量（Tensor）
================================================================================

本文件的学习目标：
  1. 知道 CIFAR-10 是什么，里面有什么数据
  2. 理解 PyTorch 中最核心的数据结构 —— Tensor（张量）
  3. 学会用 torchvision 加载数据集
  4. 理解"归一化"是什么，为什么重要
  5. 学会用 matplotlib 把图片数据可视化

建议学习方式：
  把代码一段一段复制到 VSCode 的 Python 交互窗口运行，
  或者在终端执行：python 01_explore_data.py
"""

import torch                     # PyTorch 核心库
import torchvision               # PyTorch 的计算机视觉工具箱
import torchvision.transforms as transforms  # 数据预处理/增强
import matplotlib.pyplot as plt  # 画图用的
import numpy as np               # 数值计算库

# ============================================================================
# 一、什么是 CIFAR-10？
# ============================================================================
# CIFAR-10 是一个包含 60000 张 32×32 彩色图片的数据集，分 10 个类别：
#
#   飞机(airplane)、汽车(automobile)、鸟(bird)、猫(cat)、鹿(deer)、
#   狗(dog)、青蛙(frog)、马(horse)、船(ship)、卡车(truck)
#
# 其中 50000 张用于训练（train），10000 张用于测试（test）
# 每张图片是 32×32 像素的 RGB 三通道彩色图 = 32*32*3 = 3072 个数值
# 每个像素值范围是 0-255（整数），代表颜色强度
#
# 选 CIFAR-10 而不是 MNIST（手写数字）的原因：
#   - MNIST 太简单了，28×28 灰度图，全连接网络都能做到 98%
#   - CIFAR-10 是彩色图，需要 CNN 才能真正做好，更有学习价值
#   - 32×32 的尺寸在你的 M5 上训练很快（几分钟一个 epoch）

print("=" * 60)
print("第1步：加载并认识 CIFAR-10 数据集")
print("=" * 60)

# ============================================================================
# 二、数据预处理：为什么需要 transforms？
# ============================================================================
# 深度学习有一条铁律：输入数据的"尺度"对训练影响极大。
#
# 原始图片的像素值是 0-255 的整数。如果直接喂给神经网络：
#   - 数值范围太大 → 梯度不稳定 → 训练困难
#   - 不同特征的数值范围不同 → 优化器要花更多时间调整
#
# 解决方法：归一化（Normalization）
#   - transforms.ToTensor()：把 0-255 变成 0.0-1.0 的浮点数
#   - transforms.Normalize(mean, std)：进一步变成均值0、标准差1的分布
#     (0.5, 0.5, 0.5) 和 (0.5, 0.5, 0.5) 意味着：
#     对每个通道：(x - 0.5) / 0.5 = 把 [0,1] 映射到 [-1, 1]
#
# 这样做的好处：
#   1. 模型收敛更快
#   2. 梯度更新更稳定
#   3. 不同特征在相同尺度上，优化器更容易找到最优解
#   4. 这是 ImageNet 等所有标准图像任务的通用做法

transform = transforms.Compose([  # Compose 把多个处理步骤串联起来
    transforms.ToTensor(),        # 步骤1: PIL图像(0-255整数) → Tensor(0.0-1.0浮点数)
    transforms.Normalize(         # 步骤2: 标准化到 [-1, 1]
        mean=(0.5, 0.5, 0.5),    # 三个通道(R,G,B)各自的均值
        std=(0.5, 0.5, 0.5)      # 三个通道各自的标准差
    )
])

# ============================================================================
# 三、下载并加载数据集
# ============================================================================
# torchvision.datasets 里内置了 CIFAR-10，一行代码就能下载
#
# 参数说明：
#   root='./data'  → 数据存到当前目录下的 data 文件夹
#   train=True     → True=训练集(50000张), False=测试集(10000张)
#   download=True  → 如果本地没有，自动从网上下载（大约 170MB）
#   transform=...  → 读取每张图片时，自动应用上面的预处理

print("\n正在加载 CIFAR-10 数据集...（首次运行会下载 ~170MB）")

train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"✅ 训练集大小: {len(train_dataset)} 张图片")
print(f"✅ 测试集大小: {len(test_dataset)} 张图片")

# ============================================================================
# 四、理解 Tensor（张量）—— PyTorch 的"万能数据容器"
# ============================================================================
# 这是深度学习最重要的概念之一：
#
# Tensor 就是一个多维数组（可以理解为 numpy array 的 GPU 加强版）
#
#   标量 (Scalar):   torch.tensor(3.0)         → shape: []     (0维)
#   向量 (Vector):   torch.tensor([1, 2, 3])   → shape: [3]    (1维)
#   矩阵 (Matrix):   torch.randn(3, 4)         → shape: [3,4]  (2维)
#   高维张量:        torch.randn(2, 3, 32, 32) → shape: [2,3,32,32] (4维)
#
# 图像数据的 shape 惯例：[批次大小, 通道数, 高度, 宽度]
#   - 批次大小 (Batch)：一次喂多少张图给模型
#   - 通道数 (Channel)：RGB=3，灰度=1
#   - 高度 (Height)：像素行数
#   - 宽度 (Width)：像素列数
#
# Tensor 和 numpy array 的区别：
#   - Tensor 可以在 GPU（或 Apple MPS）上运算 → 大幅加速
#   - Tensor 自带 autograd（自动求导）→ 反向传播的根基
#   - Tensor 和 numpy 可以互相转换：tensor.numpy() / torch.from_numpy()

print("\n" + "=" * 60)
print("五、取一张图片，理解数据结构")
print("=" * 60)

# 取训练集第一张图片
image, label = train_dataset[0]  # image是Tensor, label是整数(0~9)

print(f"\n图片的 Tensor shape: {image.shape}")
print(f"  → 解读: [通道数={image.shape[0]}, 高度={image.shape[1]}, 宽度={image.shape[2]}]")
print(f"  这是一张 {image.shape[1]}×{image.shape[2]} 的 RGB 彩色图")

print(f"\n标签 (label): {label}")
print(f"  → 类别名称: {train_dataset.classes[label]}")

# CIFAR-10 的 10 个类别
print(f"\n全部类别: {train_dataset.classes}")

# 看看像素值的范围
print(f"\n像素值统计:")
print(f"  最小值: {image.min():.3f}")   # 归一化后应该在 -1 附近
print(f"  最大值: {image.max():.3f}")   # 归一化后应该在 1 附近
print(f"  均值:   {image.mean():.3f}")   # 归一化后应该在 0 附近
print(f"  标准差: {image.std():.3f}")    # 归一化后应该在 1 附近

# ============================================================================
# 六、可视化：直观感受数据
# ============================================================================
# 训练神经网络前，一定要先"看"数据——这是所有人的必修课
# 因为如果数据有问题（标签错、归一化错），后面所有工作都白费

print("\n正在生成数据可视化...")

# 创建 4×8 = 32 张图片的画布
fig, axes = plt.subplots(4, 8, figsize=(14, 8))
axes = axes.flatten()

for i in range(32):
    img, label = train_dataset[i]

    # 反归一化：把 [-1,1] 的图片还原到 [0,1] 以便显示
    # 公式：img = img * std + mean，即 img * 0.5 + 0.5
    img = img * 0.5 + 0.5

    # Tensor → numpy，并调整维度顺序（从 [C,H,W] 变成 [H,W,C]）
    # 因为 matplotlib 需要 [高, 宽, 通道] 格式
    img_np = img.numpy().transpose(1, 2, 0)

    # 确保像素值在 [0,1] 范围内（防止显示异常）
    img_np = np.clip(img_np, 0, 1)

    axes[i].imshow(img_np)
    axes[i].set_title(train_dataset.classes[label], fontsize=9)
    axes[i].axis('off')

plt.suptitle("CIFAR-10 训练集样本（32张）", fontsize=16)
plt.tight_layout()
plt.savefig('cifar10_samples.png', dpi=100, bbox_inches='tight')
print("✅ 可视化图片已保存为 cifar10_samples.png")

# ============================================================================
# 七、DataLoader：批处理 + 打乱 + 多线程加载
# ============================================================================
# 训练时不会一张一张地处理图片，而是一次处理一个"批次"（batch）
# DataLoader 帮我们自动做三件事：
#   1. 打乱数据 (shuffle)：每个 epoch 随机重排 → 防止模型记住顺序
#   2. 按批次取数据 (batch_size)：一次取 N 张 → GPU 并行处理
#   3. 多线程加载 (num_workers)：后台读数据，不阻塞训练

# batch_size 的选择：
#   - 太小（如 8）：训练慢，梯度估计不准
#   - 太大（如 512）：内存不够，泛化能力可能下降
#   - 常用值：32/64/128（你的 16GB 内存用 64 或 128 都绰绰有余）

batch_size = 64  # 每次取 64 张图片

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,   # 每批 64 张
    shuffle=True,            # 每个 epoch 打乱顺序
    num_workers=2            # 用 2 个 CPU 线程加载数据
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,           # 测试集不需要打乱
    num_workers=2
)

# 看看一个 batch 长什么样
images, labels = next(iter(train_loader))  # iter()创建迭代器，next()取下一批

print(f"\n" + "=" * 60)
print("八、DataLoader 的一个批次")
print("=" * 60)
print(f"一批图片的 shape: {images.shape}")
print(f"  → [批次大小={images.shape[0]}, 通道={images.shape[1]}, 高={images.shape[2]}, 宽={images.shape[3]}]")
print(f"一批标签的 shape: {labels.shape}")
print(f"  → [批次大小={labels.shape[0]}]")
print(f"\n这一批的前10个标签: {labels[:10].tolist()}")
print(f"对应的类别名: {[train_dataset.classes[l] for l in labels[:10].tolist()]}")

num_batches = len(train_loader)
print(f"\n训练集总批次数: {num_batches}")
print(f"  → {len(train_dataset)} 张图片 ÷ {batch_size} 张/批 ≈ {num_batches} 批")
print(f"  → 每训练一个 epoch，模型要处理这 {num_batches} 批数据")

print(f"\n" + "=" * 60)
print("🎉 第1步完成！你已经理解了：")
print("   ✅ CIFAR-10 数据集的结构")
print("   ✅ Tensor（张量）是什么")
print("   ✅ 归一化的作用和原理")
print("   ✅ DataLoader 的批处理机制")
print("   ✅ 如何可视化图像数据")
print("=" * 60)
print("\n下一步：运行 02_cnn_model.py 学习如何搭建 CNN 网络")
