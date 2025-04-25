---
layout: post
title: "Stable Diffusion 模型结构"
date: 2025-04-25 09:00:00 +0800 
categories: 研究生涯
tag: diffusion
---



Stable Diffusion 是一种 **文本引导的潜空间扩散模型（Latent Diffusion Model, LDM）**

<!-- more -->

## Stable Diffusion 模型结构



Stable Diffusion 是一种 **文本引导的潜空间扩散模型（Latent Diffusion Model, LDM）**，其核心组件主要包括以下几个部分：

------

### 1. **Encoder / Decoder（VAE）**

- 将原始图像映射到 **潜空间（latent space）**，大大减少了扩散过程的计算开销。
- 编码器：图像 → 潜在表示 `z`
- 解码器：潜在表示 `z` → 图像

------

### 2. **UNet（扩散模型的核心）**

- 结构：U-Net 编码器-解码器框架，支持跳跃连接。
- 输入：
  - 噪声图像（latent `z_t`）
  - 时间步 `t`
  - 文本条件向量（来自 CLIP 的文本嵌入）

#### 子模块包括：

- **ResNet 块**（带时间步 & 条件嵌入）
- **Cross Attention**：用于融合文本条件（CLIP 编码后的文本嵌入）
- **Self-Attention**

> ✅ 支持 Classifier-Free Guidance 以控制文本条件对图像生成的影响力。

- $\mathbf{x}_t$：当前扩散图像（latent）
- $\mathbf{c}$：文本条件（CLIP 编码向量）
- $\epsilon_\theta(\mathbf{x}_t, t, \mathbf{c})$：UNet 预测的噪声（有条件）
- $\epsilon_\theta(\mathbf{x}_t, t, \emptyset)$：UNet 预测的噪声（无条件）
- $\mathbf{c} = \emptyset$ 表示没有输入条件，即 unconditional 分支。
- $s$：引导系数（guidance scale）

则 **最终使用的预测为：**
$$
\epsilon_{\text{guided}} = \epsilon_\theta(\mathbf{x}_t, t, \emptyset) + s \cdot \left( \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}) - \epsilon_\theta(\mathbf{x}_t, t, \emptyset) \right)
$$

> 在实际中，`s` 通常设置为 **7.5 ~ 12.0**，视具体应用而定。

------

### 3. **Text Encoder（条件部分）**

- 通常使用 **CLIP 的文本编码器**（ `CLIPTextModel`）将提示词（prompt）转化为向量表示。
- 输出：嵌入向量用于输入 UNet 中的 cross-attention 模块。

------

### 4. **Diffusion Process（扩散与反扩散）**

- 前向扩散：逐步向潜在表示 `z` 添加噪声。
- 反向过程：
  - UNet 预测噪声
  - 使用调度器（ DDIM、DDPM）一步步去噪。

------

## 🗂 模型结构流程图

```
Text Prompt --→ CLIP Encoder --→ Text Embedding
                                     ↓
          +--------------------+     ↓
Image --> |   Encoder (VAE)    |     ↓
          |    ↓ latent z      |     ↓
          +--------------------+     ↓
                                     ↓
            +-----------------------------------------+
 z_t (latent) → UNet(noise predictor) ← t, Text Embed |
            +-----------------------------------------+
                                     ↓
          +--------------------+
          |   Decoder (VAE)    | → Output Image
          +--------------------+
```

------

## 🧩 总结

| 模块         | 功能                                |
| ------------ | ----------------------------------- |
| VAE 编码器   | 图像 → 潜空间（压缩）               |
| UNet         | 在潜空间中执行去噪过程              |
| Text Encoder | 提供条件引导（如 CLIP）             |
| Decoder      | 潜在图像重构回原始图像              |
| Diffusion    | 添加/去除噪声以生成图像（采样过程） |



## 实现

[XinbeiCai/pytorch-stable-diffusion](https://github.com/XinbeiCai/pytorch-stable-diffusion)