---
layout: post
title: "Stable Diffusion 微调方法"
date: 2025-04-25 09:00:00 +0800 
categories: 研究生涯
tag: diffusionb
---



>本文介绍了不同的stable diffusion的微调方法



<!-- more -->

# Stable Diffusion 微调方法



| 方法                  | 怎么做（原理/操作）                                      | 训练资源需求 | 效果表现         | 适合场景                     |
| --------------------- | -------------------------------------------------------- | ------------ | ---------------- | ---------------------------- |
| **Full Fine-tuning**  | 整个模型（U-Net、VAE、CLIP）继续训练                     | 很高         | 超强（全面改造） | 训练大数据集、做全新模型     |
| **Textual Inversion** | 只训练一个新的词向量（嵌入），不改模型参数               | 极低         | 中（单一概念）   | 学特定风格、物体、小角色     |
| **DreamBooth**        | 少量图片 + 关键词绑定，微调 U-Net（有时加 Text Encoder） | 中等         | 很强（特定概念） | 个人定制（人物、宠物、角色） |
| **LoRA**              | 冻结原模型，仅训练插入的小矩阵（低秩适配层）             | 低           | 强（灵活细致）   | 快速训练风格、角色、服装设计 |
| **ControlNet**        | 微调带条件输入的辅助网络（结构分支，不动原模型主干）     | 中等         | 很强（结构控制） | 姿态、轮廓、边缘图指导生成   |



## 1. Textual Inversion

**Textual Inversion** 是一种著名的**扩散模型（如 Stable Diffusion）微调技术**，由 Google Research 于 2022 年提出。
 它的目标是：

> **用极少量的数据（通常几张图）让模型学习一个新概念（如新的人、物体、艺术风格）**。

------

**核心思路**

Textual Inversion 的本质是：

> **定义一个新的、可学习的词向量（embedding），用它引导扩散模型生成特定概念的图像。**

------

**训练流程**

1. **引入新 token**
   - 自定义一个新词，如 `"<my_cat>"`，代表你家的猫。
2. **只训练词向量**
   - 冻结模型主体（UNet、VAE、CLIP encoder），**仅优化新 token 的词向量**。
3. **优化目标**
   - 通过 prompt（如 `"A photo of <my_cat> sitting on a chair"`）生成图像，
   - 用重建损失（如 MSE Loss）让生成结果接近真实照片。
4. **训练完成后**
   - 在任意 prompt 中使用 `"<my_cat>"`，模型即可识别并正确生成！

------

**优势总结**

- **高效**：只训练一个小向量，计算开销极小。
- **稳定**：不破坏原模型的通用能力。
- **灵活**：让扩散模型学习稀有、个性化元素。





**例如：**prompt：a <cat-toy> is on the grass



没有用<cat-toy>微调：

![image-20250426235542228](https://s2.loli.net/2025/04/27/RjOM2frV8cxTaAY.png)



用了<cat-toy>微调

![image-20250426235138092](https://s2.loli.net/2025/04/27/27VO5oGKzXfJ3Z6.png)



## 2. LoRA 微调 Stable Diffusion 

### 1. 什么是 LoRA？

- **Low-Rank Adaptation (LoRA)** 是一种轻量化微调方法。
- 它通过**冻结大模型权重**，只引入小的可训练模块（低秩矩阵）来学习新任务。
- 这样可以**节省显存**、**加速训练**，且**保持大模型原本的能力**。

------

### 2. 为什么 LoRA 有效？

Stable Diffusion 中，主要微调的是 **UNet** 和有时的 **Text Encoder**，它们的权重矩阵通常很大：

$W \in \mathbb{R}^{d_{out} \times d_{in}}$

直接微调 $W$ 代价高。而 LoRA 的做法是：

**冻结 $W$，新增可训练的低秩偏置项**：
$$
W' = W + \Delta W
$$
其中：

- $\Delta W = BA$

- $A \in \mathbb{R}^{r \times d_{in}}$
- $B \in \mathbb{R}^{d_{out} \times r}$
- $r \ll \min(d_{in}, d_{out})$

即：**只训练 $A$ 和 $B$，且 $r$ 很小（如4、8）**

------

### 3. LoRA 插入位置（Stable Diffusion）

通常在以下位置加 LoRA：

- **UNet 中的 Attention 层**（尤其是 Cross Attention）
- 有时在 **Self-Attention** 层也加
- Text Encoder 层（如 CLIP Encoder）也可以加

**Cross Attention 示例：**

原始Attention：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d}} \right)V
$$
其中：

- $Q = XW_Q$
- $K = YW_K$
- $V = YW_V$

LoRA应用在 $W_Q$, $W_K$, 或 $W_V$ 上，比如：
$$
W_Q' = W_Q + B_Q A_Q
$$
同理 $W_K$、$W_V$。

------

### 4. 超参数推荐

| 参数               | 建议值          |
| ------------------ | --------------- |
| rank ($r$)         | 4、8、16        |
| learning rate      | 1e-4 ~ 1e-5     |
| batch size         | 2~8（根据显存） |
| optimizer          | AdamW           |
| warmup steps       | 100~500         |
| clip gradient norm | 1.0             |

------

### 5. 训练流程概览

1. 加载预训练Stable Diffusion模型。
2. 在指定模块（如Attention）插入LoRA层。
3. 冻结原模型参数，只让LoRA参数可训练。
4. 按常规Diffusion训练流程训练（预测噪声）。
5. 保存LoRA权重（小文件）。

------

### 6. 推理时使用

推理阶段可以**加载LoRA权重**并**以一定比例融合**：

如果融合比例是 $\alpha$，那么：
$$
W' = W + \alpha \times (BA)
$$
一般 $\alpha = 0.5$ ~ $1.0$​，可以调整以改变LoRA效果的强度。



```python
# ==== LoRA模块 (只针对Attention里的qkv) ====
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, scale=1.0):
        super().__init__()
        self.linear = linear_layer  # 原本的Linear层
        self.rank = rank
        self.scale = scale

        self.lora_up = nn.Linear(linear_layer.in_features, rank, bias=False)
        self.lora_down = nn.Linear(rank, linear_layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_down.weight)

    def forward(self, x):
        return self.linear(x) + self.scale * self.lora_down(self.lora_up(x))


def apply_lora_to_unet(unet, rank=4, scale=1.0, device='cuda'):
    # 先把整个unet所有参数冻结
    for param in unet.parameters():
        param.requires_grad = False

    # 再在需要的地方加LoRA
    for name, module in unet.named_modules():
        if isinstance(module, SelfAttention):
            module.in_proj = LoRALinear(module.in_proj, rank=rank, scale=scale).to(device)
        elif isinstance(module, CrossAttention):
            module.q_proj = LoRALinear(module.q_proj, rank=rank, scale=scale).to(device)
            module.k_proj = LoRALinear(module.k_proj, rank=rank, scale=scale).to(device)
            module.v_proj = LoRALinear(module.v_proj, rank=rank, scale=scale).to(device)
```

------

可以看到，也能训练出特定的<toy_cat>，但是grass的元素没有很好地展示出来，可能是因为<cat_toy>特征过于明显

![lora_output](https://s2.loli.net/2025/04/27/RgPp5JHinYrqoF9.png)

### 总结

> LoRA 是通过小矩阵 $BA$ 替代大矩阵 $W$ 的微调方法，训练快、参数少，适合对 Stable Diffusion 进行定制和个性化优化。







## DreamBooth 微调

### 1. 什么是 DreamBooth？

DreamBooth 是 Google Research 提出的个性化生成技术，
 可以让扩散模型（如 Stable Diffusion）**学会特定的个体概念**，比如某个人、某条狗、某个物体。

------

### 2. DreamBooth 的基本思路

**目标**：
 给定少量特定个体图片（通常3-5张），让模型在生成时能准确重现它，且还能理解 prompt 中的上下文。

**怎么做**：

- 引入一个**新词**（比如 `sks dog`），指代特定个体。
- 微调扩散模型，使它在条件生成时，能把这个新词和特定外观联系起来。

------

### 3. DreamBooth 的训练流程

#### 3.1 新词引入

定义一个稀有词（比如 `sks`），加上类别词（如 `dog`），组成新prompt：

> ```
> "a photo of sks dog"
> ```

其中：

- `sks`：专门指代你的狗。
- `dog`：保持类别的一般特性。

#### 3.2 训练目标 (Loss)

在每一步，模型要学会：

- 在看到 `"a photo of sks dog"` 时，重建带特定特征的图片。
- 保持对其他"dog"类的一般理解，不只记忆这几张图。

主要优化的仍是标准的扩散损失（预测噪声）：
$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{x, \epsilon, t} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t, c) \right\|^2 \right]
$$
其中：

- $x$：原始个体图片
- $t$：时间步
- $\epsilon$：加的高斯噪声
- $c$：prompt编码 (`"a photo of sks dog"`)

### 3.3 Class Regularization（类别正则化）

为了避免模型只记住几张图片，DreamBooth引入了**类别正则化**：

- 额外采样一些**类别图像**（比如普通的"dog"图像）。
- 用 prompt `"a photo of a dog"` 来训练，保持类别分布。

对应的**类别正则化Loss**：
$$
\mathcal{L}_{\text{class}} = \mathbb{E}_{x_{\text{class}}, \epsilon, t} \left[ \left\| \epsilon - \epsilon_\theta(x_{\text{class},t}, t, c_{\text{class}}) \right\|^2 \right]
$$


最终总Loss：
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{simple}} + \lambda \mathcal{L}_{\text{class}}
$$

- $\lambda$ 是类别正则的权重（一般取 1）



### 冻结情况

| 模块                | 是否冻结 | 备注                    |
| :------------------ | :------- | :---------------------- |
| UNet                | ❌ 不冻结 | 主体微调对象            |
| Text Encoder (CLIP) | ✅ 冻结   | 可选小步微调（很少做）  |
| VAE (Autoencoder)   | ✅ 冻结   | 不动，负责图像编码/解码 |
| Scheduler           | 无需动   | 控制噪声添加/还原       |





## 4. ControlNet

### 1. 什么是 ControlNet？

**ControlNet** 是一种扩展 Stable Diffusion 的方法， 可以在生成图像时**增加额外的控制信号**，比如：

- 边缘图（Canny）
- 深度图（Depth）
- 姿态估计（Pose）
- 草图（Scribble）
- 分割图（Segmentation）
- …

✅ 这样可以让 Stable Diffusion 不再纯靠 prompt，生成更符合用户期望的内容！

------

### 2. ControlNet 的基本思路

在原本 Stable Diffusion 的 **UNet** 上， **复制一份结构**，用来输入控制条件（比如边缘图）。

训练时：

- 把控制输入（比如Canny图）经过一套卷积层，提取特征。
- 在每个 U-Net Block（尤其是中间部分）**加上这些特征作为引导**。

------

### 3. 训练原理（公式）

基本的扩散模型优化是：
$$
\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{x, \epsilon, t, c} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t, c) \right\|^2 \right]
$$

- $c$ 是prompt条件。

加上 ControlNet 之后，目标变成：
$$
\mathcal{L}_{\text{control}} = \mathbb{E}_{x, \epsilon, t, c, m} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t, c, m) \right\|^2 \right]
$$

- $m$ 是**控制图**（比如canny边缘图的编码特征）。即：让模型不仅根据 prompt，还根据 control 图来预测噪声。

------

### 4. ControlNet 网络结构

核心是：

- **原本的UNet主干：** 保持预训练权重（可以冻结，也可以微调）
- **Control分支：** 新加的，专门处理控制输入。

![img](https://github.com/lllyasviel/ControlNet/raw/main/github_page/he.png)

------

### 5. 训练时冻结吗？

- **可以选择**：
  - 完全冻结原UNet，只训练Control分支（收敛快）
  - 或者少量微调UNet（效果更好但慢）

大多数官方ControlNet模型：

- **UNet权重拷贝** + **Control分支单独训练**。

------

### 6. 推理（使用ControlNet）

推理阶段，需要同时提供：

- 正常的 prompt
- 加上 control 图（比如 Canny 边缘）

生成时，ControlNet 会自动根据控制图引导模型生成内容！

**推理示例：**

```python
output = model(
    prompt="a dog playing guitar",
    control_image=canny_edge_image,
)
```

------

### 总结：

> **ControlNet 是在 Stable Diffusion 基础上增加一个控制分支，使模型在生成时可以根据输入的图像特征引导输出，同时保持原有的自由文本能力。**

------

