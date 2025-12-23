# 大模型微调LoRA篇

## 一、LoRA原理

​	LoRA的原理其实比较简单，就是任意一个矩阵$W$，都可以对他进行SVD分解$A = U \Sigma V^T$ （具体过程可以参考我的另一篇笔记）。

$U$和$V^T$带是正定矩阵,它们代表着对向量进行旋转。

$\Sigma$（拉伸）：沿着坐标轴方向进行缩放（拉伸或压缩）。

- ​    这里面的数值（奇异值 \sigma）代表了拉伸的倍数。
- ​    如果 \sigma 很大，说明在这个方向上拉伸得很长（重要）。
- ​    如果 \sigma 很小或为 0，说明在这个方向上被压扁了（不重要）

SVD会把奇异值 $\sigma$ 从大到小排列，大奇异值代表数据中的主要特征、信号、规律，小奇异值代表数据中的次要特征、噪音、细节。

 **最佳近似**：如果你只保留前 $k$ 个最大的奇异值，重构回去的矩阵，是原矩阵在低维空间下的**最佳近似**（Best Low-Rank Approximation）

------

LoRA 的论文开头就提出了一个核心假设：**“低内在维度（Low Intrinsic Dimension）”**。

- **问题**：全量微调时，权重矩阵的变化量 $ΔW$ 很大（维度比如 `4096×4096`）。

- **SVD 的视角**：
  如果我们把全量微调后的“理想变化量” $ΔW$ 拿来进行 SVD 分解：
  我们会发现，$Σ$（奇异值对角矩阵）中，**只有前几个奇异值非常大，后面绝大多数奇异值都接近于 0**。
  这意味着：虽然 $ΔW$ 看起来很大，但它真正有用的信息（能量）只集中在极少数几个方向上。

- **LoRA 的做法**：
  既然 SVD 告诉我们 $ΔW$ 本质上是低秩的，那我们何必训练整个大矩阵呢？我直接定义两个小矩阵 $A$ 和 $B$（秩为 r），强制让 $ΔW=BA$。
  **这本质上就是让神经网络通过梯度下降，自己去“学习”出那个最佳的 SVD 低秩近似。**

  

那么就有如下公式，很多讲解都认为AB是对$W_0$的近似，这实际上是不正确的：
$$
output = (W_0 + \Delta w)x = (W_0 + BA)x
$$
![image-20251222114941154](assets/LoRA流程.png)

## 二、秩的选择策略

### 1.权衡表示能力与效率

- **参数：**$\Delta w = BA$，引入的可训练参数量$r·(d+k) $，由于$r$远小于d和k，这个需要微调的参数是远小于$d·k$,虽然存储$\Delta w$会带来显存占用的增加，但是这个增加以及对应梯度占用的显存（等于$\Delta w$的大小），是远小于$d·k$
- **近似能力：**秩$r$决定了BA表征能力的上线，更大的$r$允许BA近似更加复杂的$\Delta W$，但不是越高越高，越高容易过拟合。这与SCD分解思路相关。

### 2.实用方法

1. **经验评估：** 最常见的方法是尝试一系列 **r**值，并在独立的验证集上评估表现。研究和实践中常试用的典型值通常包括 2 的幂次方。最优值很大程度上取决于具体的模型、数据集和任务。
2. **计算预算:**也就是GPU内存。
3. **性能饱和:**当r增加一定程度就会导致过拟合
4. **结合α考虑**：α控制着整体的更新幅度，而r决定结构能力，二者相互影响。（在下小节，可以详细看看）



## 三、缩放因子$\frac{α}{r}$

​	首先为什么要除以$r$？实际相当于一个**归一化的操作**。

​	在计算之前A和B矩阵都要进行**初始化**，不能都为0，不然就会导致梯度都为0无法更新。所以通常对A采取高斯分布初始化。那么r越大，$\Delta W$中的一项则是通过**r对**数相乘得来的，在相同的初始化条件下（A高斯初始化），**r 越大，矩阵乘积 BA 的输出值的数值幅度（Magnitude）通常会越大**（或者说梯度的范数会变大）。这时候就要调整学习率，每调整一次r，就要调整学习率，这个很麻烦。于是人为除以$r$，可以基本保持数值稳定。

​	那么$\alpha$是做什么的呢？相当于一个**整体的缩放**，在前者归一化后，乘以$\alpha$就相当于梯度扩大了α倍，实际上也就是学习率扩大了。实际调参过程。

**一句话：$\frac{1}{r}$ 是为了让你换 $r$ 的时候不用换学习率；而 $\alpha$ 是为了让你在不改 $r$ 的时候也能调学习率。**

​	这其实也表明：α涉及到平衡$W_0$中的通用训练知识和在$\Delta W$中学到的特定的调整的贡献。

1. **设定 α=r**：这是一种常用的启发式方法将 **α=r** 设定会有效抵消，从而得到更简单的更新形式 $h = W_0x+BAx$。这表示调整r，整体调整的实际幅度可能会随之隐式变化。
2. **将 α 设定为固定值**：实践者通常选择一个固定的 α*值（例如16、32、64），而不考虑 **r**。当使用$\frac{α}{r}$缩放时，这会将 α 视为期望调整强度的更直接衡量，相对于秩 r 提供的容量进行缩放。
3. **将 α 视为独立的超参数**：就像学习率或秩 r 一样，α 可以使用网格搜索、随机搜索或更复杂的贝叶斯优化等方法进行系统地调整。实验可能表明，当 α*α* 与 r*r* 有很大不同时，例如 α=2r 或 α=r/2，模型表现最佳。



## 四、手搓LoRA的线性层

​	在这之前，手推一下公式，因为很多时候我们在讲解理论的时候是没有考虑转置的。

标准的线性层：$y=xW_0^{T}+b,W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 

修改后的线性层，冻结原始权重，进行：$y = x W_0^T + x (BA)^T\frac{\alpha}{r} + b=y = x W_0^T + x A^T B^T\frac{\alpha}{r} + b$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Linear(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.r = r
        self.lora_alpha = lora_alpha
        # lora 参数
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # 冻结主权重，只训练 LoRA 参数
            self.weight.requires_grad = False
        self.reset_parameters()

        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # 初始化 LoRA 参数
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
      
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # LoRA 权重从主权重里剥离出来，保证训练时只更新 A 和 B，不更新主权重
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
                self.merged = False # 表示当前模型融合过
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):

        if self.r > 0 and not self.merged:
            result = F.linear(x, self.weight, bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, self.weight, bias=self.bias)

if __name__ == '__main__':
    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)
    
    # 定义参数
    in_features = 10
    out_features = 5
    r = 4
    lora_alpha = 8
    
    print("=== 开始测试 Linear LoRA 层 ===")
    
    # 1. 测试初始化
    layer = Linear(in_features, out_features, r=r, lora_alpha=lora_alpha)
    print(f"初始化检查: lora_A shape={layer.lora_A.shape}, lora_B shape={layer.lora_B.shape}")
    assert layer.lora_A.shape == (r, in_features)
    assert layer.lora_B.shape == (out_features, r)
    assert layer.weight.requires_grad == False, "主权重应该被冻结"
    assert layer.lora_A.requires_grad == True, "LoRA A 应该是可训练的"
    print("初始化测试通过 ✅")

    # 2. 测试前向传播 (初始状态)
    # 由于 lora_B 初始化为 0，初始输出应该与普通 Linear 层一致
    x = torch.randn(2, in_features)
    output = layer(x)
    expected = F.linear(x, layer.weight, layer.bias)
    assert torch.allclose(output, expected, atol=1e-6), "初始输出应该与普通 Linear 层一致"
    print("初始前向传播测试通过 ✅")

    # 3. 测试 LoRA 效果
    # 手动设置权重以验证计算逻辑
    nn.init.constant_(layer.lora_A, 1.0)
    nn.init.constant_(layer.lora_B, 1.0)
    # scaling = 8/4 = 2
    # LoRA term = (x @ A.T @ B.T) * scaling
    # 如果 x 全为 1 (1, 10)
    # x @ A.T (1, 4) -> 每个元素是 10 (1*1 * 10)
    # (x @ A.T) @ B.T (1, 5) -> 每个元素是 10 * 4 = 40
    # result * scaling (2) -> 80
    
    x_ones = torch.ones(1, in_features)
    output_ones = layer(x_ones)
    base_output = F.linear(x_ones, layer.weight, layer.bias)
    diff = output_ones - base_output
    
    expected_diff = 80.0
    assert torch.allclose(diff, torch.tensor(expected_diff), atol=1e-5), f"LoRA 增量计算错误, 期望 {expected_diff}, 实际 {diff.mean().item()}"
    print("LoRA 增量计算测试通过 ✅")

    # 4. 测试权重合并 (eval 模式)
    print("测试权重合并逻辑...")
    original_weight = layer.weight.data.clone()
    
    # 切换到 eval 模式 -> 应该触发合并
    layer.eval()
    assert layer.merged == True, "Eval 模式下 merged 标记应为 True"
    assert not torch.allclose(layer.weight.data, original_weight), "Eval 模式下主权重应该发生变化 (合并了 LoRA)"
    
    # 再次前向传播，结果应该保持一致
    output_eval = layer(x_ones)
    assert torch.allclose(output_eval, output_ones, atol=1e-6), "合并权重后的输出应该与未合并时一致"
    
    # 切换回 train 模式 -> 应该触发解绑
    layer.train()
    assert layer.merged == False, "Train 模式下 merged 标记应为 False"
    assert torch.allclose(layer.weight.data, original_weight, atol=1e-5), "Train 模式下主权重应该恢复原状"
    print("权重合并/解绑测试通过 ✅")
    
    print("=== 所有测试通过 ===")

        
```

