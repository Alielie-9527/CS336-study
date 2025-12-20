# AWQ量化（激活感知权重量化）

之前的量化方法的认为：**权重的数值越大，权重越重要**

AWQ认为：**决定一个权重是否重要，不是权重本身，而是输入的大小**

因为下一层的输入是由这一层的输入和权重得来的，那么再大的权重，只要输入很小，最后的输出都很小。所以我们应该保护对应着大激活值的权重。

## 核心原理与公式解析

### 1.尺度不变性

对于一个线性层矩阵乘法有：
$$
Y=WX
$$
如果我们引入一个缩放系数**s**(标量或者向量):
$$
Y = W X = (W \cdot s) (X \cdot \frac{1}{s})
$$

### 2.为什么引入S能减小误差

​	量化过程中的误差主要来自于取整，在量化过程中，比如INT4量化，权重数值越大，那么相对量化误差越小。

量化函数写作$$Q(w)$$

我们的**目标是让量化后的重要权重位置输出误差最小**：
$$
\text{Error} = || Q(W \cdot s) \cdot (X / s) - WX ||
$$
让我们看单个元素的量化。假设我们使用 INT4（范围 -8 到 7），步长（Scale）为 $\Delta$。
$$
\Delta = \frac{max(|w|)}{2^{N-1}} (绝对值量化：INT4，N=4)
$$
量化过程其实是：
$$
% 导言区需要加入：\usepackage{amsmath}

\begin{align*}
    \text{量化：}Q(w) &= \operatorname{Round}\left( \frac{w}{\Delta} \right) \cdot \Delta \\
    \text{Quantization Error: } \mathcal{E}(Q(w)x) &= Q(w)x - wx \\
    &= \bigl( Q(w) - w \bigr) x \\
    &= \left[ \operatorname{Round}\left( \frac{w}{\Delta} \right) \cdot \Delta - w \right] x \\
    &=\Delta\cdot\text{RoundErr}\left( \frac{w}{\Delta} \right) \cdot x
\end{align*}
$$
记$\delta=\Delta\cdot\text{RoundErr}\left( \frac{w}{\Delta} \right)$右半部分取整函数，我们近似为均匀分布，在$[-0.5,0.5]$上的均匀分布，则有如下

*   **分布假设**：$e \sim U(-\frac{\Delta}{2}, \frac{\Delta}{2})$
*   **误差期望**（Mean）：$E[e] = 0$
*   **绝对误差期望**（Expected Magnitude）：$\mathbb{E}[|e|] = \frac{1}{\Delta} \int_{-\Delta/2}^{\Delta/2} |z| \,dz = \frac{\Delta}{4}$

**关键推导：**
考虑第$$i$$个权重，假设它的激活值 $X_i$ 很大。
这就意味着对应的权重 $W_i$ 很重要，一点点误差 $\delta$ 都会被 $X_i$ 放大成 $\delta \cdot X_i$。

**AWQ 的操作：**

1.  我们把这个重要的权重 $W_i$ **放大 $s$ 倍**（比如 $s=2$）。
2.  同时把输入 $X_i$ **缩小 $s$ 倍**。

**此时发生了什么？**

**公式说明：**
$$
% 导言区需要加入：\usepackage{amsmath}

\begin{align*}
    \text{Scaled Output: } \hat{y} &= Q(w \cdot s) \cdot \frac{x}{s} \\
    &= \left[ \operatorname{Round}\left( \frac{w \cdot s}{\Delta'} \right) \cdot \Delta' \right] \cdot \frac{x}{s} \\
    \text{AWQ Error: } \mathcal{E}_{\text{scaled}} &= \hat{y} - wx \\
    &= Q(w \cdot s) \cdot \frac{x}{s} - (w \cdot s) \cdot \frac{x}{s} \\
    &= \left[ Q(w \cdot s) - w \cdot s \right] \cdot \frac{x}{s} \\
    &= \left[ \operatorname{Round}\left( \frac{w \cdot s}{\Delta'} \right) \cdot \Delta' - w \cdot s \right] \cdot \frac{x}{s} \\
    &= \Delta' \cdot \operatorname{RoundErr}\left( \frac{w \cdot s}{\Delta'} \right) \cdot \frac{x}{s}
\end{align*}
$$
因为对$w$进行放缩后，$$w$$所在列的$max(W)$可能发生变化，所以引入了$\Delta^`$，但是$RoundErr(·)$分布是不变的，对比可知到，虽然 $\Delta'$ 变大了（导致前面的绝对误差变大），但后面的激活项 $\frac{x}{s}$ 变小了，总体来说还是变小的，**大部分情况下$\Delta$是不变的（统计得出）**。

**在一定范围内，随着** **的增大，误差比值越来越小，这是完全支持作者观点的。**

AWQ 的精髓在于寻找最优的 $s$，使得这两者的乘积（总重建误差）最小化。



### 3.S不是越大越好以及S如何确定

​	这个其实比较容易理解，当S非常大的时候，这时候$max(W)$会变大，导致 $\Delta$变得非常大，那么其余参数虽然没那么重要但是不代表完全不重要，它们经过量化函数$Q(w) = \operatorname{Round}\left( \frac{w}{\Delta} \right) \cdot \Delta$ 后大部分值都变得非常小，这里其实就相当于之前量化中出现异常值一样的道理。





------

### 4.S进行放缩的方向和W进行quant的方向

我将用**具体的数值矩阵**，配合**PyTorch 的存储视角**，演示 **Scale（竖着切）** 和 **Quant（横着切）**

---

### 一、 场景设定：坐标系与数据

#### 1. PyTorch 中的 Linear 层
假设我们定义了一个层：`nn.Linear(in_features=4, out_features=2)`。
*   **输入 $X$**：Batch=1，有 4 个特征。
*   **权重 $W$**：PyTorch 中存储的形状是 `[out_features, in_features]`，即 **$[2, 4]$**。

#### 2. 原始权重数据 ($W$)
为了演示效果，我们假设所有权重初始值都是 **0.1**（非常平庸，但也包含信息）。
$$
W = 
\begin{bmatrix} 
0.1 & 0.1 & 0.1 & 0.1 \\ 
0.1 & 0.1 & 0.1 & 0.1 
\end{bmatrix}
$$
*   **第 0 行**：负责计算第 1 个输出神经元。
*   **第 1 行**：负责计算第 2 个输出神经元。
*   **每一列**：对应一个输入特征通道（Channel 0 ~ 3）。

#### 3. 激活值情况 ($X$)
假设输入数据 $X = [100, 1, 1, 1]$。
*   **Feature 0 (Channel 0)**：数值是 **100**（超级显著，VIP）。
*   **Feature 1~3**：数值是 **1**（平民）。

**结论**：因为 Feature 0 的值很大，所以 **$W$ 的第 0 列** 非常重要（它的误差会被放大 100 倍）。

---

### 二、 动作一：Scale 的方向（竖着切，Column-wise）

AWQ 决定保护第 0 列。
策略：把 **第 0 列的权重放大**，同时把输入缩小（输入的变化在上一层处理，这里只看权重）。

假设我们计算出的缩放因子 $s = 100$（为了让效果夸张点）。

**操作**：只针对 **第 0 列 (Column 0)** 乘以 100。
$$
W_{scaled} = 
\begin{bmatrix} 
\mathbf{10.0} & 0.1 & 0.1 & 0.1 \\ 
\mathbf{10.0} & 0.1 & 0.1 & 0.1 
\end{bmatrix}
$$
*   **方向**：你看，这是一个**垂直**的操作。它**跨越了所有行（输出神经元）**，修改了整整一列。

---

### 三、 动作二：Quant 的方向（横着切，Row-wise）

现在进入量化阶段。
在 INT4 量化中，为了硬件计算点积方便，我们通常是**按行（Row）**或者**按行内的组（Group）**来量化的。
简单起见，假设 **Group Size = 4**，也就是 **一整行共用一套量化参数**。

现在我们来看看 **第 0 行** 发生了什么

#### 1. 确定量化标尺 (Grid)
数据是：`Row_0 = [10.0, 0.1, 0.1, 0.1]`。

*   **最大值 (AbsMax)**：$10.0$。
*   **INT4 范围**：$-8$ 到 $7$（共 16 个刻度）。
*   **步长 (Scale)**：$\frac{10.0}{7} \approx \mathbf{1.42}$。
    *   意思是：INT4 的每一个刻度（1, 2, 3...）代表真实的 1.42。

#### 2. 开始量化 (Round)

*   **VIP 权重 (10.0)**：
    $10.0 / 1.42 \approx 7.04 \to \text{Round} \to \mathbf{7}$。
    *   反量化回去：$7 \times 1.42 = 9.94$。
    *   误差：$0.06$。相对误差很小。**VIP 被保护得很好！**

*   **平民权重 (0.1)**：
    $0.1 / 1.42 \approx 0.07 \to \text{Round} \to \mathbf{0}$。
    *   反量化回去：$0 \times 1.42 = \mathbf{0}$。
    *   **误差**：$0.1 - 0 = 0.1$。
    *   **相对误差**：$100\%$！信息完全丢失了！

**这也是“S 并非越大越好”的数学证明：**
因为 Scale 动作（竖向）把 `10.0` 塞进了 `Row_0` 这个横向的宿舍里，导致整个宿舍的“天花板（Max）”被强行撑高了。为了容纳这个巨人，刻度变得非常稀疏（步长 1.42），导致原来的小个子 `0.1` 连第一个刻度都够不着，直接被抹零了。

---

### 四、 为什么要结合 W 的转置来看？

 `linear` 中 $W$ 的转置问题，其实是矩阵乘法的视角问题。

公式：$Y = X W^T$

*   $X$: $[B, C_{in}]$
*   $W$: $[C_{out}, C_{in}]$
*   $W^T$: $[C_{in}, C_{out}]$

当我们做矩阵乘法 $X \times W^T$ 时：
1.  我们是拿 $X$ 的 **一行** (1, $C_{in}$) 去乘 $W^T$ 的 **一列** (相当于 $W$ 的**一行**)。
2.  这个点积操作，就是把 $C_{in}$ 个元素对应相乘再相加。

**冲突的根源就在这里**：
*   **相乘相加**：意味着这 $C_{in}$ 个元素（$W$ 的一行）必须在一起运算。所以量化参数（Scale/Zero-point）通常是绑定给这 $C_{in}$ 个元素的（横向绑定）。
*   **输入特征**：$X$ 的第 $j$ 个特征对应的是 $W$ 的第 $j$ 列。所以激活值的强弱，影响的是 $W$ 的纵向一列（纵向影响）。



**同样，整理一下S计算的维度**：

---

### 一、 场景设定：数据与维度

#### 1. 权重矩阵 $W$
`nn.Linear(in_features=4, out_features=2)`。
为了演示清晰，我们设计一个极端的权重矩阵：
*   **第 0 列**：数值大（比如 10），且对应的激活值也大。
*   **第 1~3 列**：数值小（1），对应的激活值也小。

$$
W = 
\begin{bmatrix} 
10 & 1 & 1 & 1 \\ 
5 & 2 & 2 & 2 
\end{bmatrix}
$$
**维度**：$[2, 4]$ (Output=2, Input=4)。

#### 2. 校准数据集 $X_{calib}$
我们需要喂给模型一小批数据来观察“谁是 VIP”。假设 Batch Size = 3。

$$
X = 
\begin{bmatrix} 
100 & 0 & 0 & 0 \\ 
80 & 1 & 1 & 1 \\ 
60 & 2 & 2 & 2 
\end{bmatrix}
$$
**维度**：$[3, 4]$ (Batch=3, Input=4)。

---

### 二、 第一阶段：收集统计信息 (Statistics)

AWQ 不训练参数，而是基于统计。我们需要统计出 $W$ 和 $X$ 的**通道（列）强度**。

#### 1. 计算激活值的幅度 $s_X$
我们要看每一列（每个特征）在所有样本中的最大绝对值是多少。
*   **操作**：沿着 Batch 维度（第 0 维）取 Max。
*   **计算**：
    *   Col 0: $\max(|100|, |80|, |60|) = 100$
    *   Col 1: $\max(0, 1, 2) = 2$
    *   ...
*   **结果 $s_X$**：
    $$ [100, \quad 2, \quad 2, \quad 2] $$
    **维度**：$[1, 4]$ (对应 4 个输入通道)。

#### 2. 计算权重的幅度 $s_W$
我们要看每一列权重本身的大小（因为这一列是一起进行放缩的，结合之前讲到的S的方向）。
*   **操作**：沿着输出维度（第 0 维，即每一行）取 Max（或者平均值，论文通常用 Max）。
*   **计算**：
    *   Col 0: $\max(|10|, |5|) = 10$
    *   Col 1: $\max(|1|, |2|) = 2$
    *   ...
*   **结果 $s_W$**：
    $$ [10, \quad 2, \quad 2, \quad 2] $$
    **维度**：$[1, 4]$。

---

### 三、 第二阶段：网格搜索最优 $s$ (Grid Search)（最后取值，就是希望Loss最小）

我们不知道缩放多少倍最好，所以我们引入超参数 $\alpha$。
公式：$s = s_X^\alpha \cdot s_W^{1-\alpha}$。
搜索空间：$\alpha \in \{0, 0.1, \dots, 1.0\}$。

我们来模拟**两次尝试**，看看数据发生了什么变化。

#### 尝试 A：$\alpha = 1.0$ (完全依赖激活值，忽略权重本身)
$$ s = s_X^{1.0} \cdot s_W^{0.0} = s_X = [100, 2, 2, 2] $$
**维度**：$[1, 4]$。

**1. 应用缩放 (Apply Scale)**
我们将权重 $W$ 的每一列乘以对应的 $s$：
*   第 0 列：$W[:, 0] \times 100$
*   第 1 列：$W[:, 1] \times 2$

$$
W_{scaled} = 
\begin{bmatrix} 
10 \times 100 & 1 \times 2 & 1 \times 2 & 1 \times 2 \\ 
5 \times 100 & 2 \times 2 & 2 \times 2 & 2 \times 2 
\end{bmatrix} 
= 
\begin{bmatrix} 
\mathbf{1000} & 2 & 2 & 2 \\ 
\mathbf{500} & 4 & 4 & 4 
\end{bmatrix}
$$

**2. 模拟量化 (Simulate Quantization)**
现在我们对 $W_{scaled}$ 进行 INT4 量化（按行分组，一行一组）。

*   **看第 0 行**：`[1000, 2, 2, 2]`
    *   最大值 Max = 1000。
    *   INT4 步长 = $1000 / 7 \approx 142$。
    *   **悲剧发生**：那个 `2` (原权重1) 量化后变成 $2/142 \approx 0$。
    *   **误差**：小权重全死光了。

*   **计算重构误差**：
    我们需要计算量化后的输出 $Y'$ 与原输出 $Y$ 的差距。由于小权重误差巨大，且 $s$ 很大，**导致最终 Loss 很高**。

---

#### 尝试 B：$\alpha = 0.5$ (折中方案，开根号)
$$ s = s_X^{0.5} \cdot s_W^{0.5} = \sqrt{s_X} \cdot \sqrt{s_W} $$

**计算 $s$ 向量**：
*   Col 0: $\sqrt{100} \times \sqrt{10} = 10 \times 3.16 = 31.6$
*   Col 1: $\sqrt{2} \times \sqrt{2} = 2$
*   **结果 $s$**：$[31.6, \quad 2, \quad 2, \quad 2]$。

**1. 应用缩放**
$$
W_{scaled} = 
\begin{bmatrix} 
10 \times 31.6 & 1 \times 2 & \dots \\ 
5 \times 31.6 & 2 \times 2 & \dots 
\end{bmatrix} 
= 
\begin{bmatrix} 
\mathbf{316} & 2 & 2 & 2 \\ 
\mathbf{158} & 4 & 4 & 4 
\end{bmatrix}
$$

**2. 模拟量化**
*   **看第 0 行**：`[316, 2, 2, 2]`
    *   最大值 Max = 316。
    *   INT4 步长 = $316 / 7 \approx 45$。
    *   **小权重的情况**：$2 / 45 \approx 0.04 \to 0$。
    *   虽然小权重还是被抹零了，但相比于尝试 A 的步长 142，现在的步长 45 更加细腻了一些（如果是 INT8 就能保住了）。更重要的是，**VIP 权重（第 0 列）并没有像尝试 A 那样被过度放大**，这通常能获得更好的整体 Loss。

**3. 计算 Loss**
系统发现这一组 $\alpha=0.5$ 算出来的 $W_{scaled}$，在配合 $X_{scaled} = X/s$ 进行计算时，输出结果 $Y'$ 和原始 FP16 的 $Y$ 最接近。

**决策**：选定 $\alpha=0.5$，对应的 $s=[31.6, 2, 2, 2]$ 就是我们要找的**最优缩放向量**。

---

### 四、 最终流程总结（数据流向）

1.  **输入**：
    *   $W$: $[C_{out}, C_{in}]$
    *   $X_{calib}$: $[Batch, C_{in}]$

2.  **统计 (Reduce)**：
    *   $X \to \text{Max}(dim=0) \to s_X: [1, C_{in}]$
    *   $W \to \text{Max}(dim=0) \to s_W: [1, C_{in}]$

3.  **搜索循环 (For Loop)**：
    *   `for alpha in [0, 0.1, ..., 1.0]:`
        *   计算 $s = s_X^\alpha \cdot s_W^{1-\alpha}$ （形状 $[1, C_{in}]$）
        *   $W_{new} = W \cdot s$ （广播乘法，列放大）
        *   $W_{q} = \text{Quantize}(W_{new})$ （按行分组量化，产生误差）
        *   $W_{recover} = W_{q} / s$ （恢复成原始比例，用于算误差）
        *   **计算误差**：$|| W_{recover} \cdot X - W \cdot X ||$
        *   记录误差最小的那个 $\alpha$。

4.  **输出**：
    *   得到最优的 **$s$ 向量**（形状 $[1, C_{in}]$）。
    *   这个 $s$ 会被融合到模型的前一层（LayerNorm 或 Linear）中，或者作为一个独立参数保存下来。

### 关键点回顾

*   **$s$ 是一个向量**：它长得跟输入特征通道数一样长。
*   **$s$ 的作用方向**：竖着乘到 $W$ 上（改变列的大小）。
*   **搜索的目的**：找到一个 $s$，让 $W$ 变形后，既能让 VIP 显眼，又不至于让 Group 里的平民完全没法活，从而使整体量化后的输出误差最小。