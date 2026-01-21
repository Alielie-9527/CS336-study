#  中文注释版

为了解MOE的实现方式，以及作用原理，本代码基于 [davidmrau/mixture-of-experts](https://github.com/davidmrau/mixture-of-experts) 添加了中文注释和维度说明，来方便小白更好理解和学习代码。
## 引用

如果使用本代码，请引用原作者：

```bibtex
@misc{rau2019moe,
    title={Sparsely-gated Mixture-of-Experts PyTorch implementation},
    author={Rau, David},
    journal={https://github.com/davidmrau/mixture-of-experts},
    year={2019}
}

## 代码讲解说明
  这部分代码是MOE的实现部分，主要包括门控网络（Gating Network）和专家网络（Experts）的定义和前向传播逻辑。还不是当前主流的transformer架构，只是一个基础的MOE实现。