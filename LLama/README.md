# Tiny-LLM Implementation (LLaMA2)

本项目实现了一个轻量级的 LLaMA2 模型训练框架（参照DataWhale），涵盖了从分词器训练、预训练 (Pre-training) 到指令微调 (SFT) 的全流程。代码结构清晰，适合学习和研究 LLM 的训练细节。

## 📂 文件说明

- **`tokenizer.py`**: BPE 分词器训练脚本。基于 `tokenizers` 库，支持自定义特殊 token (`<|im_start|>`, `<|im_end|>` 等) 和 Chat 模板（用于多轮对话）。
- **`train_model.py`**: 主训练脚本。包含数据集处理 (`PretrainDataset`, `SFTDataset`)、训练循环、混合精度训练、梯度累积等逻辑。
- **`train.sh`**: 启动训练的 Shell 脚本示例。

## 🚀 功能特性

- **自定义分词器**: 训练 Byte-Level BPE 分词器，针对中文优化。（custom_tokenizer 目录存储分词器配置,不建议全部训练，对内存要求过高）
- **预训练 (Pretraining)**: 支持流式读取 JSONL 数据进行因果语言模型训练，数据集来源“出门问问猴子”的中文语料库。
- **指令微调 (SFT)**: 支持对话格式数据微调，实现了 **Loss Mask** 机制，仅计算 Assistant 回复部分的 Loss。数据集来源： BellGroup 的开源指令微调数据集。
- **训练优化**:
  - **混合精度 (Mixed Precision)**: 支持 `bfloat16` 和 `float16`，利用 `torch.amp` 加速。
  - **梯度累积 (Gradient Accumulation)**: 支持在有限显存下模拟大 Batch Size 训练。
  - **梯度裁剪 (Gradient Clipping)**: 防止梯度爆炸，稳定训练过程。
  - **余弦退火学习率 (Cosine Annealing)**: 包含 Warmup 的学习率调度策略。
- **实验追踪**: 集成 SwanLab 进行指标可视化。

## 🛠️ 使用指南

### 1. 训练分词器 (Tokenizer)

如果需要训练自己的分词器：

```bash
python LLama/tokenizer.py
```

这将读取数据并保存分词器配置到 `./custom_tokenizer` 目录。

### 2. 预训练 (Pretraining)

使用 `train.sh` 脚本启动单卡训练：

```bash
bash LLama/train.sh
```

或者直接运行 Python 命令：

```bash
python LLama/train_model.py \
    --out_dir "base_model_215M" \
    --data_path "./LLama/seq_monkey_datawhale.jsonl" \
    --epochs 3 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --device "cuda:0" \
    --dtype "float16"
```

### 3. 指令微调 (SFT)

对于 SFT 任务，脚本会自动识别文件名中是否包含 `sft` (或者手动修改代码逻辑)，并使用 `SFTDataset`。

```bash
python LLama/train_model.py \
    --out_dir "sft_model_215M" \
    --data_path "./LLama/BellGroup_sft.jsonl" \
    --learning_rate 3e-4
```

## 📊 数据格式（需要自己下载）

- **预训练数据**: JSONL 格式，每行包含 `text` 字段。
- **SFT 数据**: JSONL 格式，符合 Chat 模板结构 (List of messages)。

## ⚙️ 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--accumulation_steps` | 梯度累积步数，用于模拟大 Batch Size | 8 |
| `--dtype` | 数据类型 (`float16` 或 `bfloat16`) | `bfloat16` |
| `--grad_clip` | 梯度裁剪阈值 | 1.0 |
| `--use_swanlab` | 是否开启 SwanLab 记录 | False |
