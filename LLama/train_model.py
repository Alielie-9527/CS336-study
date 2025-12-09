import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset  # 添加这行导入
from transformers import AutoTokenizer
from torch.nn import functional as F
from tqdm import tqdm
import argparse
import time
from LLaMA2 import LLaMA2Model
from LLaMA2 import ModelConfig

# test tokenizer
# text = "<|im_start|>assistant\n"     # [   0,   69,   87,   87,   77, 7421, 2425,   88,  203]
# encoded_input = tokenizer(text, return_tensors='pt')
# print(encoded_input['input_ids'])
# decoded_output = tokenizer.decode(encoded_input['input_ids'][0])
# print(decoded_output)


class PretrainDataset(Dataset):
    def __init__(self,data_path,tokenizer,max_length=512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length # 最大序列长度
        # 这里用unk 做padding
        self.padding = 2
        with open(data_path,'r',encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        sample = json.loads(self.data[idx])
        text = f"{self.tokenizer.bos_token}{sample['text']}"
        input_id = self.tokenizer.encode(text)[:self.max_length]
        text_len = len(input_id)
        # padding
        if text_len < self.max_length:
            input_id +=  [self.padding] * (self.max_length - text_len)
        loss_mask = [1]*text_len + [0] * (self.max_length - text_len)
        input_id = np.array(input_id,dtype=np.int64)
        loss_mask = np.array(loss_mask,dtype=np.float32)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)


class SFTDataset(Dataset):
    def __init__(self,data_path,tokenizer,max_length=512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length # 最大序列长度
        # 这里用unk 做padding
        self.padding = 2
        with open(data_path,'r',encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)
    
    def generate_loss_mask(self, input_ids):
        loss_mask = np.zeros_like(input_ids, dtype=np.int64)
        # <|im_start|>assistant\n
        a_sequence = np.array([0, 69, 87, 87, 77, 7421, 2425, 88, 203])
        a_len = len(a_sequence)
        n = len(input_ids)
        
        # 确保 input_ids 是 numpy array 以利用向量化操作
        if not isinstance(input_ids, np.ndarray):
            input_ids = np.array(input_ids)

        i = 0
        while i <= n - a_len:
            # 优化: 使用 numpy 切片比较，替代内层循环
            if np.array_equal(input_ids[i:i+a_len], a_sequence):
                # 从子序列结束的位置开始查找第一个 1 (<|im_end|>)
                # 使用 np.where 替代循环查找
                remaining = input_ids[i+a_len:]
                eos_indices = np.where(remaining == 1)[0]
                
                if eos_indices.size > 0:
                    # 找到 EOS，计算绝对位置
                    j = i + a_len + eos_indices[0]
                    # 优化: 使用切片赋值替代循环赋值
                    loss_mask[i+a_len : j+1] = 1
                    i = j + 1
                else:
                    # 未找到 EOS，跳过该前缀
                    i += a_len
            else:
                i += 1
        return loss_mask
    
    def __getitem__(self,idx):
        sample = json.loads(self.data[idx])
        text = self.tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        input_id = self.tokenizer.encode(text)[:self.max_length]
        text_len = len(input_id)
        # padding
        if text_len < self.max_length:
            input_id +=  [self.padding] * (self.max_length - text_len)
        loss_mask = self.generate_loss_mask(input_id)
        input_id = np.array(input_id,dtype=np.int64)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)


# 标准的训练流程

# 学习率计算
def get_lr(it,all):
        """
    计算当前迭代的学习率，使用余弦退火调度策略
    
    学习率调度策略：
    1. Warmup阶段：学习率从0线性增长到目标学习率
    2. 余弦退火阶段：学习率按余弦函数衰减到最小学习率
    3. 超出训练步数后：保持最小学习率
    
    Args:
        it (int): 当前迭代步数
        all (int): 总迭代步数
        
    Returns:
        float: 当前步数对应的学习率
    """
        warmup_steps = args.warmup_iters # 预热步数
        lr_decay_steps = all - warmup_steps # 学习率衰减步数
        min_lr = args.learning_rate * 0.1 # 最小学习率为初始学习率的10%

        if it < warmup_steps:
            # 线性增长阶段
            lr = args.learning_rate * it / warmup_steps
        elif it < all:
            # 余弦退火阶段
            # 计算公式： lr = min_lr + 0.5 * (args.learning_rate - min_lr) * (1 + np.cos(np.pi * (it - warmup_steps) / lr_decay_steps))
            lr = min_lr + 0.5 * (args.learning_rate - min_lr) * (1 + np.cos(np.pi * (it - warmup_steps) / lr_decay_steps))
        else:
            # 超出总步数，保持最小学习率
            lr = min_lr
        return lr


def train_epoch(epoch, train_loader, model, optimizer, device):
    '''
    训练一个epoch
    1. 数据加载，转移到设备
    2. 前向传播，计算损失
    3. 反向传播，梯度裁剪
    4. 优化器更新参数
    '''
    start_time = time.time()
    # 优化1: 将 model.train() 移到循环外，避免每次迭代重复调用
    model.train()
    
    # 使用 tqdm 可视化每个 epoch 的进度
    loader = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for batch_idx, (X, Y, loss_mask) in enumerate(loader):
        # 优化2: 使用 non_blocking=True，允许数据传输与 GPU 计算重叠
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        loss_mask = loss_mask.to(device, non_blocking=True)

        # 获得学习率 按照step进行跟新，一个epoch 进行了多次 step 按照step进行更新
        iter_per_epoch = len(train_loader)
        current_step = epoch * iter_per_epoch + batch_idx
        total_steps = args.epochs * iter_per_epoch
        
        lr = get_lr(current_step, total_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # 使用混合精度训练上下文
        with ctx:
            # 前向传播
            out = model(X, Y)
            # 计算损失并除以累积步数（用于梯度累积）
            loss = out.last_loss / args.accumulation_steps
            # 将loss_mask展平为一维
            loss_mask = loss_mask.view(-1)
            # 应用掩码计算有效损失（忽略padding位置）
            # 优化3: 添加 1e-6 防止除以零的数值稳定性问题
            loss = torch.sum(loss * loss_mask) / (loss_mask.sum() + 1e-6)

   
        # 反向传播
        # 使用 scaler 处理梯度缩放，防止 FP16 下溢
        scaler.scale(loss).backward()

        # 梯度累积：每 accumulation_steps 步更新一次参数
        if (batch_idx + 1) % args.accumulation_steps == 0:
            # 梯度裁剪前必须先 unscale 梯度
            scaler.unscale_(optimizer)
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # 更新参数
            scaler.step(optimizer)
            
            # 更新 scaler 的缩放因子
            scaler.update()
            
            # 清空梯度       
            # 优化4: set_to_none=True 比默认的 set to 0 更快，减少内存写操作
            optimizer.zero_grad(set_to_none=True)


        # 日志与进度条更新
        if batch_idx % args.log_interval == 0:
            # 还原真实的 loss 值用于显示 (乘以 accumulation_steps)
            real_loss = loss.item() * args.accumulation_steps
            speed = (batch_idx + 1) / (time.time() - start_time + 1e-6)
            # 更新 tqdm 的后缀信息（便于可视化）
            loader.set_postfix({'loss': f"{real_loss:.4f}", 'lr': f"{lr:.6f}", 'it/s': f"{speed:.2f}"})
            # 额外日志（可选）
            if args.use_swanlab:
                import swanlab
                swanlab.log({"train/loss": real_loss, "train/lr": lr}, step=current_step)


def init_model(device):
    '''
    init_model 的 Docstring
    
    '''
    config = ModelConfig(vocab_size=8192,max_seq_len=512)
    model = LLaMA2Model(config)
    model.to(device)
    return model    


if __name__ == "__main__":
    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining")
    
    # 基础训练参数
    parser.add_argument("--out_dir", type=str, default="base_model_215M", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
    
    # 实验跟踪和数据加载参数
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用SwanLab进行实验跟踪")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载的工作进程数")
    parser.add_argument("--data_path", type=str, default="./LLama/seq_monkey_datawhale.jsonl", help="训练数据路径")
    
    # 训练优化参数
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="学习率预热迭代次数")
    
    # 日志和保存参数
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    
    # 多GPU训练参数 ctx =  torch.amp.autocast(device_type=args.device.split(':')[0], dtype=getattr(torch, args.dtype))
    parser.add_argument("--gpus", type=str, default='0,1,2,3,4,5,6,7', help="使用的GPU ID，用逗号分隔 (例如: '0,1,2')")

    args = parser.parse_args()
    

    # 混合精度上下文管理器
   
    # ========================================================
    

    # 训练好的分词器
    # 注意：这里假设在项目根目录下运行，且 custom_tokenizer 在根目录
    tokenizer = AutoTokenizer.from_pretrained('./LLama/custom_tokenizer')
    # 训练数据集和数据加载器
    # 根据数据路径初始化数据集
    file_name = os.path.basename(args.data_path)
    if 'sft' in file_name:
        train_dataset = SFTDataset(args.data_path,tokenizer,max_length=512)
    else:
        train_dataset = PretrainDataset(args.data_path,tokenizer,max_length=512)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 模型初始化
    model = init_model(args.device)

    # ==================== 优化器和训练组件初始化 ====================
    # 初始化混合精度训练的梯度缩放器
    # 注意：bfloat16 通常不需要 GradScaler，因为它的动态范围足够大
    # 只有 float16 需要 GradScaler 来防止梯度下溢
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))

    # adam优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ============================训练循环=============================
    total_steps = args.epochs * len(train_loader)
    print("Starting training...")
    for epoch in range(args.epochs):
        train_epoch(epoch,train_loader,model,optimizer,args.device)
    