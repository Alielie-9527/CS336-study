from typing import Optional, Tuple
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
from torch import nn as nn
from torch.nn import functional as F

# RMSnorm
class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-6):
        super().__init__()
        self.eps=eps
        self.scale=nn.Parameter(torch.ones(dim))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return x * self.scale / torch.sqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)
    

class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"

    def __init__(
            self,
            dim: int = 768,
            n_layers: int = 12,
            n_heads: int = 16,
            n_kv_heads: int = 8,
            vocab_size: int = 6144,
            hidden_dim: int = None,
            multiple_of: int = 64,
            norm_eps: float = 1e-5,
            max_seq_len: int = 512,
            dropout: float=0.0,
            flash_attn: bool =True,
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim * 4
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复键或值张量的头以匹配查询头的数量。
    Args:
        x: 输入的键或值张量，形状为 (batch_size, seq_len, n_kv_heads, head_dim)
        n_rep: 每个键/值头需要重复的次数 (n_rep = n_heads // n_kv_heads)
    Returns:
        重复后的张量，形状为 (batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1: # 如果不需要重复，直接返回原张量（相当于标准多头注意力MHA）
        return x
    # 核心操作：先增加一个维度，然后扩展，最后重塑形状
    return (
        x[:, :, :, None, :]              # 形状变为 [bs, slen, n_kv_heads, 1, head_dim]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim) # 扩展为 [bs, slen, n_kv_heads, n_rep, head_dim]
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim) # 重塑为 [bs, slen, n_kv_heads * n_rep, head_dim]
    )

# 旋转嵌入
# 注意：此处的dim应为 dim//n_head，因为我们是对每个head进行旋转嵌入
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float()生成了一个从0开始，步长为2的序列，长度为dim的一半
    # 然后每个元素除以dim，再取theta的倒数，得到频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成一个从0到end的序列，长度为end
    t = torch.arange(end, device=freqs.device)
    # 计算外积，得到一个二维矩阵，每一行是t的元素乘以freqs的元素
    freqs = torch.outer(t, freqs).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 获取x的维度数
    ndim = x.ndim
    
    # 断言，确保1在x的维度范围内
    assert 0 <= 1 < ndim
    
    # 断言，确保freqs_cis的形状与x的第二维和最后一维相同
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    
    # 构造一个新的形状，除了第二维和最后一维，其他维度都为1，这样做是为了能够将freqs_cis与x进行广播操作
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    
    # 将freqs_cis调整为新的形状，并返回
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1) # unbind沿最后一个维度分离实部和虚部
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重新塑形频率张量以进行广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转，分别计算旋转后的实部和虚部
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 将最后两个维度合并，并还原为原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self,args:ModelConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        assert args.dim % args.n_heads == 0, "dim must be divisible by n_heads"
        model_parallel_size = 1 # 并行处理
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_heads_kv = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_heads_kv
        # 每个头的维度
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim,self.n_local_heads*self.head_dim,bias=False)
        self.wk = nn.Linear(args.dim,self.n_local_heads_kv*self.head_dim,bias=False)
        self.wv = nn.Linear(args.dim,self.n_local_heads_kv*self.head_dim,bias=False)
        self.wo = nn.Linear(args.dim,args.dim,bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # 若不支持Flash Attention，则使用手动实现的注意力机制，并设置mask。
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 创建一个上三角矩阵，用于遮蔽未来信息。
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)
        
    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = x.size()  
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x) # 线性变换得到查询、键、值

        # 分离头部
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_heads_kv, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_heads_kv, self.head_dim)

        # 应用旋转嵌入
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # KV Cache logic
        if past_kv is not None:
            k_cache, v_cache = past_kv
            xk = torch.cat([k_cache, xk], dim=1)
            xv = torch.cat([v_cache, xv], dim=1)
        
        current_kv = (xk, xv)

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2) # bsz, n_heads, seq_len, head_dim
        xk = xk.transpose(1, 2) # bsz, n_heads, seq_len, head_dim
        xv = xv.transpose(1, 2) # bsz, n_heads, seq_len, head_dim

        if self.flash:
            # 使用Flash Attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=(past_kv is None)
            )
        else:
            # 计算点击缩放
            attn_score = torch.matmul(xq, xk.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if past_kv is None:
                assert hasattr(self, "mask"), "Mask buffer not found. Ensure model is initialized correctly."
                attn_score = attn_score + self.mask[:, :, :seq_len, :seq_len]
            
            attn_probs = torch.softmax(attn_score, dim=-1)
            attn_probs = self.dropout(attn_probs)
            attn_output = torch.matmul(attn_probs, xv)
        # 重新组合头部并通过输出线性层
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.wo(attn_output)
        output = self.resid_dropout(attn_output)

        return output, current_kv

class MLP(nn.Module):
    def __init__(self,dim:int,hidden_dim:int,multiple_of:int,dropout:float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim*4
            hidden_dim = int(2* hidden_dim / 3)
            # multiple_of调整hidden_dim为multiple_of的倍数 
            # 为什么要确保hidden_dim是multiple_of的倍数？
            # 这是为了优化计算效率，特别是在使用某些硬件加速器时，确保张量维度是特定数值的倍数可以提高内存对齐和计算速度。
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim,hidden_dim,bias=False)
        self.w2 = nn.Linear(hidden_dim,dim,bias=False)
        self.w3 = nn.Linear(dim,hidden_dim,bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        # 使用SiLU激活函数，并通过线性层进行前向传播
        # SiLU（Sigmoid Linear Unit）是一种常用的激活函数，定义为 x * sigmoid(x)。  
        # w3用于增加非线性变换的复杂度，从而提升模型的表达能力。
        # 门口信号机制：通过w3线性变换后的输出作为门控信号，调节主路径（w1和w2）的信息流。
        # 正确的 SwiGLU / gate 计算：先计算两个投影，然后逐元素相乘作为门控，最后再投影回原始维度。
        # 具体流程：
        # 1) a = F.silu(self.w1(x))          -> shape [..., hidden_dim]
        # 2) b = self.w3(x)                  -> shape [..., hidden_dim]
        # 3) gated = a * b                   -> shape [..., hidden_dim]
        # 4) out = self.w2(gated)            -> shape [..., dim]
        a = F.silu(self.w1(x)) # 激活信息
        b = self.w3(x)  # 门口信号
        gated = a * b
        return self.dropout(self.w2(gated))


class DecoderLayer(nn.Module):
    def __init__(self,layer_id:int,args:ModelConfig):
        super().__init__()
        self.layer_id =layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim # 输入维度
        self.head_dim = args.dim // args.n_heads # 每个头的维度
        self.attn = Attention(args)

        self.mlp = MLP(args.dim,args.hidden_dim,args.multiple_of,args.dropout)
        self.atten_norm = RMSNorm(args.dim,args.norm_eps)
        self.mlp_norm = RMSNorm(args.dim,args.norm_eps)
    
    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 注意力层
        x_norm = self.atten_norm(x)
        attn_output, current_kv = self.attn(x_norm, freqs_cos, freqs_sin, past_kv)
        x = x + attn_output
        # MLP层
        x_norm = self.mlp_norm(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output

        return x, current_kv
    

class LLaMA2Model(PreTrainedModel):
    config_class = ModelConfig
    last_loss : Optional[torch.Tensor]

    def __init__(self,args:ModelConfig):
        super().__init__(args)
        self.args = args
        self.vocab_size = args.vocab_size
        
        self.n_layers = args.n_layers

        # 词嵌入层
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        # dropout层
        self.dropout = nn.Dropout(args.dropout)

        # 解码器层
        self.layers = nn.ModuleList(
            [DecoderLayer(layer_id,args) for layer_id in range(args.n_layers)]
        )
        self.norm = RMSNorm(args.dim,args.norm_eps)

        self.output = nn.Linear(args.dim,args.vocab_size,bias=False)
    
        # 初始化权重 共享嵌入和输出层的权重
        self.tok_embeddings.weight = self.output.weight 

        freqs_cos, freqs_sin = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos,persistent=False)
        self.register_buffer("freqs_sin", freqs_sin,persistent=False)

        # 初始化模型参数
        self.apply(self._init_weights)
        # 残差投影特殊的
        for pn, p in self.named_parameters():
            if pn.endswith("w3.weight") or pn.endswith("wo.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * self.n_layers) ** 0.5)

        '''
        "w3.weight" - SwiGLU 中的门控投影层
        "wo.weight" - Attention 中的输出投影层
        这些层的权重初始化为均值为0.0，标准差为0.02 / sqrt(2 * n_layers) 的正态分布。
        这样做的目的是为了在训练开始时控制这些层的输出幅度，从而有助于稳定训练过程，特别是在深层网络中。
        '''
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()
        # 避免拆分模块
        self._no_split_modules = [name for name, _ in self.named_modules()]

    def _init_weights(self,module:nn.Module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, start_pos: int = 0, past_key_values=None, **kwargs) -> torch.Tensor:
        """
        - tokens: Optional[torch.Tensor], 输入 token 张量。
        - targets: Optional[torch.Tensor], 目标 token 张量。
        - start_pos: int, 旋转嵌入的起始位置。
        - past_key_values: list, 过去的键值对缓存。
        - kwargs: 其他关键字参数。

        - self.OUT: CausalLMOutputWithPast, 包含 logits 和损失。
        """
        # 兼容 transformers 调用：优先使用 input_ids / labels
        if 'input_ids' in kwargs and tokens is None:
            tokens = kwargs['input_ids']
        if 'labels' in kwargs and targets is None:
            targets = kwargs['labels']

        _bsz, seq_len = tokens.size()

        # 输入通过词嵌入层和dropout层
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        # 获取旋转嵌入的频率
        freqs_cos = self.freqs_cos[start_pos : start_pos + seq_len, :].to(h.device)
        freqs_sin = self.freqs_sin[start_pos : start_pos + seq_len, :].to(h.device)
        
        next_key_values = []
        # 通过每个解码器层
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            h, kv = layer(h, freqs_cos, freqs_sin, past_kv=past_kv)
            next_key_values.append(kv)
        
        h = self.norm(h)
        if targets is not None:
            logits = self.output(h)
            # 计算交叉熵损失，按 (batch * seq_len) 展平
            self.last_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
                reduction="none",
            )
        else:
            # 推理时只计算最后一个时间步的 logits，保持三维形状 (B, 1, V)
            logits = self.output(h[:, [-1], :])
            self.last_loss = None
        
        # 设置输出
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        self.OUT.__setitem__('past_key_values', next_key_values)
        return self.OUT
    
    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        给定输入序列 idx（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
        在 model.eval() 模式下运行。使用 KV Cache 加速。
        """
        # Prefill phase
        outputs = self.forward(idx)
        past_key_values = outputs.past_key_values
        logits = outputs.logits
        
        # Get the first next token
        logits = logits[:, -1, :]
        if temperature == 0.0:
            _, idx_next = torch.topk(logits, k=1, dim=-1)
        else:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        if idx_next == stop_id:
            return idx

        idx = torch.cat((idx, idx_next), dim=1)

        # Decoding phase
        for _ in range(max_new_tokens - 1):
            # Check if we reached max sequence length
            if idx.shape[1] >= self.args.max_seq_len:
                break
                
            # Only pass the last generated token
            outputs = self.forward(idx_next, start_pos=idx.shape[1]-1, past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == stop_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
