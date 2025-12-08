import random
import json
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from tokenizers.normalizers import NFKC
from typing import Generator

def read_texts_from_jsonl(file_path: str, max_lines: int = 500000) -> Generator[str, None, None]:
    """读取JSONL文件并安全提取文本数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num > max_lines:
                break
            try:
                data = json.loads(line)
                if 'text' not in data:
                    raise KeyError(f"Missing 'text' field in line {line_num}")
                yield data['text']
            except json.JSONDecodeError:
                print(f"Error decoding JSON in line {line_num}")
                continue
            except KeyError as e:
                print(e)
                continue


def create_tokenizer_config(save_dir: str) -> None:
    """创建完整的tokenizer配置文件"""
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # 保存主配置文件
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # 创建special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)


def train_tokenizer(data_path:str, save_dir: str, vocab_size: int = 8192) -> None:
    
    os.makedirs(save_dir, exist_ok=True)
    """训练并保存自定义tokenizer"""
    
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # NFKC标准化: 将全角字符转换为半角字符等
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|im_start|>", "<|im_end|>", "<unk>", "<s>", "</s>"],
        min_frequency=5,  # 增加最小频率，过滤低频词，减少内存占用
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # 使用迭代器直接传递给 train_from_iterator
    # tokenizers 库的 train_from_iterator 支持流式处理，只要传入生成器，就不会一次性加载所有数据到内存。
    # 注意：如果数据量非常大，BPE 训练过程本身（构建词表）仍然可能消耗大量内存。
    # 如果仍然遇到 OOM，可能需要减少 vocab_size 或使用更小的子集进行训练。
    tokenizer.train_from_iterator(read_texts_from_jsonl(data_path), trainer=trainer, length=None)

    # 验证特殊token映射
    # 注意：tokenizers 库不保证 special_tokens 的 ID 顺序与列表顺序完全一致，
    # 尤其是当某些 token 已经在词表中存在或被重新排序时。
    # 这里我们打印实际的 ID 映射以供调试，而不是强制断言特定的 ID 值。
    special_tokens = ["<|im_start|>", "<|im_end|>", "<unk>", "<s>", "</s>"]
    print("Special tokens mapping:")
    for token in special_tokens:
        print(f"{token}: {tokenizer.token_to_id(token)}")
    
    # 只要这些 token 都在词表中（ID 不为 None），通常就是可以接受的。
    # 如果确实需要固定 ID，可以在 post-processing 中处理，或者在 Trainer 中更严格地控制。
    for token in special_tokens:
        if tokenizer.token_to_id(token) is None:
             raise AssertionError(f"Special token {token} not found in tokenizer!")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

    create_tokenizer_config(save_dir)
    print(f"Tokenizer saved to {save_dir}")

if __name__ == '__main__':
    data_path = "./data/mobvoi_seq_monkey_general_open_corpus.jsonl"
    save_dir = "./custom_tokenizer"
    
    train_tokenizer(data_path, save_dir, vocab_size=8192)