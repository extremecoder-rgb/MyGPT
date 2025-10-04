# MyGPT a fully custom implementation of the GPT (Generative Pre-trained Transformer) architecture in PyTorch. The project features everything from scratch tokenizers to advanced parameter-efficient fine-tuning, supporting both educational and research workflows for large language models.

***

## Overview**MyGPT** is designed for deep learning practitioners who want to understand, customize, or extend transformer-based language models. It supports modular tokenization (including BPE), full transformer stack construction, robust training and evaluation pipelines, native integration of OpenAI GPT-2 pretrained weights, and LoRA-based fine-tuning for cost-effective adaptation.

***

## Architectural ChartGPT Model Architecture - Complete data flow from tokenization to text generation:***

## Directory Structure```
MyGPT/
├── MyGPT.ipynb                              # Main implementation notebook (model logic & experiments)
├── gpt_download3.py                         # Helper for programmatic GPT-2 weights download & parsing
├── 01 Harry Potter and the Sorcerers Stone.txt  # Sample text dataset for training/testing
└── .gitignore                               # Standard ignore file
```

***

## Major Features- **Tokenizer Construction**: Build both word-level and subword-level (BPE) tokenizers. Supports special tokens and unknown mapping.
- **Custom Dataset Loader**: PyTorch-ready, handling long text with sliding windows and overlapping sequences, plus custom collate/padding strategies.
- **Embedding Layers**: Implements token and positional embedding with real GPT-2 dimensions (vocab_size=50257; emb_dim=768).
- **Multi-Head Attention & Transformer Blocks**: Full reproducibility of GPT attention, causal masking, layer normalization, and feed-forward stacks using GELU activations.
- **Text Generation**: Rich sampling options: temperature, top-k, top-p, greedy decoding, EOS handling.
- **Pretrained Weights Loader**: Download, parse, and map original OpenAI GPT-2 weights directly into model layers.
- **Fine-Tuning with LoRA**: Parameter-efficient adaptation on datasets like Stanford Alpaca using HuggingFace PEFT.
- **Training & Evaluation Pipeline**: Built-in loss computation, optimizer config (AdamW), and sample generation.

***

## How It Works: Component Breakdown### 1. Tokenization- **Simple Tokenizer**: Regex splitting, vocabulary mapping, encoding/decoding functions.
- **Advanced BPE Tokenizer**: Integrates OpenAI's `tiktoken` for true subword tokenization and compatibility.

### 2. Data Processing- **Custom PyTorch Dataset**: Input/target generation via sliding windows.
- **Loader Factory**: Batch size, window length, stride, shuffle, and worker config for DataLoader.

### 3. Model Architecture- **Token Embedding**: Dense vector mapping of input token IDs.
- **Positional Embedding**: Absolute position vectors for sequence modeling.
- **Transformer Block(s)**:
    - LayerNorm (applied before both attention & feedforward)
    - MultiHeadAttention (Q, K, V computation, causal mask, dropout, residual)
    - FeedForward (2 linear + GELU + dropout + residual)
- **Final Layers**: LayerNorm, linear projection to vocab logits, softmax for text generation.

### 4. Training & Generation- **Optimizer**: AdamW (with weight decay)
- **Loss**: CrossEntropy
- **Evaluation**: Validation and real-time sampling in training loop.

### 5. Pretrained Model Support- **Weights Download/Parsing**: Supports GPT-2 Small, Medium, Large, XL models.
- **Custom Layer Mapping**: Ensures shape consistency and parameter assignment.

### 6. LoRA Fine-Tuning- **PEFT Integration**: Hooks for adapting only critical layers using LoRA, drastically reducing memory footprint.
- **Instruction Dataset Support**: Format and pad Stanford Alpaca or custom instruction data.

***

## Usage Examples### Setup```bash
pip install torch torchvision torchaudio tiktoken transformers peft numpy tqdm requests
```

### Basic Training Loop```python
from MyGPT import GPTModel, create_dataloader, train_model_simple

raw_text = open("01 Harry Potter and the Sorcerers Stone.txt").read()
train_loader = create_dataloader(raw_text, batch_size=8, max_length=1024)
model = GPTModel(config=GPT_CONFIG)
train_model_simple(model, train_loader, None, optimizer, device, num_epochs=10)
```

### Text Generation```python
context = "The wizard entered the room"
ids = tokenizer.encode(context)
out = generate(model, torch.tensor([ids]), max_new_tokens=50, temperature=0.8)
print(tokenizer.decode(out[0].tolist()))
```

### Loading and Fine-Tuning Pretrained Weights```python
from gpt_download3 import download_and_load_gpt2
settings, params = download_and_load_gpt2("124M", "gpt2")
load_weights_into_gpt(model, params)
```

### LoRA Fine-Tuning```python
from peft import LoraConfig, get_peft_model, TaskType
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj", "c_fc"]
)
model = get_peft_model(model, peft_config)
```

***

## Parameters & Configuration### GPT-2 Default```python
GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

### Custom Model Example```python
GPT_CONFIG_SMALL = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

***

## Advanced Implementation Highlights- **Causal Masking**: Enforces autoregressive restriction by upper-triangular masking in attention scores.
- **Positional Encoding**: Learned embedding, absolute positioning for up to 1024 tokens.
- **Layer Normalization**: Pre-LN architecture for stability.
- **Dropout/Regularization**: Both on embeddings and inside blocks; AdamW optimizer for robustness.
- **Flexible Sampling**: Supports temperature, top-k, top-p (nucleus), greedy and EOS-aware sampling.
- **Custom Collate Functions**: Handles dynamic padding, masking for instruction datasets.

***

## Requirements- Python 3.7+
- PyTorch 1.12+
- tiktoken
- transformers
- peft
- numpy
- tqdm
- requests

***

## LicenseMIT License

***

## Reference Links- [Attention Is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)
- [Stanford Alpaca Dataset](https://github.com/tatsu-lab/stanford_alpaca)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [OpenAI GPT-2](https://openai.com/research/better-language-models)

***

Feel free to further tweak this README for maximum clarity or branding!Here’s a fully professional README tailored for your MyGPT repository, suitable for public release, research, and portfolio purposes. It gives deep technical insight and clarity for both beginners and advanced users.

***

# MyGPT**A full-featured GPT (Generative Pre-trained Transformer) implementation from scratch with PyTorch, supporting custom tokenization, OpenAI GPT-2 compatibility, and LoRA-based fine-tuning.**

***

## Architecture OverviewBelow is a high-level diagram showing the entire data flow of the GPT model, from input tokenization through transformer blocks to the final text output.

GPT Model Architecture - Complete data flow from tokenization to text generation


***

## Repository Structure```
MyGPT/
├── MyGPT.ipynb                  # End-to-end implementation (tokenizer, model, experiments)
├── gpt_download3.py             # Utility to fetch and map GPT-2 weights
├── 01 Harry Potter...txt        # Example corpus for training
└── .gitignore                   # Standard ignore file
```

***

## Key Features- **Tokenization**: Word-level, custom, and Byte Pair Encoding (BPE) via tiktoken
- **PyTorch Dataset**: Efficient input/target creation with flexible sliding windows for language modeling
- **Embedding Layers**: Token and positional embeddings (GPT-2 format)
- **Transformer Block(s)**: Includes LayerNorm, multi-head causal attention, skip/residual connections, and feed-forward networks (GELU)
- **Text Generation**: Temperature, top-k, top-p, greedy and EOS stopping options
- **Pretrained Model Loading**: Download and utilize OpenAI GPT-2 weights (all sizes)
- **Fine-Tuning**: LoRA-enabled parameter-efficient adaptation (PEFT library)
- **Training/Evaluation**: AdamW optimizer, full train loop, sample generation with validation tracking

***

## Model Building: Step-by-Step### 1. Tokenization- **Simple**: Regex-based, full word/token mapping
- **Advanced/BPE**: Uses tiktoken for robust, language-model compatible subword encoding

### 2. Data Pipeline- `GPTDataset`: PyTorch dataset turns raw text into chunked input/target pairs
- Dataloader creation is customizable for batch size, sequence length, stride, shuffle, and workers

### 3. GPT Model Components- **Token/Positional Embedding**: Each input token combines a learned vector and positional context
- **Dropout Layer**: Regularization at embedding level
- **Transformer Blocks**: Each block features:
    - LayerNorm (pre-attention and pre-MLP)
    - Multi-Head Causal Attention (query, key, value matrices; masking prevents peeking ahead)
    - Feed-Forward (linear layers + GELU activation)
    - Residual/skip connections
- **Final LayerNorm & Projection**: Normalization and mapping back to vocab logits
- **Softmax**: Converts output logits to probabilities for sampling/generated tokens

### 4. Text Generation- Generate flexible sequences with temperature for creativity, top-k/p for diversity, and EOS for completion.

### 5. OpenAI GPT-2 Support- Download and parse weights for models sized 124M, 355M, 774M, and 1558M
- Automatic layer mapping and assignment

### 6. LoRA Fine-Tuning- HuggingFace PEFT integration: efficiently fine-tune targeted layers
- Alpaca and other instruction datasets supported

***

## Usage Examples### Training```python
from MyGPT import GPTModel, create_dataloader, train_model_simple

raw_text = open("01 Harry Potter and the Sorcerers Stone.txt").read()
train_loader = create_dataloader(raw_text, batch_size=8, max_length=1024, stride=256)
model = GPTModel(GPT_CONFIG)
train_model_simple(model, train_loader, None, optimizer, device, num_epochs=5)
```

### Text Generation```python
context_ids = tokenizer.encode("The wizard entered the room")
generated = generate(model, torch.tensor([context_ids]), max_new_tokens=50, temperature=0.8)
print(tokenizer.decode(generated[0].tolist()))
```

### Load GPT-2 Weights```python
from gpt_download3 import download_and_load_gpt2
settings, params = download_and_load_gpt2("124M", "gpt2")
load_weights_into_gpt(model, params)
```

### LoRA Fine-Tuning```python
from peft import LoraConfig, get_peft_model, TaskType
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj", "c_fc"]
)
model = get_peft_model(model, peft_config)
```

***

## Model Configuration Example```python
GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

***

## Requirements- Python 3.7+
- PyTorch 1.12+
- tiktoken
- transformers
- peft
- numpy
- tqdm
- requests

***

## LicenseMIT (Open Source)

***- Attention Is All You Need (Transformer Architecture)
- OpenAI GPT-2 Research
- LoRA (Low-Rank Adaptation)
- Stanford Alpaca Dataset

***

**For issues, feature requests, and contributions, please open a GitHub Issue or Pull Request.**

***