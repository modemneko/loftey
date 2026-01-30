<div align="center">

# Loftey
***[ENGLISH](https://github.com/modemneko/loftey/blob/master/README.md)|简体中文***

**一个轻量级的基于 Transformer 的语言模型，支持混合专家（Mixture of Experts, MoE）架构**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ModelScope](https://img.shields.io/badge/ModelScope-Loftey-red.svg)](https://www.modelscope.cn/models/LightNeko/loftey-mini-235m)

![Loftey Logo](https://raw.githubusercontent.com/modemneko/loftey/master/logo.png)

</div>

---

## 特性

- **混合专家（MoE）**：高效稀疏架构，支持可配置的专家路由
- **GQA（分组查询注意力）**：通过分组键值头减少内存占用
- **低功耗推理**：支持在低功耗设备上运行，如移动设备
- **灵活配置**：易于自定义的模型架构
- **HuggingFace 兼容**：与 [HuggingFace](https://huggingface.co/) 生态系统无缝集成

## 模型架构

Loftey 是一个仅解码器的 Transformer 模型，包含以下核心组件：

- **Attention**：支持分组查询注意力（GQA）的多头注意力机制
- **FeedForward**：可在标准 FFN 和基于 MoE 的 FFN 之间切换
- **Normalization**：输入和后注意力层均使用 RMSNorm

### 默认配置

| 参数 | 值 |
|-----------|-------|
| 词汇表大小 | 151,646 |
| 隐藏层大小 | 512 |
| 中间层大小 | 2,048 |
| 层数 | 6 |
| 注意力头数 | 16 |
| 键值头数 | 8 |
| 最大序列长度 | 512 |
| 专家数量 | 4 |
| Top-k 专家数 | 2 |

## 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- transformers >= 4.30.0
- safetensors >= 0.3.0

### 从源码安装

```bash
git clone https://github.com/modemneko/loftey.git
cd loftey
python -m venv venv
source venv/bin/activate  # 在 Windows 上使用 venv\Scripts\activate 
pip install -r requirements.txt
```

## 模型下载

预训练模型可在以下平台下载：

| 模型 | 参数量 | ModelScope | HuggingFace |
|------|--------|------------|-------------|
| Loftey Mini | 235M | [![ModelScope](https://img.shields.io/badge/ModelScope-Download-red.svg)](https://www.modelscope.cn/models/LightNeko/loftey-mini-235m) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Download-yellow.svg)](https://huggingface.co/Zeppesey/loftey-mini-235m) |

### 使用 ModelScope 下载模型

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download

model_dir = snapshot_download('LightNeko/loftey-mini-235m')
```

### 使用 HuggingFace 下载模型

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download(repo_id='Zeppesey/loftey-mini-235m')
```

下载后，将模型文件放置在项目的 `models/` 目录下，或在代码中指定正确的模型路径。

## 快速开始

### 推理示例

```python
from transformers import AutoTokenizer
from model import LofteyForCausalLM

# 加载模型和分词器
model = LofteyForCausalLM.from_pretrained('models')
tokenizer = AutoTokenizer.from_pretrained('models')

# 准备输入
prompt = "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# 生成回复
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id
)

# 解码输出
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
```

### 使用推理脚本

```bash
python infer.py
```

## 模型使用

### 加载预训练模型

```python
from model import LofteyForCausalLM, Config

# 从目录加载
model = LofteyForCausalLM.from_pretrained('path/to/model', device='cuda')

# 或使用自定义配置创建
config = Config(
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12,
    use_moe=True,
    expert_num=8
)
model = LofteyForCausalLM(config)
```

## 性能指标

- **参数量**：约 235M（默认配置）
- **内存占用**：约 900MB（FP32）
- **推理速度**：CPU 上约 30 tokens/s（取决于硬件）

## 许可证

本项目采用 [MIT 许可证](LICENSE) - 详见 LICENSE 文件。

## 引用

如果您在研究中使用了 Loftey，请考虑引用：

```bibtex
@software{loftey2026,
  title={Loftey: A Lightweight MoE Language Model},
  author={modemneko},
  year={2026},
  url={https://github.com/modemneko/loftey}
}
```

## 致谢

- 235M 参数模型使用 [Qwen3 0.6B](https://www.modelscope.cn/models/qwen/Qwen3-0.6B-Instruct) 的 tokenizer 训练
- 模型训练数据来源于 [minimind](https://github.com/jingyaogong/minimind)
- 基于 [PyTorch](https://pytorch.org/) 和 [HuggingFace Transformers](https://huggingface.co/docs/transformers/) 构建

## 贡献

欢迎贡献！请随时提交 [Pull Request](https://github.com/modemneko/loftey/pulls)。

## 联系方式

如有问题或反馈，请在 [GitHub](https://github.com/modemneko/loftey/issues) 上提交 issue。

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 Star！**

Made with love by [modemneko](https://github.com/modemneko)

</div>
