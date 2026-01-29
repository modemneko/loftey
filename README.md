<div align="center">

# Loftey
![Loftey Logo](https://raw.githubusercontent.com/modemneko/loftey/master/logo.png)

**A lightweight Transformer-based language model with Mixture of Experts (MoE) support**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ModelScope](https://img.shields.io/badge/ModelScope-Loftey-red.svg)](https://www.modelscope.cn/models/LightNeko/loftey-mini-235m)

</div>

---

## Features

- **Mixture of Experts (MoE)**: Efficient sparse architecture with configurable expert routing
- **GQA (Grouped Query Attention)**: Reduced memory footprint with grouped key-value heads
- **Low-power Inference**: Supports running on low-power devices, such as mobile devices
- **Flexible Configuration**: Easily customizable model architecture
- **HuggingFace Compatible**: Seamless integration with [HuggingFace](https://huggingface.co/) ecosystem

## Model Architecture

Loftey is a decoder-only Transformer model with the following core components:

- **Attention**: Multi-head attention with grouped query attention (GQA)
- **FeedForward**: Switchable between standard FFN and MoE-based FFN
- **Normalization**: RMSNorm for both input and post-attention layers

### Default Configuration

| Parameter | Value |
|-----------|-------|
| Vocab Size | 151,646 |
| Hidden Size | 512 |
| Intermediate Size | 2,048 |
| Number of Layers | 6 |
| Attention Heads | 16 |
| Key-Value Heads | 8 |
| Max Sequence Length | 512 |
| Number of Experts | 4 |
| Top-k Experts | 2 |

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- transformers >= 4.30.0
- safetensors >= 0.3.0

### Install from source

```bash
git clone https://github.com/modemneko/loftey.git
cd loftey
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
pip install -r requirements.txt
```

## Model Download

Pre-trained models are available on the following platforms:

| Model | Parameters | ModelScope | HuggingFace |
|-------|------------|------------|-------------|
| Loftey Mini | 235M | [![ModelScope](https://img.shields.io/badge/ModelScope-Download-red.svg)](https://www.modelscope.cn/models/LightNeko/loftey-mini-235m) | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Download-yellow.svg)](https://huggingface.co/Zeppesey/loftey-mini-235m) |

### Download using ModelScope

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download

model_dir = snapshot_download('LightNeko/loftey-mini-235m')
```

### Download using HuggingFace

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download(repo_id='Zeppesey/loftey-mini-235m')
```

After downloading, place the model files in the `models/` directory of the project, or specify the correct model path in your code.

## Quick Start

### Inference Example

```python
from transformers import AutoTokenizer
from model import LofteyForCausalLM

# Load model and tokenizer
model = LofteyForCausalLM.from_pretrained('models')
tokenizer = AutoTokenizer.from_pretrained('models')

# Prepare input
prompt = "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate response
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id
)

# Decode output
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
```

### Using the inference script

```bash
python infer.py
```

## Model Usage

### Loading a pretrained model

```python
from model import LofteyForCausalLM, Config

# Load from directory
model = LofteyForCausalLM.from_pretrained('path/to/model', device='cuda')

# Or create with custom config
config = Config(
    hidden_size=768,
    num_layers=12,
    num_attention_heads=12,
    use_moe=True,
    expert_num=8
)
model = LofteyForCausalLM(config)
```

## Performance Metrics

- **Parameters**: ~235M (default config)
- **Memory Usage**: ~900MB (FP32)
- **Inference Speed**: ~30 tokens/s on CPU (hardware dependent)

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Citation

If you use Loftey in your research, please consider citing:

```bibtex
@software{loftey2026,
  title={Loftey: A Lightweight MoE Language Model},
  author={modemneko},
  year={2026},
  url={https://github.com/modemneko/loftey}
}
```

## Acknowledgments

- The 235M parameter model uses the tokenizer from [Qwen3 0.6B](https://www.modelscope.cn/models/qwen/Qwen2.5-0.5B-Instruct)
- Training data sourced from [minimind](https://github.com/jingyaogong/minimind)
- Built with [PyTorch](https://pytorch.org/) and [HuggingFace Transformers](https://huggingface.co/docs/transformers/)

## Contributing

Contributions are welcome! Please feel free to submit a [Pull Request](https://github.com/modemneko/loftey/pulls).

## Contact

For questions and feedback, please open an issue on [GitHub](https://github.com/modemneko/loftey/issues).

---

<div align="center">

**If this project helps you, please give us a Star!**

Made with love by [modemneko](https://github.com/modemneko)

</div>
