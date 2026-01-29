from transformers import AutoTokenizer
import torch
from model import LofteyForCausalLM, Config

# 初始化模型
model = LofteyForCausalLM.from_pretrained('models')

# 加载 tokenizer
t = AutoTokenizer.from_pretrained('models')

# 对话格式
prompt = "<|im_start|>user\n你有名字吗？<|im_end|>\n<|im_start|>assistant\n"
input_data = t.encode(prompt, add_special_tokens=False)
print(f"Input tokens: {input_data}")
print(f"Input text: {t.decode(input_data)}")

# 生成回复
input_ids = torch.tensor(input_data).unsqueeze(0)
for token in model.generate(input_ids, max_new_tokens=50, temperature=0.7, top_k=5, eos_token_id=t.eos_token_id, stream=False):
    output_text = t.decode(token[0])
    print(f"\nGenerated text:\n{output_text}")