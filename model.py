import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import os


class Config:
    model_type = "loftey"
    
    def __init__(
        self,
        vocab_size: int = 151646,
        hidden_size: int = 512,
        intermediate_size: int = 2048,
        num_layers: int = 6,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        max_seq_len: int = 512,
        expert_num: int = 4,
        topk: int = 2,
        mlp_bias: bool = False,
        use_moe: bool = True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_seq_len = max_seq_len
        self.expert_num = expert_num
        self.topk = topk
        self.mlp_bias = mlp_bias
        self.use_moe = use_moe
        
        for key, value in kwargs.items():
            setattr(self, key, value)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MoEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.expert_num = config.expert_num
        self.topk = config.topk
        self.gate = nn.Linear(config.hidden_size, self.expert_num, bias=False)
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(self.expert_num)])

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        router_logits = self.gate(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, self.topk, dim=-1)
        
        output = torch.zeros_like(x_flat)
        
        for i in range(self.expert_num):
            expert_mask = (topk_indices == i).any(dim=-1)
            if expert_mask.sum() > 0:
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[i](expert_input)
                
                expert_positions = (topk_indices[expert_mask] == i).nonzero(as_tuple=True)[1]
                expert_weights = topk_probs[expert_mask, expert_positions]
                output[expert_mask] += expert_output * expert_weights.unsqueeze(-1)
        
        return output.view(batch_size, seq_len, hidden_size)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        k = self.repeat_kv(k, self.num_heads // self.num_key_value_heads)
        v = self.repeat_kv(v, self.num_heads // self.num_key_value_heads)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(output)

    def repeat_kv(self, x, n_rep):
        batch_size, n_kv_h, seq_len, head_dim = x.shape
        if n_rep == 1: return x
        return x[:, :, None, :, :].expand(batch_size, n_kv_h, n_rep, seq_len, head_dim).reshape(batch_size, n_kv_h * n_rep, seq_len, head_dim)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = MoEFeedForward(config) if config.use_moe else FeedForward(config)

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x
        
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class LofteyModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = nn.ModuleDict({
            'embed_tokens': nn.Embedding(config.vocab_size, config.hidden_size),
            'layers': nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)]),
            'norm': RMSNorm(config.hidden_size)
        })
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.model['embed_tokens'](input_ids)
        
        batch_size, seq_len = input_ids.shape
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask.bool(), float('-inf'))
        
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(hidden_states.dtype).min
            combined_mask = causal_mask + extended_attention_mask
        else:
            combined_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        for layer in self.model['layers']:
            hidden_states = layer(hidden_states, combined_mask)
        
        hidden_states = self.model['norm'](hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        stream: bool = False
    ):
        self.eval()
        batch_size = input_ids.shape[0]
        
        for step in range(max_new_tokens):
            outputs = self.forward(input_ids)
            logits = outputs[:, -1, :]
            
            if temperature > 0:
                logits = logits / temperature
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            
            if stream:
                yield input_ids
        
        if not stream:
            yield input_ids

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = 'cpu', **kwargs):
        from safetensors.torch import load_file
        
        if os.path.isdir(model_path):
            config_path = os.path.join(model_path, 'config.json')
            weights_path = os.path.join(model_path, 'model.safetensors')
            
            config = Config(**kwargs)
            
            if os.path.exists(weights_path):
                state_dict = load_file(weights_path, device=device)
            else:
                weights_path = os.path.join(model_path, 'pytorch_model.bin')
                state_dict = torch.load(weights_path, map_location=device)
        else:
            config = Config(**kwargs)
            state_dict = torch.load(model_path, map_location=device)
        
        model = cls(config)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        return model
    
    def save_pretrained(self, save_path: str):
        from safetensors.torch import save_file
        
        os.makedirs(save_path, exist_ok=True)
        
        state_dict = self.state_dict()
        save_file(state_dict, os.path.join(save_path, 'model.safetensors'))
        
        config_dict = {
            'vocab_size': self.config.vocab_size,
            'hidden_size': self.config.hidden_size,
            'intermediate_size': self.config.intermediate_size,
            'num_layers': self.config.num_layers,
            'num_attention_heads': self.config.num_attention_heads,
            'num_key_value_heads': self.config.num_key_value_heads,
            'max_seq_len': self.config.max_seq_len,
            'expert_num': self.config.expert_num,
            'topk': self.config.topk,
            'mlp_bias': self.config.mlp_bias,
            'use_moe': self.config.use_moe,
            'model_type': 'loftey'
        }
        
        import json
        with open(os.path.join(save_path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Model saved to {save_path}")


class LofteyForCausalLM(LofteyModel):
    def __init__(self, config: Config):
        super().__init__(config)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = super().forward(input_ids, attention_mask)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {'loss': loss, 'logits': logits}
