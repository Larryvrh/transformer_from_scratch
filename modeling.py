import torch
from torch import nn
from dataclasses import dataclass
from enum import Enum
from typing import *
from flash_attn import flash_attn_func
from flash_attn_triton import flash_attn_func as flash_attn_func_triton
from math import ceil


class AttentionBackend(Enum):
    Naive = 0
    FlashAttentionCuda = 1
    FlashAttentionTriton = 2


global_config = {
    'attn_backend': AttentionBackend.Naive
}


@dataclass
class TransformerConfig:
    vocab_size: int = -1,
    num_layers: int = -1,
    num_heads: int = -1,
    hidden_size: int = -1,
    max_seq_len: int = -1,
    root_model: 'ToyTransformer' = None
    device: torch.device = torch.device('cpu')
    dtype: torch.dtype = torch.float32


def expand_attn_mask(custom_attn_mask: torch.Tensor):
    B, T = custom_attn_mask.shape
    mask = custom_attn_mask.unsqueeze(1).repeat((1, T, 1))
    seq_index_mask = (mask == custom_attn_mask[:, torch.arange(T)].view(B, T, 1))
    return seq_index_mask & (torch.tril(mask) > 0)


# expand attn mask to cu_seqlens for flash attn
def expand_attn_mask_to_seq_lengths(attn_mask: torch.Tensor):
    attn_mask = attn_mask.to('cpu')
    seq_len = attn_mask.shape[0] * attn_mask.shape[1]
    disjoint_point = torch.cat([torch.tensor([[True]] * attn_mask.shape[0]), attn_mask[:, 1:] != attn_mask[:, :-1]], dim=1)
    return torch.cat([torch.nonzero(disjoint_point.view((-1,))), torch.tensor([[seq_len]])]).to(dtype=torch.int32)


# naive RoPE implementation following https://arxiv.org/pdf/2104.09864.pdf
def get_rope_cache_slow(seq_len: int, dim: int, theta: int, device: torch.device, dtype: torch.dtype):
    assert dim % 2 == 0
    freqs = theta ** (-2 * torch.arange(0, dim // 2, 1.) / dim)
    freqs = torch.repeat_interleave(freqs, 2)
    v1 = torch.cos(torch.arange(seq_len, dtype=torch.float).view((seq_len, 1)) * freqs)
    v2 = torch.sin(torch.arange(seq_len, dtype=torch.float).view((seq_len, 1)) * freqs)
    v2 = v2 * torch.tensor([1, -1] * (dim // 2))
    indices = torch.tensor([j for i in range(0, dim, 2) for j in (i + 1, i)])
    return v1.to(device, dtype=dtype), v2.to(device, dtype=dtype), indices.to(device)


def apply_rope_slow(x, rope_cache, positions: Optional[torch.Tensor] = None):
    v1, v2, indices = rope_cache
    seq_len, dim = x.shape[1:]
    if positions is None:
        v1 = v1[:seq_len, :]
        v2 = v2[:seq_len, :]
    else:
        v1 = v1[positions, torch.arange(dim)].view((-1, dim))
        v2 = v2[positions, torch.arange(dim)].view((-1, dim))
    applied_x = x * v1 + (x * v2)[:, :, indices]
    return applied_x


# Optimized RoPE implementation adapted from https://github.com/facebookresearch/llama/blob/main/llama/model.py
def get_rope_cache_fast(seq_len: int, dim: int, theta: int, device: torch.device, dtype: torch.dtype):
    freqs = (1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.to(device)


def apply_rope_fast(x, rope_cache, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    if positions is None and x.shape[1] < rope_cache.shape[0]:
        freqs_cis = rope_cache[:x.shape[1], :]
    elif positions is not None:
        freqs_cis = rope_cache[positions, :]
    else:
        freqs_cis = rope_cache
    freqs_cis = freqs_cis.view([d if i == 1 or i == x_.ndim - 1 else 1 for i, d in enumerate(x_.shape)])

    applied_x = torch.view_as_real(x_ * freqs_cis).flatten(2)
    return applied_x.type_as(x)


# RMSNorm implementation following https://arxiv.org/pdf/1910.07467.pdf
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, dtype, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        x_ = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x_


class AttentionHead(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.head_size = config.hidden_size // config.num_heads
        self.dtype = config.dtype
        self.q_proj = nn.Linear(config.hidden_size, self.head_size, dtype=config.dtype)
        self.k_proj = nn.Linear(config.hidden_size, self.head_size, dtype=config.dtype)
        self.v_proj = nn.Linear(config.hidden_size, self.head_size, dtype=config.dtype)

    def forward(self, x: torch.Tensor, attn_masked_bias: Optional[torch.Tensor],
                kv_cache: Optional[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # if global_config['attn_backend'] == AttentionBackend.FlashAttentionTriton:
        # padding the position indices for alignment
        # positions = torch.tensor([kv_cache[0].shape[1]] * q.shape[1]).to(q.device) if kv_cache is not None else torch.arange(0, x.shape[1], 1).to(q.device)

        positions = torch.tensor([kv_cache[0].shape[1]]).to(q.device) if kv_cache is not None else None
        q = apply_rope_fast(q, self.config.root_model.rope_cache, positions)
        k = apply_rope_fast(k, self.config.root_model.rope_cache, positions)

        if kv_cache is not None:
            k = torch.concat([kv_cache[0], k], dim=1)
            v = torch.concat([kv_cache[1], v], dim=1)

        if global_config['attn_backend'] == AttentionBackend.FlashAttentionCuda:
            q, k, v, = q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2)
            attn_result = flash_attn_func(q, k, v, causal=True)
            q, k, v, attn_result = q.squeeze(2), k.squeeze(2), v.squeeze(2), attn_result.squeeze(2)
        elif global_config['attn_backend'] == AttentionBackend.FlashAttentionTriton:
            q, k, v, = q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2)
            attn_result = flash_attn_func_triton(q, k, v, attn_masked_bias.unsqueeze(1) if attn_masked_bias is not None else None,
                                                 True if kv_cache is None else False)
            q, k, v, attn_result = q.squeeze(2), k.squeeze(2), v.squeeze(2), attn_result.squeeze(2)
        else:
            attn_score = (q @ k.permute(0, 2, 1) / (self.head_size ** 0.5)) + attn_masked_bias
            attn_result = torch.softmax(attn_score, dim=2) @ v

        return attn_result, [k, v]


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.attn_heads = nn.ModuleList([AttentionHead(config) for _ in range(config.num_heads)])
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, dtype=config.dtype)

    def forward(self, x: torch.Tensor, attn_masked_bias: Optional[torch.Tensor],
                kv_cache: Optional[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        head_outputs = [head(x, attn_masked_bias, kv_cache[idx] if kv_cache is not None else None) for idx, head in
                        enumerate(self.attn_heads)]
        return self.o_proj(torch.concat([o[0] for o in head_outputs], dim=2)), [o[1] for o in head_outputs]


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.mha = MultiHeadAttention(config)
        self.up_proj = nn.Linear(config.hidden_size, config.hidden_size * 4, dtype=config.dtype)
        self.down_proj = nn.Linear(config.hidden_size * 4, config.hidden_size, dtype=config.dtype)
        self.ln_mha = nn.LayerNorm(config.hidden_size, dtype=config.dtype)
        self.ln_ffn = nn.LayerNorm(config.hidden_size, dtype=config.dtype)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, attn_masked_bias: Optional[torch.Tensor],
                kv_cache: Optional[List[torch.Tensor]]) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
        mha_output, new_kv_cache = self.mha(self.ln_mha(x), attn_masked_bias, kv_cache)
        mha_output = x + mha_output
        ffn_output = self.down_proj(self.act(self.up_proj(self.ln_ffn(mha_output))))
        return mha_output + ffn_output, new_kv_cache


class ToyTransformer(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, hidden_size: int, max_seq_len: int,
                 device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = TransformerConfig(vocab_size, num_layers, num_heads, hidden_size, max_seq_len, self, device,
                                        dtype)

        self.sem_embed = nn.Embedding(vocab_size, hidden_size, dtype=dtype)

        self.rope_cache = get_rope_cache_fast(max_seq_len, hidden_size // num_heads, 10000, device, dtype)

        self.decoder_layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, dtype=dtype)
        self.to(device)

    def forward(self, seq: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[List[List[torch.Tensor]]]]:
        # sanity checks
        assert attn_mask is None or kv_cache is None  # No support for attn_mask and kv_cache both enabled
        if kv_cache is not None:
            assert seq.shape[0] == 1, 'kv_cache is not supported for batch inference'
        # handle flash-attn triton alignment requirement (actually only needed for backward)
        seq_length = seq.shape[1]
        if kv_cache is None and global_config['attn_backend'] == AttentionBackend.FlashAttentionTriton and seq_length % 128 != 0:
            if attn_mask is None:  # forcibly enable attn_mask due to padding
                attn_mask = torch.ones(seq.shape, device=self.device)
            pad_length = (ceil(seq_length / 128) * 128) - seq_length
            seq = nn.functional.pad(seq, (0, pad_length))
            attn_mask = nn.functional.pad(attn_mask, (0, pad_length))

        # handle attn_bias
        if global_config['attn_backend'] == AttentionBackend.FlashAttentionCuda:
            assert attn_mask is None, 'FlashAttn-Cuda does not support custom attn_mask'
            attn_masked_bias = None
        elif global_config['attn_backend'] == AttentionBackend.FlashAttentionTriton and attn_mask is None:
            attn_masked_bias = None
        elif attn_mask is not None:
            attn_masked_bias = expand_attn_mask(attn_mask)
        elif attn_mask is None and kv_cache is None:
            attn_masked_bias = expand_attn_mask(torch.ones(seq.shape, device=self.device))
        elif kv_cache is not None:
            attn_masked_bias = torch.ones((1, seq.shape[1], seq.shape[1]), dtype=torch.bool, device=self.device)
        else:
            attn_masked_bias = None

        if attn_masked_bias is not None:
            mask_zero = torch.tensor(0, dtype=self.config.dtype)
            mask_val = torch.tensor(torch.finfo(self.config.dtype).min / 2, dtype=self.config.dtype)
            attn_masked_bias = torch.where(attn_masked_bias, mask_zero, mask_val).to(self.device)

        hidden = self.sem_embed(seq)

        new_kv_cache = []
        for idx, decoder in enumerate(self.decoder_layers):
            hidden, layer_kv_cache = decoder(hidden, attn_masked_bias, kv_cache[idx] if kv_cache is not None else None)
            new_kv_cache.append(layer_kv_cache)

        logits = self.lm_head(hidden)

        # remove padding for flash-attn triton
        if kv_cache is None and global_config['attn_backend'] == AttentionBackend.FlashAttentionTriton and seq_length % 128 != 0:
            logits = logits[:, :seq_length, :]
            new_kv_cache = [[[cache[:, :seq_length, :] for cache in head] for head in layer] for layer in new_kv_cache]

        return logits, new_kv_cache

    @property
    def device(self):
        return next(self.parameters()).device
