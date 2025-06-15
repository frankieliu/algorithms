import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# ====================== Rotary Positional Embeddings ======================
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
        # Build cache
        self.build_cache(max_seq_len)
    
    def build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cache', emb.cos())
        self.register_buffer('sin_cache', emb.sin())
    
    def forward(self, seq_len):
        if seq_len > self.max_seq_len:
            self.build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.view(1, 1, *cos.shape)
    sin = sin.view(1, 1, *sin.shape)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ====================== Mixture of Experts ======================
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.d_ff = d_ff
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Gating logic
        gate_logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Dispatch to experts
        for i in range(self.num_experts):
            expert_mask = (topk_indices == i).any(dim=-1)
            if expert_mask.sum() == 0:
                continue
                
            expert_input = x[expert_mask]
            expert_output = self.experts[i](expert_input)
            
            # Weighted sum
            for k in range(self.top_k):
                output[expert_mask] += (topk_indices[expert_mask] == i)[..., k].float().unsqueeze(-1) * \
                                     topk_probs[expert_mask][..., k].unsqueeze(-1) * expert_output
        
        return output

# ====================== Attention Layers ======================

# Not correct implementation
# Since you need to add the cross attention bit
class MultiHeadAttention1(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        
    def forward(self, x, mask=None, causal=False):
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply Rotary Positional Embeddings
        cos, sin = self.rotary_emb(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.head_dim)
        
        # Apply masks
        if causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(output)

# ====================== Transformer Layers ======================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts=4, top_k=2):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + attn_output)
        moe_output = self.moe(x)
        return self.norm2(x + moe_output)

# This decoder layer has moe but cross function is not
# implemented correctly
class DecoderLayer1(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts=4, top_k=2):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention with causal mask
        attn_output = self.self_attn(x, tgt_mask, causal=True)
        x = self.norm1(x + attn_output)
        
        # Cross-attention
        cross_output = self.cross_attn(x, enc_output, src_mask)
        x = self.norm2(x + cross_output)
        
        # MoE FFN
        moe_output = self.moe(x)
        return self.norm3(x + moe_output)

# ====================== Full Transformer ======================
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 d_ff=2048, num_layers=6, num_experts=4, top_k=2):
        super().__init__()
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)
        
        # Encoder and Decoder stacks
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, num_experts, top_k)
            for _ in range(num_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, num_experts, top_k)
            for _ in range(num_layers)
        ])
        
        self.final_proj = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embeddings
        enc_output = self.encoder_embed(src)
        dec_output = self.decoder_embed(tgt)
        
        # Encoder
        for layer in self.encoder:
            enc_output = layer(enc_output, src_mask)
        
        # Decoder (with causal masking)
        for layer in self.decoder:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        return self.final_proj(dec_output)

# ====================== Usage Example ======================
if __name__ == "__main__":
    model = Transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        num_experts=4,
        top_k=2
    )
    
    src = torch.randint(0, 10000, (32, 20))  # (batch_size, src_seq_len)
    tgt = torch.randint(0, 10000, (32, 15))  # (batch_size, tgt_seq_len)
    
    # Causal mask is applied automatically in decoder
    output = model(src, tgt)
    print(output.shape)  # torch.Size([32, 15, 10000])

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts=4, top_k=2):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention with causal mask
        attn_output = self.self_attn(x, mask=tgt_mask, causal=True)
        x = self.norm1(x + attn_output)
        
        # Cross-attention: x as q, enc_output as k/v
        cross_output = self.cross_attn(
            q=x,  # Decoder states as queries
            k=enc_output,  # Encoder outputs as keys
            v=enc_output,  # Encoder outputs as values
            mask=src_mask  # Optional source mask
        )
        x = self.norm2(x + cross_output)
        
        # MoE FFN
        moe_output = self.moe(x)
        return self.norm3(x + moe_output)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        
    def forward(self, q, k=None, v=None, mask=None, causal=False):
        if k is None and v is None:
            # Self-attention case
            k = v = q
        elif v is None:
            # Cross-attention case where k and v come from encoder
            v = k
            
        batch_size = q.size(0)
        
        # Project queries, keys, values
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply Rotary Positional Embeddings to queries and keys
        seq_len = q.size(2)
        cos, sin = self.rotary_emb(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.head_dim)
        
        # Apply masks
        if causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(output)