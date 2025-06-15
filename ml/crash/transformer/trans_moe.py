import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V (all in one for efficiency)
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(output)

class Expert(nn.Module):
    """Single expert FFN (used in MoE)"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        
    def forward(self, x):
        #   l1   GeLU   l2
        # x -> z1 -> a1 -> z2         
        #
        # dmodel -> d_ff -> dmodel 
        return self.linear2(self.activation(self.linear1(x)))

class MoELayer(nn.Module):
    """Mixture of Experts (sparse gating)"""
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        # gate is learned from 
        self.gate = nn.Linear(d_model, num_experts)
        
    def forward(self, x):
        # BSD
        batch_size, seq_len, d_model = x.shape
        
        # Gating logic (softmax over experts)
        gate_logits = self.gate(x)  # (batch_size, seq_len, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Top-k experts
        topk_probs, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Dispatch to experts
        for i in range(self.num_experts):
            # Create mask for current expert
            expert_mask = (topk_indices == i).any(dim=-1)
            if expert_mask.sum() == 0:
                continue
            
            # Process inputs with expert[i]
            expert_input = x[expert_mask]
            expert_output = self.experts[i](expert_input)
            
            # Weighted sum
            for k in range(self.top_k):
                output[expert_mask] += (topk_indices[expert_mask] == i)[..., k].float().unsqueeze(-1) * \
                                      topk_probs[expert_mask][..., k].unsqueeze(-1) * expert_output
        
        return output

class TransformerEncoderLayer(nn.Module):
    """Encoder layer: Self-attention + MoE FFN"""
    def __init__(self, d_model, num_heads, d_ff, num_experts=4, top_k=2):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Self-attention
        # BSD
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + attn_output)
        
        # MoE FFN
        moe_output = self.moe(x)
        return self.norm2(x + moe_output)

class TransformerDecoderLayer(nn.Module):
    """Decoder layer: Self-attn + Cross-attn + MoE FFN"""
    def __init__(self, d_model, num_heads, d_ff, num_experts=4, top_k=2):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention
        attn_output = self.self_attn(x, tgt_mask)
        x = self.norm1(x + attn_output)
        
        # Cross-attention
        cross_output = self.cross_attn(x, enc_output, src_mask)
        x = self.norm2(x + cross_output)
        
        # MoE FFN
        moe_output = self.moe(x)
        return self.norm3(x + moe_output)

class Transformer(nn.Module):
    """Full Transformer with Encoder/Decoder and MoE"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 d_ff=2048, num_layers=6, num_experts=4, top_k=2):
        super().__init__()
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)
        
        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, num_experts, top_k)
            for _ in range(num_layers)
        ])
        
        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, num_experts, top_k)
            for _ in range(num_layers)
        ])
        
        self.final_proj = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embeddings
        enc_output = self.encoder_embed(src)
        dec_output = self.decoder_embed(tgt)
        
        # Encoder
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # Decoder
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        return self.final_proj(dec_output)

# Hyperparameters
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
num_experts = 4
top_k = 2

# Initialize model
model = Transformer(
    src_vocab_size=10000, 
    tgt_vocab_size=10000,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    num_experts=num_experts,
    top_k=top_k
)

# Example inputs
src = torch.randint(0, 10000, (32, 20))  # (batch_size, src_seq_len)
tgt = torch.randint(0, 10000, (32, 15))  # (batch_size, tgt_seq_len)

# Forward pass
output = model(src, tgt)
print(output.shape)  # torch.Size([32, 15, 10000])