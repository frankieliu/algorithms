import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, positions):
        """Generate rotary embeddings for given positions"""
        # positions: [seq_len]
        # seq_len = positions.shape[0]
        t = positions.float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # [seq_len, dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        return emb

def apply_rotary_pos_emb(x, sin_emb, cos_emb):
    """Apply rotary embeddings to input tensor"""
    # x: [batch, heads, seq_len, dim]
    # sin_emb/cos_emb: [seq_len, dim]
    sin_emb = sin_emb.view(1, 1, *sin_emb.shape) # unsqueeze
    cos_emb = cos_emb.view(1, 1, *cos_emb.shape)
    # each dim of x gets multiplied by cos(f_d t), t: position in the seq
    return (x * cos_emb) + (rotate_half(x) * sin_emb)

def rotate_half(x):
    """Rotate half the dimensions"""
    # x: BHSD
    # break up X into two chunks along D
    # a single x:  a0  a1  a2  a3 a4 a5 a6 a7
    #             -a4 -a5 -a6 -a7 a0 a1 a2 a3
    # a0 * c - a4 * s
    # a0 * s + a4 * c
    # so combining a0 and a4 
    # 
    # a0 = [c  -s]  a0
    # a4   [s   c]  a4 
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply Rotary Positional Embeddings to Q and K
        positions = torch.arange(seq_len, device=x.device)
        sin_emb, cos_emb = self.get_rotary_embeddings(positions)
        # S x D   S x D
        q = apply_rotary_pos_emb(q, sin_emb, cos_emb)
        k = apply_rotary_pos_emb(k, sin_emb, cos_emb)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(output)
    
    def get_rotary_embeddings(self, positions):
        """Generate sin/cos embeddings for given positions"""
        emb = self.rotary_emb(positions)  # [S x D]
        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        return sin_emb, cos_emb

class TransformerEncoderLayer(nn.Module):
    def __init__(self, **args):
        pass

class TransformerDecoderLayer(nn.Module):
    def __init__(self, **args):
        pass
# (Keep the Expert, MoELayer, TransformerEncoderLayer, TransformerDecoderLayer, and Transformer classes 
# from the previous implementation, but replace MultiHeadAttention with MultiHeadAttentionWithRoPE)

class TransformerWithRoPE(nn.Module):
    """Full Transformer with Rotary Positional Embeddings"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 d_ff=2048, num_layers=6, num_experts=4, top_k=2):
        super().__init__()
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)
        
        # Encoder stack (using RoPE attention)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, num_experts, top_k)
            for _ in range(num_layers)
        ])
        
        # Decoder stack (using RoPE attention)
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


# Initialize model
model = TransformerWithRoPE(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    num_experts=4,
    top_k=2
)

# Example inputs
src = torch.randint(0, 10000, (32, 20))  # (batch_size, src_seq_len)
tgt = torch.randint(0, 10000, (32, 15))  # (batch_size, tgt_seq_len)

# Forward pass
output = model(src, tgt)
print(output.shape)  # torch.Size([32, 15, 10000])