import torch
import torch.functional as F
from torch import nn
from math import sqrt, log

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model,
                 num_heads,
                 d_ff,
                 num_experts,
                 top_k,
                 num_layers,
                 ):
        super().__init__()
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)

        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, num_experts, top_k)
            for _ in range(num_layers)
        ]) 

        self.decoder = nn.ModuleList([
            Decoder(d_model, num_heads, d_ff, num_experts, top_k)
            for _ in range(num_layers)
        ])

        self.final_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder_embed(src) 
        dec_output = self.decoder_embed(tgt)
        
        for layer in self.encoder:
            enc_output = layer(enc_output, src_mask)
        
        for layer in self.decoder:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
            
        return self.final_proj(dec_output)

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts=4, top_k=2):
        super().__init__()
        self.self_atten = MultiHeadAttention(d_model, num_heads)
        self.cross_atten = MultiHeadAttention(d_model, num_heads)
        self.moe = MoE(d_model, d_ff, num_experts, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        # https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/normalization.py
        # normalizes over the last dimension
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output = self.self_attn(x, tgt_mask, causal=True)
        x = self.norm1(x + attn_output)

        cross_output = self.cross_attn(x, enc_output, src_mask)
        x = self.norm2(x + cross_output)

        moe_output = self.moe(x)
        return self.norm3(x + moe_output)

class MultiHeadAttention(nn.Module):
    def __init__(self, slen, d_model, num_heads):
        self.slen = slen
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.rope = Rope(slen, self.head_dim)

    def forward(self, x, xcross = None, causal = False):
        # Use upper case for the dimension
        # use lower case for the size of the dimension 
        b, s, d = x.shape
        nh, hd = self.num_heads, self.head_dim
        if xcross is None:
            xcross = x
        # project x in to qkv space
        # x : BSD
        # q : BHSd
        # view:
        # BSD -> BSHd
        # transpose:
        # BSHd -> BHSd
        q = self.q_proj(x).view(b, s, nh, hd).transpose(1,2)
        k = self.k_proj(xcross).view(b, s, nh, hd).transpose(1,2)
        v = self.v_proj(xcross).view(b, s, nh, hd).transpose(1,2)

        # BHSd : applying R on last dimenion       
        q = self.rope(q)
        k = self.rope(k)
        # BHSS
        # First S is the index of the query token
        # Second S is the index of the key token
        scores = torch.einsum('bhsd,bhtd -> bhst', q, k)/sqrt(hd)

        # apply causal mask
        if causal:
            causal_mask = torch.triu(
                torch.ones(self.slen, self.slen), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # scores: BHSS
        # v 
        # The softmax is over the key tokens for a particular query
        # weights: BHSS  First S is from Q and S from K
        # V:       BHSd  
        # So we want to sum over the weights
        # einsum(bhst,bhtd -> bhsd)
        # that is for each S(q) you weight the add the
        # proportionate amount of V at S(k)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bhst, bhtd -> bhsd',
                              attn_weights, v)

        # join all the heads together
        # output : BHSd -> BSHd
        # final view: BSHd -> BSD
        output.transpose(1,2).contiguous().view(b,s,-1)
        return self.o_proj(output)


class Rope(nn.Module):
    def __init__(self, slen, dim):
        super().__init()
        self.slen = slen
        self.dim = dim

        # : dim/2
        inv_freq = 1.0/(10000 ** (torch.arange(0, dim, 2).float()/dim))
        self.register_buffer('inv_freq', inv_freq)

        # : slen
        t = torch.arange(slen).float()
        # : slen, dim/2  
        angles = torch.einsum('i,j->ij', t, self.inv_freq)
        # : slen, dim
        angles = torch.cat((angles, angles), dim=-1)
        self.register_buffer("cos_cache", torch.cos(angles))
        self.register_buffer("sin_cache", torch.sin(angles))

    def rotate90(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        """
        x0 c1 - x4 s1
        x1 c2 - x5 s2
        x2 c3 - x6 s3
        x3 c4 - x7 s4
        x4 c1 + x0 s1
        x5 c2 + x1 s2
        x6 c3 + x2 s3
        x7 c4 + x3 s4

        so you get
        [c1 -s1] x0
        [s1  c1] x4
        """
        cos = self.cos_cache.view(1, 1, *self.cos_cache.shape)
        sin = self.sin_cache.view(1, 1, *self.sin_cache.shape)
        return x * cos + self.rotate90(x) * sin

class MoE(nn.Module):
    def __init__(self, d_model, num_experts, top_k):
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k
    def forward(self, x):
        b,s,d = x.shape

        # gate
        # BSD -> BSE (experts)
        g_logits = self.gate(x)
        g_probs = F.softmax(g_logits, dim=-1)

        # e.g.
        # num_experts = 10, k = 3
        # 1,2,3,4..,10      3,6,7
        # BSE -> BSK
        # for each b,s : [3, 6, 7]
        #                [0.3, 0.2, 0.5]
        topk_probs, topk_indices = torch.topk(g_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x)

        for i in range(self.num_experts):
            # any returns BSK -> BS
            # Boolean matrix for each b,s
            # say whether one of the top indices == i 
            # e.g. there may be 10 experts,
            # and for k=3, only three of the
            # experts will be choosen 
            expert_mask = (topk_indices == i).any(dim=-1)
            if expert_mask.sum() == 0:
                # expert isn't used anywhere
                continue 
            
            # choose a subset of x where b,s is True
            # x : BSD
            # x[mask] : bsD  (Here small b and s represents the subset)
            expert_input = x[expert_mask]

            # all the experts are equivalent

            expert_output = self.experts[i](expert_input)

            # sum contribution
            for k in range(self.top_k):
                # look only at the subset of top_k_indices
                # why do I need topk indicies again?
                # for the current expert i
                # expert_mask picks out the b,s which uses
                # this expert
                # but now we need to multiply this expert
                # by the prob weight for this expert, i.e.
                # the top-k experts are weighed by a softmax
                # topk_indices[expert_mask] == i
                # gives boolean for the [3, 6, 7] == 6
                # and [...,k] picks one of the top_k
                # for in this case [False, True, False]
                # so you pick one at at time
                # first line gives:
                # bsk
                # where the size of bs is given by the expert_mask
                # also pick the probabity for the same
                # 
                # only add the expert output
                # seems it might be easier to generate
                # bs for the probability
                # i = 6
                # topk_indices[e
                output[expert_mask] += (
                    (topk_indices[expert_mask] == i)[...,k].float().unsqeeze(-1) *
                    topk_probs[expert_mask][..., k].unsqueeze(-1) * expert_output
                )
                 