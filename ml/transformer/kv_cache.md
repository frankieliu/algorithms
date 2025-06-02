<frankie>
Could you show a version with caching for faster inference
</frankie>
Absolutely! Caching is essential for speeding up **autoregressive generation** with Transformers like GPT, especially during inference where we generate tokens one at a time.

Below is a version of the **Transformer decoder** with **key-value caching** for faster inference.

---

## 🧠 Why Cache?

In autoregressive generation:

* You generate one token at a time.
* Without caching, each new token requires recomputing attention over the **entire past sequence**.
* With caching, you **reuse** previously computed key and value projections.

---

## ✅ Transformer Decoder Block with Caching

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, past_kv=None):
        B, T, D = x.shape
        H = self.num_heads
        d = self.head_dim

        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)  # [B, H, T, d]
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)  # [B, H, T, d]
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)  # [B, H, T, d]

        # Append cached keys and values
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # [B, H, T_total, d]
            v = torch.cat([past_v, v], dim=2)

        # Save for next step
        present_kv = (k, v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)  # [B, H, T, T_total]

        # Causal mask: only needed if not generating 1 token at a time
        if T > 1 or k.size(-2) != T:
            causal_mask = torch.tril(torch.ones(T, k.size(-2), device=x.device)).bool()
            scores = scores.masked_fill(~causal_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, T, d]
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        return self.out_proj(out), present_kv

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, past_kv=None):
        attn_out, present_kv = self.attn(self.ln1(x), past_kv)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, present_kv

class TransformerWithCache(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, past_kvs=None):
        B, T = x.size()
        x = self.token_emb(x) + self.pos_emb[:, :T]
        presents = []
        for i, block in enumerate(self.blocks):
            past_kv = None if past_kvs is None else past_kvs[i]
            x, present_kv = block(x, past_kv)
            presents.append(present_kv)
        x = self.ln_f(x)
        return self.head(x), presents  # logits and new cache

```

---

## 🔄 Example: Generating One Token at a Time

```python
# Create model
vocab_size = 1000
seq_len = 1  # one token at a time
model = TransformerWithCache(vocab_size, d_model=128, num_heads=4, d_ff=512, num_layers=2, max_len=1024)

# Generate step-by-step
token = torch.tensor([[1]])  # starting token
past_kvs = None

for _ in range(10):
    logits, past_kvs = model(token, past_kvs)  # one token at a time
    token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B, 1]
    print("Next token:", token.item())
```

---

## ✅ Summary

| Feature               | Supported            |
| --------------------- | -------------------- |
| Key/Value caching     | ✅                    |
| One-token generation  | ✅                    |
| LayerNorm + residuals | ✅                    |
| Causal masking        | ✅ (only when needed) |

---

Let me know if you'd like:

* Multi-token generation with efficient mask handling
* Beam search or sampling strategy integration
* Extension with cross-attention (for encoder-decoder setup)

This is exactly how GPT-style models do efficient token-by-token inference.
