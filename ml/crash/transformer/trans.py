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
        #
        # values, indices = topk(input, k, dim) 
        # values: input (b,s,e) : output (b,s,k)
        #         => dim will shrink to k
        # indices: input (b,s,e) : output (b,s,k)!
        #         => for each k will give index of its location in the original tensor
        #         => for each b,s give k indices 
        topk_probs, topk_indices = torch.topk(g_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # output will be size of x (b,s,d)
        output = torch.zeros_like(x)

        for i in range(self.num_experts):
            # any returns BSK -> BS
            # Boolean matrix for each b,s
            # say whether one of the top indices == i 
            # e.g. there may be 10 experts,
            # and for k=3, only three of the
            # experts will be choosen 
            expert_mask = (topk_indices == i).any(dim=-1)  # size b,s
            if expert_mask.sum() == 0:
                # expert isn't used anywhere
                continue 
            
            # choose a subset of x where b,s is True
            # x : BSD
            # x[mask] : bsD  (Here small b and s represents the subset)
            expert_input = x[expert_mask]    # sb,ss,d  (subset of b,s where this expert acts)

            # output from each expert
            expert_output = self.experts[i](expert_input)  # sb,ss,d

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
                output[expert_mask] += (
                    (topk_indices[expert_mask] == i)[...,k].float().unsqueeze(-1) *
                    topk_probs[expert_mask][..., k].unsqueeze(-1) * expert_output
                )
                # for k in range(self.k):
                #   topk_indices[expert_mask] == i 
                #   b,s: [ 3 4 9 ] == 9
                #   b,s: [ False False True ]
                #   [..., k] -> pick the kth one
                #   unsqueeze: b,s -> b,s,1
                #   topk_probs[expert_mask][...,k] -> pick the kth one
                #  
                # This basically says for each expert i:
                # if it matches the topk expert index then it will be added:
                # 
                # As in the example above
                # b,s: [ 3 4 9 ], we will be checking against all possible expert index
                # that might overlap with one of the topk_indices, if it does then
                # use topk_values b,s: [ 0.2 0.1 0.7 ]
                #   
                # expert_output  sb,ss,d
        return output    
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts, top_k):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.moe = MoE(d_model, d_ff, num_experts, top_k)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + attn_output)
        moe_output = self.moe(x)
        return self.norm2(x + moe_output)

""" Not optimized """

class DeepSeekMoe(nn.Module):
    def moe(self, x, topk_indices, topk_weights, num_experts):
        """
        topk_indices |= b*s_i : [e_idx1, e_idx2, e_idx3]
        topk_weights |= b*s_i : [e_wgt1, e_wgt2, e_wgt3]               
        
        Note: that b*s is for all tokens (batch * sequence)
        For each you get 2*k number:
        - k numbers for the index of the topk experts
        - k numbers for the weights for each of the topk experts
        """
        
        # x   |=  b*s,d
        # ind |=  b*s,k
        # expert_mask (line 1) |= b*s,k,e (one-hot)
        # expert_mask (line 2) |= e,b*s,k
        # e
        #   b*s
        #       [1 0 0] -> this means expert e, for token b*s, is the topk #1
        # e
        #   b*s
        #       [0 0 0] -> this will NOT show up because topk_indices already
        #                  filters b*s, i.e. so one of these slots will be 
        #                  filled by some expert
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=num_experts) 
        expert_mask = expert_mask.permute(2,0,1)
        """
        export_mask is ordered by:
        - each expert
        
        Some experts will be empty:
        - one-hot encoding guarantees that each expert receives
        receives for all b*s, [[one-hot] [one-hot] [one-hot]]  (line 1, k=3)
        
        - after transpose (line 2, k=3)
        for each expert
         for each b*s token
          [True, False, False]  

          ...[Falee, False, False]  - happens often if using 0 experts out of many
          - this makes one-hot encoding inefficient 
        """

        final = torch.zeros_like(x)
        for expert_idx in range(num_experts):

            # pick out info a particular expert idx 
            mask = expert_mask[expert_idx]    # b*s,k

            # where returns a tuple of dim1, dim2, ..., dim where 
            # tensor is true
            token_indices, weight_indices = torch.where(mask)

            # length of token_indices corresponds which b*s are True
            if token_indices.numel() > 0:

                # topk_weights: b*s,k
                # token_indices picks out the rows (b*s) for a particular
                # expert, and weight_indices picks out which of the
                # topk experts it belongs to
                expert_weights = topk_weights[token_indices, weight_indices]
                # b*s, d -> filter(b*s),d
                expert_input = x[token_indices]
                expert_output = self.expert[expert_idx](expert_input)
                
                # unqueeze is necessary for broadcasting
                # - expert output should be filterd(b*s),d
                # - expert>weights is filtered(b*s)
                #   - unsqueeze to filtered(b*s),1
                weighted_output = expert_output * expert_weights.unsqueeze(-1)

                # Note:
                # final[token_indices] += weighted_output
                # - each filtered(b*s) has contribution from only one expert
                # 
                # .index_add_ is good if you have token_indices
                # - has repeat indices
                final.index_add_(0, token_indices, weighted_output)

""" DeepseekV3 has a different type of router"""
class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts)))

    @torch.no_grad()
    def get_topk_indices(self, scores):
        """

        This introduces the concept of a group of experts
        - the sum of the scores in an expert group
        - determine which of the groups will be chosen
        - all other groups' score are set to zero

        """
        # scores: b*s,e
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)

        # grouping experts together
        # say n_group == 20 and n_routed_experts == 100
        # then you would have b*s, 100 -> b*s, 20, 5
        # then find the topk and sum   -> b*s, 20
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        # find the topk groups
        # group_idx: index in dim for the topk_group
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        # group_scores: b*s, n_group
        # group_mask:   b*s, n_group
        # put a 1 for the topk_group
        group_mask.scatter_(1, group_idx, 1)

        # expand the mask to all 5 experts
        # b*s: [1 0 0 1 0 0 0...0]  # 20 groups (topk_group = 2)
        # step 1: [[1] [0] [0] [1] [0] ... ]  # unsqueeze
        # step 2: [[1 1 1 1 1] [0 0 0 0 0] [0 0 0 0 0] [1 1 1 1 1] ... ]  # expand
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )

        # Zero out the scores for groups that were not chosen
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        
        # The "normal" topk from the scores
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        # flatten to b*s,d
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        # b*s,d e,d
        # torch.nn.functional.linear() 
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))

        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

class OlmoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([OlmoeMLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # Flatten b*s,d
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        # takes the softmax and then normalizes for the ones in the topk
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be selected
        # selected_experts : b*s,k 
        # expert_mask: b*s,k -> b*s,k,e -> e,k,b*s  *** here the permute is (2,1,0) ***
        # This is different than DeepSeek's  e,b*s,k
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # expert_mask[expert_idx] = b*s,k
            # you get 
            # idx:   which of the k experts
            # top_x: which of the b*s
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Seems simpler to:
            #  current_state = hidden_states[top_x]
            #  current_hidden_states = expert_layer(current_state) * routing_weights[topx, idx, None]
            #
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # First argument to index_add_ is the dimension
            # So selecting dim=0 b*s dimension
            # for the b*s in top_x
            # current_hidden_state : subset(b*s), d
            #
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

@use_kernel_forward_from_hub("Llama4TextMoe")
class Llama4TextMoe(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = Llama4TextExperts(config)

        # this much simpler, set bias=False
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.shared_expert = Llama4TextMLP(config)

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)

        # router_logits:     b*s,e
        # scatter on e (dim = 1)
        # router_indices:    b*s,k
        # router_top_value:  b*s,k 
        # router_scores:     b*s,e
        # - -inf everywhere
        # for each b*s:
        # - recover only the top k scores
        router_scores = (
            torch.full_like(router_logits, float("-inf")).scatter_(1, router_indices, router_top_value).transpose(0, 1)
        )
        # Note: -inf becomes sigmoid(-inf) = 0
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        # router_in
        # b*s,d -> b*s*e, d
        routed_in = hidden_states.repeat(self.num_experts, 1)
        # b*s*e,d * b*s*e,1 (from reshape)
        # Note only the enabled/gated states will be retained
        routed_in = routed_in * router_scores.reshape(-1, 1)
        # apply the experts all at once, here we don't
        # loop over the experts, we gate the input
        routed_out = self.experts(routed_in)

        out = self.shared_expert(hidden_states)

        # reshape:
        # b*s*e,d
        # e,b*s,d .sum(dim=0) : sum contributes from the top-k
        out.add_(routed_out.reshape(self.num_experts, -1, self.hidden_dim).sum(dim=0))

        return out, router_scores
