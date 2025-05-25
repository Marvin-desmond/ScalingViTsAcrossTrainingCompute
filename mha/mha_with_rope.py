import torch 
import torch.nn as nn 
softmax = nn.functional.softmax

import sys 
import os
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from utils import apply_rotary_emb

mha_root_dir = os.path.abspath(os.path.join(current_dir,".."))
sys.path.append(mha_root_dir)
from pos_freqs import precompute_freqs_cis

device = torch.device("cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"))

fc = precompute_freqs_cis(64, 32_000, device=device)

class MHAandRoPE(nn.Module):
    def __init__(self, d_in, d_out, n_heads, context_window, bias=False, device="cpu"):
        super(MHAandRoPE, self).__init__()
        """
        Scaled Dot Product Attention
        ============================
        the core mechanism in the #AttentionIsAllYouNeed paper,
        computes the 
        ==================================================================
        =  Q @ K.T =>|                                                   =
        =            |=> (.) / K.shape[-1] =>|                           =
        =                                    |=> softmax(.) =>|          =
        =                                                     |=>(.) @ V =
        =                                                                =          
        ==================================================================
        such that knowing that the input 
        x => batch * seq_len * d_in
        Q => batch * seq_len * n_heads * head_dim
        K => batch * seq_len * n_heads * head_dim
        V => batch * seq_len * n_heads * head_dim
        
        the dimensions of the steps are
        1. (batch * n_heads * seq_len * head_dim) @ (batch * n_heads * head_dim * seq_len)
           ===> (batch * n_heads * seq_len * seq_len)
           
        2. batch * n_heads * seq_len * seq_len / (1)
           ===> batch * n_heads * seq_len * seq_len
        
        3. op( batch * n_heads * seq_len * seq_len )
           ===> batch * n_heads * seq_len * seq_len

        4. (batch * n_heads * seq_len * seq_len) @ (batch * n_heads * seq_len * head_dim)
           ===> batch * n_heads * seq_len * head_dim
        """
        assert d_out % n_heads == 0, "d_out must be divisible by number of heads"
        self.n_heads = n_heads
        self.d_out = d_out 
        self.head_dim = int(d_out / n_heads) 
        self.device = device
        self.Wq = nn.Linear(d_in, d_out, bias, device=device)
        self.Wk = nn.Linear(d_in, d_out, bias, device=device)
        self.Wv = nn.Linear(d_in, d_out, bias, device=device)
        self.Wout = nn.Linear(d_out, d_out, bias, device=device)
        mask = torch.triu(torch.ones(context_window, context_window,device=device), diagonal=1)
        self.register_buffer("mask", mask)
        
    def __call__(self, x, m_theta_polar_tensor):
        assert len(x.shape) == 3, "batch of embedding sequence expected"
        batch, seq_len, d_in = x.shape
        Q = self.Wq(x) # batch * seq_len * d_out
        K = self.Wk(x)
        V = self.Wv(x)
        # introduce multiple heads
        Q = Q.view(batch, seq_len, self.n_heads, self.head_dim)
        K = K.view(batch, seq_len, self.n_heads, self.head_dim)
        V = V.view(batch, seq_len, self.n_heads, self.head_dim)
        # apply the rope transforms
        # with the m_theta_polar created in the transformer block initializer
        # in the __init__ part of the class
        Q = apply_rotary_emb(Q, m_theta_polar_tensor,self.device)
        K = apply_rotary_emb(K, m_theta_polar_tensor,self.device)
      #   Q, K = apply_rotary_emb(Q, K, m_theta_polar_tensor)
        # make each head across the batches have dim (seq_len,head_dim)
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        # compute step 1
        scores = Q @ K.transpose(-1,-2)
        # compute step 2
        scaled_scores = scores / K.shape[-1]**0.5
        # compute step 3
        masked_scores = scaled_scores.masked_fill_(self.mask.bool()[:seq_len, :seq_len], -torch.inf)
        weights = softmax(masked_scores, dim=-1)
        # compute step 4
        vectors = weights @ V
        vectors = vectors.transpose(1,2)
        vectors = vectors.contiguous().view(batch,seq_len,self.d_out)
        context_vectors = self.Wout(vectors)
        return context_vectors
    

if __name__ == "__main__":
    batch = 8
    d_in = 4096 # 768 for original paper
    d_out = 4096
    n_heads = 32 # 8 for original paper, 12 for GPT 2, 32 for Llama2
    seq_len = 100
    context_window = 4096
   
    m_theta_polar_tensor = precompute_freqs_cis(d_in // n_heads, context_window*2, device=device)
    baseMHA = MHAandRoPE(d_in, d_out, n_heads, context_window, device=device)
    baseMHA = baseMHA.to(device)

    input_tensor = torch.rand(batch, seq_len, d_in, device=device) 
    m_theta_polar_seq = m_theta_polar_tensor[:seq_len]
    out = baseMHA(input_tensor,m_theta_polar_seq)
    print(out.shape)

