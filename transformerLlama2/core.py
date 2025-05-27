import os
import sys 
from enum import Enum, IntEnum 

import torch, torch.nn as nn 
from block_utils import RMSNorm, FeedForward

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir,".."))
sys.path.append(root_dir)

class CONFIG:
    VOCAB: int = 32_000
    CONTEXT_LEN: int = 4096 
    DIM: int = 4096  
    N_HEADS: int = 32
    N_LAYERS: int = 32
    HIDDEN_DIM: int = 11008
    DTYPE: torch.dtype = torch.bfloat16 

from mha.mha_with_rope import MHAandRoPE
from pos_freqs import precompute_freqs_cis

class TransformerBlock(nn.Module):
    def __init__(self,d_in,d_out,n_heads,context_window,device="cpu"):
        super(TransformerBlock,self).__init__()
        self.rms_attn = RMSNorm(d_in,device=device)
        self.attn = MHAandRoPE(d_in,d_out,n_heads,context_window,device=device)
        self.rms_ffn = RMSNorm(d_in,device=device)
        self.ffn = FeedForward(d_in,4*d_in,device=device)
        
    def forward(self, x, m_thetas):
        attn_x = self.rms_attn(x)
        h = self.attn(attn_x, m_thetas) + x 

        ffn_x = self.rms_ffn(h)
        out_x = self.ffn(ffn_x)
        x = out_x + h
        return x

class TransformerLlama2(nn.Module):
    def __init__(self, CONFIG: CONFIG, device="cpu"):
        super(TransformerLlama2,self).__init__()
        self.token_embeddings = nn.Embedding(
            CONFIG.VOCAB, CONFIG.DIM,
            device=device
        )
        self.layers = nn.ModuleList()
        for _ in range(CONFIG.N_LAYERS):
            self.layers.append(
                TransformerBlock(
                    CONFIG.DIM,
                    CONFIG.DIM,
                    CONFIG.N_HEADS,
                    CONFIG.CONTEXT_LEN, device=device)
            )
        self.norm = RMSNorm(CONFIG.DIM, device=device)
        self.output = nn.Linear(CONFIG.DIM,CONFIG.VOCAB,bias=False, device=device)
        self.m_thetas = precompute_freqs_cis(CONFIG.DIM//CONFIG.N_HEADS,CONFIG.CONTEXT_LEN*2,device=device)
    def forward(self,x):
        batch, seq_len = x.shape
        x = self.token_embeddings(x)
        m_thetas_seq = self.m_thetas[:seq_len]
        for layer in self.layers:
            x = layer(x, m_thetas_seq)
        x = self.norm(x)
        x = self.output(x).float()
        return x 

if __name__ == "__main__":
    n_heads=32
    d_in=4096 
    d_out=4096
    context_window=4096
    B,S,D=10,50,4096
    x=torch.randn(B,S,D,device="mps")

    m_thetas = precompute_freqs_cis(d_in//n_heads,context_window*2,device="mps")
    transformerBlock = TransformerBlock(d_in,d_out,n_heads,context_window,device="mps")
    m_thetas_seq=m_thetas[:S]
    print(transformerBlock(x,m_thetas_seq).shape)



