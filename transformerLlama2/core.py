import sys 

import torch, torch.nn as nn 
from block_utils import RMSNorm, FeedForward

sys.path.append("../")
# from mha import MHAandRoPE
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


