import torch 
import torch.nn as nn 

class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-5,device="cpu"):
        super(RMSNorm,self).__init__()
        self.weights = nn.Parameter(torch.ones(dim,device=device))
        self.eps=eps 
    
    def forward(self,x):
        means = x.pow(2).mean(-1,keepdims=True)
        means_eps = self.eps + means 
        rms_coeff = torch.rsqrt(means_eps)
        rms_x = x*rms_coeff*self.weights
        return rms_x 
    
class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,multiple_of=256,bias=False,device="cpu"):
        super(FeedForward,self).__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim,hidden_dim,bias=bias,device=device)
        self.v = nn.Linear(dim,hidden_dim,bias=bias,device=device)
        self.w2 = nn.Linear(hidden_dim,dim,bias=bias,device=device)
        self.SiLU = nn.SiLU()
    def forward(self,x):
        x = (self.SiLU(self.w1(x))*self.v(x))
        x = self.w2(x)
        return x 

