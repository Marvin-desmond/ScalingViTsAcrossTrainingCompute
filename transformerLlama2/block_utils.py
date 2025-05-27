import torch 
import torch.nn as nn 

class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-5,device="cpu"):
        super(RMSNorm,self).__init__()
        self.weight = nn.Parameter(torch.ones(dim,device=device))
        self.eps=eps 
    
    def forward(self,x):
        means = x.pow(2).mean(-1,keepdims=True)
        means_eps = self.eps + means 
        rms_coeff = torch.rsqrt(means_eps)
        rms_x = x*rms_coeff*self.weight
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

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")

    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_llama(model, CONFIG, params):
    model.token_embeddings.weight = assign(model.token_embeddings.weight, params["tok_embeddings.weight"])
    for l in range(CONFIG.N_LAYERS):
        # Load attention weights
        model.layers[l].attn.Wq.weight = assign(
            model.layers[l].attn.Wq.weight,
            params[f"layers.{l}.attention.wq.weight"]
        )
        model.layers[l].attn.Wk.weight = assign(
            model.layers[l].attn.Wk.weight,
            params[f"layers.{l}.attention.wk.weight"]
        )
        model.layers[l].attn.Wv.weight = assign(
            model.layers[l].attn.Wv.weight,
            params[f"layers.{l}.attention.wv.weight"]
        )
        model.layers[l].attn.Wout.weight = assign(
            model.layers[l].attn.Wout.weight,
            params[f"layers.{l}.attention.wo.weight"]
        )
        model.layers[l].rms_attn.weight = assign(
            model.layers[l].rms_attn.weight,
            params[f"layers.{l}.attention_norm.weight"]
        )

        # Load FeedForward weights
        model.layers[l].ffn.w1.weight = assign(
            model.layers[l].ffn.w1.weight,
            params[f"layers.{l}.feed_forward.w1.weight"]
        )
        # For some reason w2 and w3 are provided in the wrong order in the weights file
        model.layers[l].ffn.v.weight = assign(
            model.layers[l].ffn.v.weight,
            params[f"layers.{l}.feed_forward.w3.weight"]
        )
        model.layers[l].ffn.w2.weight = assign(
            model.layers[l].ffn.w2.weight,
            params[f"layers.{l}.feed_forward.w2.weight"]
        )
        model.layers[l].rms_ffn.weight = assign(
            model.layers[l].rms_ffn.weight,
            params[f"layers.{l}.ffn_norm.weight"]
        )

    # Load output layer weights
    model.norm.weight = assign(model.norm.weight, params["norm.weight"])
    model.output.weight = assign(model.output.weight, params["output.weight"])
