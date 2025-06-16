import torch 
import torch.nn as nn 

class ImageEmbeddings(nn.Module):
    def __init__(self, H, W, P, dim):
        super(ImageEmbeddings,self).__init__()
        N = int((H*W)/(P**2)); self.N = N
        assert H%P==0 and W%P==0, \
          "image size must be integer multiple of patch"
        self.conv_then_project = nn.Conv2d(3,out_channels=dim,kernel_size=P,stride=P)
        self.class_tokens = nn.Parameter(torch.randn(1, 1, dim))
        self.position_embeddings = nn.Parameter(torch.randn(1,N+1,dim))
    def forward(self,image):
        x = self.conv_then_project(image)
        x = x.flatten(2)
        x = x.transpose(1,2)
        x = torch.cat((self.class_tokens,x),dim=1)
        x += self.position_embeddings[:,:(self.N+1)]
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,n_heads,bias=True):
        super(MultiHeadAttention,self).__init__()
        assert d_out%n_heads==0, "d_in must be divisible by n_heads"
        self.head_dim = int(d_out / n_heads)
        self.n_heads = n_heads
        self.dropout = 0.0
        self.Wq = nn.Linear(d_in,d_out,bias=bias)
        self.Wk = nn.Linear(d_in,d_out,bias=bias)
        self.Wv = nn.Linear(d_in,d_out,bias=bias)
        self.Wo = nn.Linear(d_out,d_out,bias=bias)
        self.attn_dropout=nn.Dropout(self.dropout)
        self.res_dropout=nn.Dropout(self.dropout)

    def forward(self,x):
        batch,seq_len,d_in = x.shape
        q = self.Wq(x) # (batch,seqlen,d_out)
        k = self.Wk(x) # (batch,seqlen,d_out)
        v = self.Wv(x) # (batch,seqlen,d_out)
        q = q.view(batch,seq_len,self.n_heads,self.head_dim) # (batch,seqlen,n_heads,head_dim)
        k = k.view(batch,seq_len,self.n_heads,self.head_dim) # (batch,seqlen,n_heads,head_dim)
        v = v.view(batch,seq_len,self.n_heads,self.head_dim) # (batch,seqlen,n_heads,head_dim)
        q = q.transpose(1,2) # (batch,n_heads,seqlen,head_dim)
        k = k.transpose(1,2) # (batch,n_heads,seqlen,head_dim)
        v = v.transpose(1,2) # (batch,n_heads,seqlen,head_dim)
        scores = q @ k.transpose(-1,-2) # (batch,n_heads,seqlen,head_dim) x (batch,n_heads,head_dim,seqlen) => (batch,n_heads,seqlen,seqlen)
        scores = scores / k.shape[-1]**.5
        norm_scores = nn.functional.softmax(scores,dim=-1)
        # norm_scores = self.attn_dropout(norm_scores)
        y = norm_scores @ v # (batch,n_heads,seqlen,seqlen) x (batch,n_heads,seqlen,head_dim) => (batch,n_heads,seqlen,head_dim)
        out = y.transpose(1,2).contiguous().view(batch,seq_len,d_in)
        out = self.Wo(out)
        # out = self.res_dropout(out)
        return out 

class MLP(nn.Module):
    def __init__(self,dim,hidden_dim,bias=True):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(dim,hidden_dim,bias=bias)
        self.fc2 = nn.Linear(hidden_dim,dim,bias=bias)
        self.act = nn.GELU(approximate='tanh')
    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(self.act(x))
        return x

class ViTBlock(nn.Module):
    def __init__(self,CONFIG):
        super(ViTBlock,self).__init__()
        self.ln1 = nn.LayerNorm(CONFIG.D_IN)
        self.attn = MultiHeadAttention(CONFIG.D_IN,CONFIG.D_OUT,CONFIG.HEADS)
        self.ln2 = nn.LayerNorm(CONFIG.D_IN)
        self.mlp = MLP(CONFIG.D_IN,CONFIG.HIDDEN_DIM)
    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x 
    
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        return torch.nn.Parameter(torch.tensor(right))

def scatter_weights(left,right,mcs,part):
    if part == 0:
        return assign(left,right[:mcs//3,:])
    elif part == 1:
        return assign(left,right[mcs//3:2*mcs//3,:])
    elif part == 2:
        return assign(left,right[-mcs//3:,:])

def scatter_biases(left,right,mcs,part):
    if part == 0:
        return assign(left,right[:mcs//3])
    elif part == 1:
        return assign(left,right[mcs//3:2*mcs//3])
    elif part == 2:
        return assign(left,right[-mcs//3:])
    
def copy_model_weights(model, params):
    model.image_embeddings.conv_then_project.weight = assign(
        model.image_embeddings.conv_then_project.weight,
        params['conv_proj.weight'])
    model.image_embeddings.conv_then_project.bias = assign(
        model.image_embeddings.conv_then_project.bias,
        params['conv_proj.bias'])
    model.image_embeddings.position_embeddings = assign(
        model.image_embeddings.position_embeddings,
        params['encoder.pos_embedding'])
    model.image_embeddings.class_tokens = assign(
        model.image_embeddings.class_tokens,
        params['class_token'])
    for i in range(len(model.vit_blocks)):
        model.vit_blocks[i].ln1.weight = assign(
                                model.vit_blocks[i].ln1.weight,
                                params[f'encoder.layers.encoder_layer_{i}.ln_1.weight'])
        model.vit_blocks[i].ln1.bias = assign(
                                model.vit_blocks[i].ln1.bias,
                                params[f'encoder.layers.encoder_layer_{i}.ln_1.bias'])
        w_name = f'encoder.layers.encoder_layer_{i}.self_attention.in_proj_weight'
        b_name = f'encoder.layers.encoder_layer_{i}.self_attention.in_proj_bias'
        mcs = params[w_name].shape[0]
        model.vit_blocks[i].attn.Wq.weight = scatter_weights(model.vit_blocks[i].attn.Wq.weight,
                        params[w_name],mcs,0)
        model.vit_blocks[i].attn.Wq.bias = scatter_biases(model.vit_blocks[i].attn.Wq.bias,
                        params[b_name],mcs,0)
        model.vit_blocks[i].attn.Wk.weight = scatter_weights(model.vit_blocks[i].attn.Wk.weight,
                        params[w_name],mcs,1)
        model.vit_blocks[i].attn.Wk.bias = scatter_biases(model.vit_blocks[i].attn.Wk.bias,
                        params[b_name],mcs,1)
        model.vit_blocks[i].attn.Wv.weight = scatter_weights(model.vit_blocks[i].attn.Wv.weight,
                        params[w_name],mcs,2)
        model.vit_blocks[i].attn.Wv.bias = scatter_biases(model.vit_blocks[i].attn.Wv.bias,
                        params[b_name],mcs,2)
        model.vit_blocks[i].attn.Wo.weight = assign(
                                model.vit_blocks[i].attn.Wo.weight,
                                params[f'encoder.layers.encoder_layer_{i}.self_attention.out_proj.weight']
                                )
        model.vit_blocks[i].attn.Wo.bias = assign(
                                model.vit_blocks[i].attn.Wo.bias,
                                params[f'encoder.layers.encoder_layer_{i}.self_attention.out_proj.bias'])
        model.vit_blocks[i].ln2.weight = assign(
                                model.vit_blocks[i].ln2.weight,
                                params[f'encoder.layers.encoder_layer_{i}.ln_2.weight'])
        model.vit_blocks[i].ln2.bias = assign(
                                model.vit_blocks[i].ln2.bias,
                                params[f'encoder.layers.encoder_layer_{i}.ln_2.bias'])
        model.vit_blocks[i].mlp.fc1.weight = assign(
                                model.vit_blocks[i].mlp.fc1.weight,
                                params[f'encoder.layers.encoder_layer_{i}.mlp.0.weight'])
        model.vit_blocks[i].mlp.fc1.bias = assign(
                                model.vit_blocks[i].mlp.fc1.bias,
                                params[f'encoder.layers.encoder_layer_{i}.mlp.0.bias'])
        model.vit_blocks[i].mlp.fc2.weight = assign(
                                model.vit_blocks[i].mlp.fc2.weight,
                                params[f'encoder.layers.encoder_layer_{i}.mlp.3.weight'])
        model.vit_blocks[i].mlp.fc2.bias = assign(
                                model.vit_blocks[i].mlp.fc2.bias,
                                params[f'encoder.layers.encoder_layer_{i}.mlp.3.bias'])
    model.norm.weight = assign(model.norm.weight,params[f'encoder.ln.weight'])
    model.norm.bias = assign(model.norm.bias,params[f'encoder.ln.bias'])
    model.out_linear.weight = assign(model.out_linear.weight,
                                        params['heads.0.weight'])
    model.out_linear.bias = assign(model.out_linear.bias,
                                    params[f'heads.0.bias'])