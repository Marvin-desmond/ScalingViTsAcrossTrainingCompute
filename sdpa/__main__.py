import numpy as np 

class ScaledDotProductAttention():
    def __init__(self, d_in, d_out, n_heads):
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
        self.Wq = np.random.randn(d_in, d_out)
        self.Wk = np.random.randn(d_in, d_out)
        self.Wv = np.random.randn(d_in, d_out)
        self.Wout = np.random.randn(d_out, d_out)
        
    def __call__(self, x):
        assert len(x.shape) == 3, "batch of embedding sequence expected"

if __name__ == "__main__":
    pass
