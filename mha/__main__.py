import numpy as np 
from utils import softmax 

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
        import numpy.ma as ma 
        assert len(x.shape) == 3, "batch of embedding sequence expected"
        batch, seq_len, d_in = x.shape
        Q = x @ self.Wq # batch * seq_len * d_out
        K = x @ self.Wk
        V = x @ self.Wv
        # introduce multiple heads
        Q = Q.reshape((batch, seq_len, self.n_heads, self.head_dim))
        K = K.reshape((batch, seq_len, self.n_heads, self.head_dim))
        V = V.reshape((batch, seq_len, self.n_heads, self.head_dim))
        # make each head across the batches have dim (seq_len,head_dim)
        Q = Q.transpose((0,2,1,3))
        K = K.transpose((0,2,1,3))
        V = V.transpose((0,2,1,3))
        # compute step 1
        scores = Q @ K.transpose((0,1,-1,-2))
        # compute step 2
        scaled_scores = scores / K.shape[-1]**0.5
        # compute step 3
        mask = np.triu(np.ones_like(scores),1)
        masked_scores = ma.array(scaled_scores, mask=mask).filled(fill_value=-np.inf)
        weights = softmax(masked_scores)
        # compute step 4
        vectors = weights @ V
        vectors = vectors.transpose((0,2,1,3))
        vectors = vectors.reshape((batch,seq_len,self.d_out))
        context_vectors = vectors @ self.Wout
        return context_vectors        

if __name__ == "__main__":
    batch = 9
    d_in = 768
    d_out = 768
    n_heads = 12
    seq_len = 10
    baseSdpa = ScaledDotProductAttention(d_in, d_out, n_heads)
    out = baseSdpa(np.random.randn(batch, seq_len, d_in))
    print(out.shape)

