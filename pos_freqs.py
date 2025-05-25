import torch 
from typing import Tuple 

def llama_precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def llama_reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    print(f"{x.shape = } {freqs_cis.shape = }")
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def llama_apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = llama_reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freqs_cis(d, end, theta = 10_000, device = "cpu"):
    assert d % 2 == 0, "dim must be divisible by 2"
    i_s = torch.arange(0,d,2).float()
    theta_s = theta ** (- i_s / d).to(device)
    m = torch.arange(end, device=device)
    freqs = torch.outer(m, theta_s).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x, freqs_cis, device):
        x_c = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], -1, 2)
        ) 
        f_c = freqs_cis.unsqueeze(0).unsqueeze(2)
        x_rotated = x_c * f_c
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(*x.shape)
        return x_out.type_as(x).to(device)


if __name__ == "__main__":
    dim=4096
    n_heads=32 
    max_seq_len=128 
    B, S, H, D = 8, 100, n_heads, dim//n_heads
    context_window=4096 
    x=torch.randn(B,S,H,D)
    out_1, _ = llama_apply_rotary_emb(
        x,
        x,
        llama_precompute_freqs_cis(dim//n_heads, max_seq_len * 2)[:S]
    )

    out_2 = apply_rotary_emb(
        x,
        precompute_freqs_cis(dim//n_heads, max_seq_len * 2)[:S],
        "cpu"
    )
    print(f"{torch.allclose(out_1,out_2,atol=10**-4)}")
