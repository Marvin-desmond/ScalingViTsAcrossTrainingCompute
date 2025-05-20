import torch 

def precompute_freqs_cis(d, end, theta = 10_000, device = "gpu"):
    assert d % 2 == 0, "dim must be divisible by 2"
    i_s = torch.arange(0,d,2).float()
    theta_s = theta ** (- i_s / d).to(device)
    m = torch.arange(end, device=device)
    freqs = torch.outer(m, theta_s).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_embs(x, freqs_cis, device):
        x_c = torch.view_as_complex(
            x.float().reshape(*x.shape[:-1], -1, 2)
        ) 
        f_c = freqs_cis.unsqueeze(0).unsqueeze(2)
        x_rotated = x_c * f_c
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(*x.shape)
        return x_out.type_as(x).to(device)


