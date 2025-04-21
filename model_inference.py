import torch
from torch import nn
import torch.nn.functional as F


def pre_compute_cis(head_dim: int, seq_len: int, theta_base: float = 1e4):
    exponent = -torch.arange(0, head_dim, 2) / head_dim
    theta = theta_base**exponent
    m = torch.arange(1, seq_len + 1).float()
    angle = torch.outer(m, theta)
    # output: (seq_len, head_dim//2)
    return torch.polar(torch.ones_like(angle), angle)


def apply_rotary_pos_embeddings(x: torch.Tensor, theta_cis: torch.Tensor):
    # (batch_size, seq_len, n_head, head_dim) -> (batch_size, n_head, seq_len, head_dim//2)
    x_complex = torch.view_as_complex(x.view(*x.shape[:-1], -1, 2).transpose(1, 2))
    # (batch_size, n_head, seq_len, head_dim//2)*(seq_len, head_dim//2) -> (batch_size, n_head, seq_len, head_dim//2)
    x_rotated = x_complex * theta_cis
    # (batch_size, n_head, seq_len, head_dim//2) -> (batch_size, n_head, seq_len, head_dim//2, 2)
    x_rotated_real = torch.view_as_real(x_rotated)
    # (batch_size, n_head, seq_len, head_dim//2, 2) -> (batch_size, seq_len, n_head, head_dim)
    return x_rotated_real.transpose(1, 2).view(*x.shape)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scalar = nn.Parameter(torch.ones(dim), requires_grad=True)

    def rec_rms(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        x_square = torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps
        return torch.rsqrt(x_square)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.scalar) * self.rec_rms(x)


# Self attention with KV cache and GQA
class SelfAttention(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.n_kv_heads = (
            args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        )
        self.n_repeats = args.n_heads // self.n_kv_heads

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, self.head_dim * self.n_kv_heads, bias=False)
        self.wv = nn.Linear(args.dim, self.head_dim * self.n_kv_heads, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

        self.k_cache = torch.zeros(
            size=(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        ).to(args.device)

        self.v_cache = torch.zeros(
            size=(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        ).to(args.device)

    def forward(
        self, x: torch.Tensor, start_pos: int, theta_cis: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, dim = x.size()
        # seq_len always equals to 1
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, n_heads, head_dim)
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, n_kv_head, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, n_kv_head, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (batch_size, seq_len, n_kv_head, head_dim) -> (batch_size, seq_len, n_kv_head, head_dim)
        xk = apply_rotary_pos_embeddings(xk, theta_cis)
        xv = apply_rotary_pos_embeddings(xv, theta_cis)

        # Cache the current k, v vectors
        self.k_cache[:batch_size, start_pos : start_pos + seq_len] = xk
        self.v_cache[:batch_size, start_pos : start_pos + seq_len] = xv

        # Get all the k and v's from cache
        # (batch_size, start_pos+1, n_kv_head, head_dim) -> (batch_size, start_pos+1, n_kv_head, head_dim)
        xk = self.k_cache[:batch_size, : start_pos + seq_len]
        xv = self.v_cache[:batch_size, : start_pos + seq_len]

        # Copy key and value heads to match query
        # (batch_size, start_pos+1, n_kv_head, head_dim) -> (batch_size, start_pos+1, n_heads, head_dim)
        xk = xk.repeat_interleave(repeats=self.n_repeats, dim=-2)
        xv = xv.repeat_interleave(repeats=self.n_repeats, dim=-2)

        # Transpose to make n_head to be part of batch dimensions
        # (batch_size, seq_len, n_heads, head_dim) -> (batch_size, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        # (batch_size, start_pos+1, n_heads, head_dim) -> (batch_size, n_heads, start_pos+1, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Remember seq_len = 1
        # (batch_size, n_head, 1, head_dim) @ (batch_size, n_heads, head_dim, start_pos+1)
        # -> (batch_size, n_heads, 1, start_pos+1)
        att = (xq @ xk.transpose(-1, -2)) * (self.head_dim**-0.5)
        att = F.softmax(att, dim=-1)
        # (batch_size, n_heads, 1, start_pos+1) @ (batch_size, n_heads, start_pos+1, head_dim)
        # -> (batch_size, n_heads, 1, head_dim)
        y = att @ xv

        # (batch_size, n_heads, 1, head_dim) -> (batch_size, 1, dim)
        y = y.transpose(1, 2).view(batch_size, seq_len, self.n_heads * self.head_dim)
        # (batch_size, 1, dim) -> (batch_size, 1, dim)
        return self.wo(y)


class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.W1 = nn.Linear(args.dim, args.dim * 4, bias=False)
        self.W2 = nn.Linear(args.dim, args.dim * 4, bias=False)
        self.W3 = nn.Linear(args.dim * 4, args.dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.silu(self.W1(x))
        x = gate * self.W2(x)
        return self.W3(x)


class Block(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.rms_attention = RMSNorm(dim=args.dim, eps=args.rms_eps)
        self.attention = SelfAttention(args)
        self.rms_ffd = RMSNorm(dim=args.dim, eps=args.rms_eps)
        self.ffd = FeedForward(args)

    def forward(
        self, x: torch.Tensor, start_pos: int, theta_cis: float
    ) -> torch.Tensor:
        x = x + self.attention(self.rms_attention(x), start_pos, theta_cis)
        x = x + self.ffd(self.rms_ffd(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.dim = args.dim
        self.tok_emb = nn.Embedding(args.vocab_size, args.dim)
        self.register_buffer(
            "theta_cis",
            pre_compute_cis(args.dim // args.n_heads, args.max_seq_len).to(args.device),
        )
        self.blocks = nn.ModuleList(Block(args) for _ in range(args.n_layers))
        self.ln_rms = RMSNorm(args.dim, eps=args.rms_eps)
        self.ln_head = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        batch_size, seq_len = x.shape
        assert seq_len == 1, (
            f"Inference code is using kv cache therefore sequence length has to be 1 not {seq_len}"
        )
        x = self.tok_emb(x)
        theta_cis = self.theta_cis[start_pos : start_pos + seq_len]
        for block in self.blocks:
            x = block(x, start_pos, theta_cis)
        x = self.ln_rms(x)
        return self.ln_head(x)
