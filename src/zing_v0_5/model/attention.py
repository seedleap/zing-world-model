from __future__ import annotations

import inspect
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_flash_attn_varlen_func = None
if torch.version.hip is None:
    try:
        from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    except ModuleNotFoundError as error:
        if error.name != "flash_attn":
            raise


os.environ.setdefault("TRITON_MAX_BLOCK_X", "8192")
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024
torch._inductor.config.realize_opcount_threshold = 100
torch._dynamo.config.recompile_limit = 1024

_MATH_SDPA_SCORE_BUDGET_BYTES = 512 * 1024**2
_MATH_SDPA_SCORE_ELEMENT_BYTES = 4
_MATH_SDPA_QUERY_ALIGNMENT = 32


class CompiledSegment:
    compiled = {}
    mode = None

    @classmethod
    def get(cls, function, enabled: bool):
        if not enabled:
            return function
        if function not in cls.compiled:
            options = {}
            if "recompile_limit" in inspect.signature(torch.compile).parameters:
                options["recompile_limit"] = 1024
            cls.compiled[function] = torch.compile(function, dynamic=True, mode=cls.mode, **options)
        return cls.compiled[function]


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, compile_fusion: bool):
        super().__init__()
        self.eps = eps
        self.compile_fusion = compile_fusion
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return CompiledSegment.get(self._norm, self.compile_fusion and value.is_cuda)(value, self.weight, self.eps)

    @staticmethod
    def _norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        value_float = value.float()
        return (value_float * torch.rsqrt(value_float.pow(2).mean(dim=-1, keepdim=True) + eps)).type_as(value) * weight


def make_rope_freqs(dim: int, num_heads: int, maximum: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def table(length: int, width: int) -> torch.Tensor:
        positions = torch.arange(length)
        frequencies = 1.0 / torch.pow(10000, torch.arange(0, width, 2).to(torch.float64).div(width))
        angles = torch.outer(positions, frequencies)
        return torch.view_as_real(torch.polar(torch.ones_like(angles), angles)).float()

    head_dim = dim // num_heads
    temporal_width = head_dim - 4 * (head_dim // 6)
    spatial_width = 2 * (head_dim // 6)
    return table(maximum, temporal_width), table(maximum, spatial_width), table(maximum, spatial_width)


def compute_rope(
    positions: torch.Tensor, temporal: torch.Tensor, height: torch.Tensor, width: torch.Tensor
) -> torch.Tensor:
    maxima = positions.max(dim=0).values
    if int(maxima[0]) >= temporal.shape[0] or int(maxima[1]) >= height.shape[0] or int(maxima[2]) >= width.shape[0]:
        raise ValueError("RoPE position exceeds generator.rope_max_seq_len")
    return torch.cat((temporal[positions[:, 0]], height[positions[:, 1]], width[positions[:, 2]]), dim=1)


def apply_rope(value: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    sequence = value.shape[-3]
    head_dim = value.shape[-1]
    shaped = rope.reshape(*([1] * (value.dim() - 3)), sequence, 1, head_dim // 2, 2)
    cosine, sine = shaped[..., 0], shaped[..., 1]
    real, imaginary = value[..., 0::2].float(), value[..., 1::2].float()
    rotated = torch.stack((real * cosine - imaginary * sine, real * sine + imaginary * cosine), dim=-1)
    return rotated.flatten(-2).to(value.dtype)


def _resolve_attention_backend() -> str:
    requested = os.environ.get("ZING_ATTENTION_BACKEND", "auto").strip().lower() or "auto"
    allowed = {"auto", "flash", "sdpa", "sdpa-math"}
    if requested not in allowed:
        raise ValueError(
            f"unsupported ZING_ATTENTION_BACKEND={requested!r}; "
            f"expected one of {sorted(allowed)}"
        )
    if requested == "auto":
        if torch.version.hip is not None:
            return "sdpa-math"
        if _flash_attn_varlen_func is None:
            raise ImportError(
                "ZING_ATTENTION_BACKEND=auto requires the CUDA flash-attn package "
                "on CUDA; select an SDPA backend explicitly to run without it"
            )
        return "flash"
    if requested == "flash" and _flash_attn_varlen_func is None:
        raise ImportError(
            "ZING_ATTENTION_BACKEND=flash requires the CUDA flash-attn package"
        )
    return requested


def _flash_attention_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
    deterministic: bool,
) -> torch.Tensor:
    assert _flash_attn_varlen_func is not None
    cumulative_query = F.pad(query_lengths.to(torch.int32).cumsum(0), (1, 0)).to(torch.int32)
    cumulative_key = F.pad(key_lengths.to(torch.int32).cumsum(0), (1, 0)).to(torch.int32)
    return _flash_attn_varlen_func(
        query,
        key,
        value,
        cumulative_query,
        cumulative_key,
        int(query_lengths.max().item()),
        int(key_lengths.max().item()),
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        deterministic=deterministic,
    )


def _scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    math_only: bool,
) -> torch.Tensor:
    kwargs = {"dropout_p": 0.0, "is_causal": False}
    if not math_only:
        return F.scaled_dot_product_attention(query, key, value, **kwargs)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(query, key, value, **kwargs)


def _math_sdpa_query_chunk_size(
    query: torch.Tensor,
    key: torch.Tensor,
) -> int:
    query_length = int(query.shape[-2])
    key_length = int(key.shape[-2])
    score_row_bytes = (
        int(query.shape[0])
        * int(query.shape[1])
        * key_length
        * _MATH_SDPA_SCORE_ELEMENT_BYTES
    )
    chunk_size = max(1, _MATH_SDPA_SCORE_BUDGET_BYTES // max(1, score_row_bytes))
    if chunk_size >= query_length:
        return query_length
    if chunk_size >= _MATH_SDPA_QUERY_ALIGNMENT:
        chunk_size = (
            chunk_size // _MATH_SDPA_QUERY_ALIGNMENT
        ) * _MATH_SDPA_QUERY_ALIGNMENT
    return max(1, chunk_size)


def _memory_bounded_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    math_only: bool,
) -> torch.Tensor:
    if not math_only:
        return _scaled_dot_product_attention(
            query,
            key,
            value,
            math_only=False,
        )
    chunk_size = _math_sdpa_query_chunk_size(query, key)
    if chunk_size >= query.shape[-2]:
        return _scaled_dot_product_attention(
            query,
            key,
            value,
            math_only=True,
        )
    return torch.cat(
        [
            _scaled_dot_product_attention(
                query[..., start : start + chunk_size, :],
                key,
                value,
                math_only=True,
            )
            for start in range(0, query.shape[-2], chunk_size)
        ],
        dim=-2,
    )


def _sdpa_attention_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
    *,
    math_only: bool,
) -> torch.Tensor:
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("packed attention tensors must have shape (total, heads, dim)")
    if query.shape[1:] != key.shape[1:] or key.shape[1:] != value.shape[1:]:
        raise ValueError("query, key, and value head shapes must match")
    query_lengths = query_lengths.to(device=query.device, dtype=torch.int64)
    key_lengths = key_lengths.to(device=key.device, dtype=torch.int64)
    if (
        query_lengths.ndim != 1
        or key_lengths.ndim != 1
        or query_lengths.numel() == 0
        or query_lengths.numel() != key_lengths.numel()
    ):
        raise ValueError("query_lengths and key_lengths must be non-empty 1-D tensors of equal size")
    if (
        int(query_lengths.sum().item()) != query.shape[0]
        or int(key_lengths.sum().item()) != key.shape[0]
    ):
        raise ValueError("packed token counts must match the length tensors")

    batch = int(query_lengths.numel())
    equal_query = bool((query_lengths == query_lengths[0]).all().item())
    equal_key = bool((key_lengths == key_lengths[0]).all().item())
    if equal_query and equal_key:
        query_length = int(query_lengths[0].item())
        key_length = int(key_lengths[0].item())
        heads, dim = query.shape[1:]
        packed_query = query.reshape(batch, query_length, heads, dim).transpose(1, 2)
        packed_key = key.reshape(batch, key_length, heads, dim).transpose(1, 2)
        packed_value = value.reshape(batch, key_length, heads, dim).transpose(1, 2)
        attended = _memory_bounded_scaled_dot_product_attention(
            packed_query,
            packed_key,
            packed_value,
            math_only=math_only,
        )
        return attended.transpose(1, 2).reshape_as(query)

    outputs = []
    query_offset = 0
    key_offset = 0
    for query_length, key_length in zip(
        query_lengths.tolist(), key_lengths.tolist(), strict=True
    ):
        sample_query = query[query_offset : query_offset + query_length].transpose(0, 1).unsqueeze(0)
        sample_key = key[key_offset : key_offset + key_length].transpose(0, 1).unsqueeze(0)
        sample_value = value[key_offset : key_offset + key_length].transpose(0, 1).unsqueeze(0)
        attended = _memory_bounded_scaled_dot_product_attention(
            sample_query,
            sample_key,
            sample_value,
            math_only=math_only,
        )
        outputs.append(attended.squeeze(0).transpose(0, 1))
        query_offset += query_length
        key_offset += key_length
    return torch.cat(outputs, dim=0)


def flash_attention_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lengths: torch.Tensor,
    key_lengths: torch.Tensor,
    deterministic: bool,
) -> torch.Tensor:
    backend = _resolve_attention_backend()
    if backend == "flash":
        return _flash_attention_varlen(
            query,
            key,
            value,
            query_lengths,
            key_lengths,
            deterministic,
        )
    return _sdpa_attention_varlen(
        query,
        key,
        value,
        query_lengths,
        key_lengths,
        math_only=backend == "sdpa-math" or deterministic,
    )


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float, qk_norm: bool, compile_fusion: bool, deterministic: bool):
        super().__init__()
        if dim % num_heads:
            raise ValueError("attention dimension must be divisible by the head count")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.deterministic = deterministic
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps, compile_fusion) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps, compile_fusion) if qk_norm else nn.Identity()

    def forward(
        self,
        hidden: torch.Tensor,
        query_rope: torch.Tensor,
        key_rope: torch.Tensor,
        history: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden.shape
        query = self.q(hidden)
        key = self.k(hidden)
        if isinstance(self.norm_q, WanRMSNorm):
            query = WanRMSNorm._norm(query, self.norm_q.weight, self.norm_q.eps)
            key = WanRMSNorm._norm(key, self.norm_k.weight, self.norm_k.eps)
        query = query.reshape(batch, sequence, self.num_heads, self.head_dim)
        key = key.reshape(batch, sequence, self.num_heads, self.head_dim)
        value = self.v(hidden).reshape(batch, sequence, self.num_heads, self.head_dim)
        history_key, history_value = history
        raw_key = key if history_key is None else torch.cat((history_key, key), dim=1)
        full_value = value if history_value is None else torch.cat((history_value, value), dim=1)
        query = apply_rope(query, query_rope)
        rotated_key = apply_rope(raw_key, key_rope)
        key_sequence = rotated_key.shape[1]
        query_lengths = torch.full((batch,), sequence, device=hidden.device, dtype=torch.int32)
        key_lengths = torch.full((batch,), key_sequence, device=hidden.device, dtype=torch.int32)
        attended = flash_attention_varlen(
            query.reshape(batch * sequence, self.num_heads, self.head_dim),
            rotated_key.reshape(batch * key_sequence, self.num_heads, self.head_dim),
            full_value.reshape(batch * key_sequence, self.num_heads, self.head_dim),
            query_lengths,
            key_lengths,
            self.deterministic,
        )
        attended = attended.reshape(batch, sequence, self.num_heads * self.head_dim)
        return self.o(attended), key, value


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float, qk_norm: bool, compile_fusion: bool, deterministic: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.deterministic = deterministic
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps, compile_fusion) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps, compile_fusion) if qk_norm else nn.Identity()

    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        context_lengths: torch.Tensor,
        cached: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, sequence, _ = hidden.shape
        query = self.norm_q(self.q(hidden)).reshape(batch * sequence, self.num_heads, self.head_dim)
        key, value = cached
        created = None
        if key is None:
            key = self.norm_k(self.k(context)).reshape(context.shape[0], self.num_heads, self.head_dim)
            value = self.v(context).reshape(context.shape[0], self.num_heads, self.head_dim)
            created = (key, value)
        query_lengths = torch.full((batch,), sequence, device=hidden.device, dtype=torch.int32)
        attended = flash_attention_varlen(
            query, key, value, query_lengths, context_lengths.to(torch.int32), self.deterministic
        )
        return self.o(attended.reshape(batch, sequence, -1)), created
