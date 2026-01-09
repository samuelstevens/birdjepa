"""JAX implementation of Bird-MAE encoder for loading pretrained weights."""

import dataclasses
import functools
import logging
import os.path

import beartype
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import requests
import safetensors.torch
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

logger = logging.getLogger(__name__)

# Bird-MAE audio preprocessing constants
BIRDMAE_SR_HZ = 32_000
BIRDMAE_CLIP_SEC = 5
BIRDMAE_TARGET_T = 512
BIRDMAE_N_MELS = 128


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    img_size_x: int = 512
    img_size_y: int = 128
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    n_heads: int = 12
    mlp_ratio: float = 4.0

    @property
    def n_patches_x(self):
        return self.img_size_x // self.patch_size

    @property
    def n_patches_y(self):
        return self.img_size_y // self.patch_size

    @property
    def n_patches(self):
        return self.n_patches_x * self.n_patches_y


# Sincos positional embeddings (matching Bird-MAE)
def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: tuple[int, int], cls_token: bool = False
) -> np.ndarray:
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size[0], grid_size[1]])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PatchEmbed(eqx.Module):
    """Patch embedding using 2D convolution."""

    proj_weight: Float[Array, "embed_dim in_chans patch_h patch_w"]
    proj_bias: Float[Array, "embed_dim"]
    patch_size: int = eqx.field(static=True)

    def __init__(self, cfg: Config, *, key: PRNGKeyArray):
        k = cfg.patch_size
        self.patch_size = k
        # Conv2d weight: [out_channels, in_channels, H, W]
        self.proj_weight = jr.normal(key, (cfg.embed_dim, 1, k, k)) * 0.02
        self.proj_bias = jnp.zeros(cfg.embed_dim)

    def __call__(
        self, x: Float[Array, "batch 1 height width"]
    ) -> Float[Array, "batch n_patches embed_dim"]:
        b, c, h, w = x.shape
        k = self.patch_size
        # Manual 2D convolution with stride=patch_size
        # Reshape to patches: [B, C, H/k, k, W/k, k] -> [B, H/k*W/k, C*k*k]
        x = x.reshape(b, c, h // k, k, w // k, k)
        x = x.transpose(0, 2, 4, 1, 3, 5)  # [B, H/k, W/k, C, k, k]
        x = x.reshape(b, (h // k) * (w // k), c * k * k)  # [B, N, C*k*k]
        # Linear projection: [B, N, C*k*k] @ [C*k*k, D] -> [B, N, D]
        weight = self.proj_weight.reshape(self.proj_weight.shape[0], -1).T
        x = x @ weight + self.proj_bias
        return x


class Attention(eqx.Module):
    """Multi-head self-attention."""

    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    qkv_weight: Float[Array, "three_dim dim"]
    qkv_bias: Float[Array, "three_dim"]
    proj_weight: Float[Array, "dim dim"]
    proj_bias: Float[Array, "dim"]

    def __init__(self, dim: int, n_heads: int, *, key: PRNGKeyArray):
        k1, k2 = jr.split(key)
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.qkv_weight = jr.normal(k1, (3 * dim, dim)) * 0.02
        self.qkv_bias = jnp.zeros(3 * dim)
        self.proj_weight = jr.normal(k2, (dim, dim)) * 0.02
        self.proj_bias = jnp.zeros(dim)

    def __call__(self, x: Float[Array, "batch n dim"]) -> Float[Array, "batch n dim"]:
        b, n, d = x.shape
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        qkv = qkv.reshape(b, n, 3, self.n_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # [3, B, H, N, D/H]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Manual scaled dot-product attention (matches PyTorch F.scaled_dot_product_attention)
        attn_weights = (q * self.scale) @ k.swapaxes(-2, -1)
        attn_probs = jax.nn.softmax(attn_weights, axis=-1)
        out = attn_probs @ v
        out = out.transpose(0, 2, 1, 3).reshape(b, n, d)
        return out @ self.proj_weight.T + self.proj_bias


class MLP(eqx.Module):
    """MLP with GELU activation."""

    fc1_weight: Float[Array, "hidden dim"]
    fc1_bias: Float[Array, "hidden"]
    fc2_weight: Float[Array, "dim hidden"]
    fc2_bias: Float[Array, "dim"]

    def __init__(self, dim: int, hidden_dim: int, *, key: PRNGKeyArray):
        k1, k2 = jr.split(key)
        self.fc1_weight = jr.normal(k1, (hidden_dim, dim)) * 0.02
        self.fc1_bias = jnp.zeros(hidden_dim)
        self.fc2_weight = jr.normal(k2, (dim, hidden_dim)) * 0.02
        self.fc2_bias = jnp.zeros(dim)

    def __call__(self, x: Float[Array, "batch n dim"]) -> Float[Array, "batch n dim"]:
        x = x @ self.fc1_weight.T + self.fc1_bias
        x = jax.nn.gelu(x)
        x = x @ self.fc2_weight.T + self.fc2_bias
        return x


class Block(eqx.Module):
    """Transformer block (pre-norm)."""

    norm1_weight: Float[Array, "dim"]
    norm1_bias: Float[Array, "dim"]
    attn: Attention
    norm2_weight: Float[Array, "dim"]
    norm2_bias: Float[Array, "dim"]
    mlp: MLP
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, *, key: PRNGKeyArray):
        k1, k2 = jr.split(key)
        self.norm1_weight = jnp.ones(dim)
        self.norm1_bias = jnp.zeros(dim)
        self.attn = Attention(dim, n_heads, key=k1)
        self.norm2_weight = jnp.ones(dim)
        self.norm2_bias = jnp.zeros(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim, key=k2)
        self.eps = 1e-6

    def _norm(self, x, weight, bias):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + self.eps) * weight + bias

    def __call__(self, x: Float[Array, "batch n dim"]) -> Float[Array, "batch n dim"]:
        x = x + self.attn(self._norm(x, self.norm1_weight, self.norm1_bias))
        x = x + self.mlp(self._norm(x, self.norm2_weight, self.norm2_bias))
        return x


class Encoder(eqx.Module):
    """JAX Bird-MAE encoder matching PyTorch architecture."""

    cfg: Config = eqx.field(static=True)
    patch_embed: PatchEmbed
    cls_token: Float[Array, "1 1 dim"]
    pos_embed: Float[Array, "1 n_tokens dim"]
    blocks: list[Block]
    norm_weight: Float[Array, "dim"]
    norm_bias: Float[Array, "dim"]
    fc_norm_weight: Float[Array, "dim"]
    fc_norm_bias: Float[Array, "dim"]
    eps: float = eqx.field(static=True)

    def __init__(self, cfg: Config, *, key: PRNGKeyArray):
        keys = jr.split(key, cfg.depth + 2)
        self.cfg = cfg
        self.eps = 1e-6

        self.patch_embed = PatchEmbed(cfg, key=keys[0])
        self.cls_token = jnp.zeros((1, 1, cfg.embed_dim))

        # Sincos positional embeddings
        pos_embed_np = get_2d_sincos_pos_embed(
            cfg.embed_dim, (cfg.n_patches_x, cfg.n_patches_y), cls_token=True
        )
        self.pos_embed = jnp.array(pos_embed_np[None, :, :], dtype=jnp.float32)

        self.blocks = [
            Block(cfg.embed_dim, cfg.n_heads, cfg.mlp_ratio, key=k)
            for k in keys[2 : 2 + cfg.depth]
        ]

        self.norm_weight = jnp.ones(cfg.embed_dim)
        self.norm_bias = jnp.zeros(cfg.embed_dim)
        self.fc_norm_weight = jnp.ones(cfg.embed_dim)
        self.fc_norm_bias = jnp.zeros(cfg.embed_dim)

    def _norm(self, x, weight, bias):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + self.eps) * weight + bias

    def __call__(self, x: Float[Array, "batch 1 height width"]) -> dict[str, Array]:
        b = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]

        # Add positional embeddings (patches only)
        x = x + self.pos_embed[:, 1:, :]

        # Prepend CLS token with its positional embedding
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = jnp.broadcast_to(cls_token, (b, 1, self.cfg.embed_dim))
        x = jnp.concatenate([cls_tokens, x], axis=1)

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Mean pooling over patches (matching Bird-MAE default)
        pooled = x[:, 1:, :].mean(axis=1)
        pooled = self._norm(pooled, self.fc_norm_weight, self.fc_norm_bias)

        return {"pooled": pooled, "tokens": x[:, 1:, :]}


_PRETRAINED_CFGS = {
    "Bird-MAE-Base": Config(depth=12, embed_dim=768, n_heads=12),
    "Bird-MAE-Large": Config(depth=24, embed_dim=1024, n_heads=16),
    "Bird-MAE-Huge": Config(depth=32, embed_dim=1280, n_heads=16),
}


@beartype.beartype
def load(ckpt: str, *, key: PRNGKeyArray | None = None) -> Encoder:
    """Load Bird-MAE weights into JAX model.

    Args:
        ckpt: Checkpoint name (e.g. "Bird-MAE-Base")
        key: PRNG key for model initialization (weights will be overwritten)

    Returns:
        JAX Encoder with pretrained weights
    """
    if ckpt not in _PRETRAINED_CFGS:
        raise ValueError(f"Checkpoint '{ckpt}' not in {list(_PRETRAINED_CFGS)}.")

    cfg = _PRETRAINED_CFGS[ckpt]

    # Download weights
    fpath = download_hf_file(ckpt)
    state_dict = safetensors.torch.load_file(fpath)

    # Create JAX model (random init, will overwrite)
    if key is None:
        key = jr.key(0)
    model = Encoder(cfg, key=key)

    # Convert PyTorch state dict to JAX
    model = _load_state_dict(model, state_dict)

    logger.info("Loaded Bird-MAE checkpoint: %s", ckpt)
    return model


def _tensor_to_jax(t) -> jnp.ndarray:
    """Convert PyTorch tensor to JAX array."""
    return jnp.array(t.numpy())


def _load_state_dict(model: Encoder, state_dict: dict) -> Encoder:
    """Load PyTorch state dict into JAX model."""

    # Patch embed
    model = eqx.tree_at(
        lambda m: m.patch_embed.proj_weight,
        model,
        _tensor_to_jax(state_dict["patch_embed.proj.weight"]),
    )
    model = eqx.tree_at(
        lambda m: m.patch_embed.proj_bias,
        model,
        _tensor_to_jax(state_dict["patch_embed.proj.bias"]),
    )

    # CLS token
    model = eqx.tree_at(
        lambda m: m.cls_token, model, _tensor_to_jax(state_dict["cls_token"])
    )

    # Positional embedding (frozen sincos, but still in state dict)
    model = eqx.tree_at(
        lambda m: m.pos_embed, model, _tensor_to_jax(state_dict["pos_embed"])
    )

    # Blocks
    for i, blk in enumerate(model.blocks):
        prefix = f"blocks.{i}."

        # Norm1
        model = eqx.tree_at(
            lambda m: m.blocks[i].norm1_weight,
            model,
            _tensor_to_jax(state_dict[prefix + "norm1.weight"]),
        )
        model = eqx.tree_at(
            lambda m: m.blocks[i].norm1_bias,
            model,
            _tensor_to_jax(state_dict[prefix + "norm1.bias"]),
        )

        # Attention
        model = eqx.tree_at(
            lambda m: m.blocks[i].attn.qkv_weight,
            model,
            _tensor_to_jax(state_dict[prefix + "attn.qkv.weight"]),
        )
        model = eqx.tree_at(
            lambda m: m.blocks[i].attn.qkv_bias,
            model,
            _tensor_to_jax(state_dict[prefix + "attn.qkv.bias"]),
        )
        model = eqx.tree_at(
            lambda m: m.blocks[i].attn.proj_weight,
            model,
            _tensor_to_jax(state_dict[prefix + "attn.proj.weight"]),
        )
        model = eqx.tree_at(
            lambda m: m.blocks[i].attn.proj_bias,
            model,
            _tensor_to_jax(state_dict[prefix + "attn.proj.bias"]),
        )

        # Norm2
        model = eqx.tree_at(
            lambda m: m.blocks[i].norm2_weight,
            model,
            _tensor_to_jax(state_dict[prefix + "norm2.weight"]),
        )
        model = eqx.tree_at(
            lambda m: m.blocks[i].norm2_bias,
            model,
            _tensor_to_jax(state_dict[prefix + "norm2.bias"]),
        )

        # MLP
        model = eqx.tree_at(
            lambda m: m.blocks[i].mlp.fc1_weight,
            model,
            _tensor_to_jax(state_dict[prefix + "mlp.fc1.weight"]),
        )
        model = eqx.tree_at(
            lambda m: m.blocks[i].mlp.fc1_bias,
            model,
            _tensor_to_jax(state_dict[prefix + "mlp.fc1.bias"]),
        )
        model = eqx.tree_at(
            lambda m: m.blocks[i].mlp.fc2_weight,
            model,
            _tensor_to_jax(state_dict[prefix + "mlp.fc2.weight"]),
        )
        model = eqx.tree_at(
            lambda m: m.blocks[i].mlp.fc2_bias,
            model,
            _tensor_to_jax(state_dict[prefix + "mlp.fc2.bias"]),
        )

    # Final norms
    model = eqx.tree_at(
        lambda m: m.norm_weight, model, _tensor_to_jax(state_dict["norm.weight"])
    )
    model = eqx.tree_at(
        lambda m: m.norm_bias, model, _tensor_to_jax(state_dict["norm.bias"])
    )
    model = eqx.tree_at(
        lambda m: m.fc_norm_weight, model, _tensor_to_jax(state_dict["fc_norm.weight"])
    )
    model = eqx.tree_at(
        lambda m: m.fc_norm_bias, model, _tensor_to_jax(state_dict["fc_norm.bias"])
    )

    return model


@beartype.beartype
def download_hf_file(ckpt: str, *, force: bool = False) -> str:
    """Download Bird-MAE checkpoint from HuggingFace."""
    url = f"https://huggingface.co/DBD-research-group/{ckpt}/resolve/main/model.safetensors"

    cache_dir = os.path.expanduser(
        os.environ.get("BIRDJEPA_CACHE", "~/.cache/birdjepa")
    )
    local_dir = os.path.join(cache_dir, "hf", ckpt)
    local_path = os.path.join(local_dir, "model.safetensors")

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path) and not force:
        return local_path

    logger.info("Downloading %s to %s", url, local_path)
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return local_path
