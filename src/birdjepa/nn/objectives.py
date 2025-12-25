"""Training objectives (JAX)."""

import abc
import dataclasses

import beartype
import chex
import equinox as eqx
import jax
import jax.numpy as jnp

# (sam) TODO: I want to use chex instead of PRNGKeyArray
from jaxtyping import Array, Float, Int, PRNGKeyArray

import birdjepa.data
import birdjepa.nn.transformer


@dataclasses.dataclass(frozen=True)
class SupervisedConfig:
    """Config for supervised objective."""

    source_rank: int = 0
    """Low-rank bottleneck for source prediction head. 0 = disabled."""
    source_weight: float = 1.0
    """Loss weight for source prediction."""


@dataclasses.dataclass(frozen=True)
class LeJEPAConfig:
    """Config for LeJEPA objective."""

    n_views: int = 4
    """Number of augmented views per sample."""
    proj_dim: int = 16
    """Projection head output dimension."""
    lamb: float = 0.02
    """Weight for SIGReg loss (1 - lamb for invariance)."""


@dataclasses.dataclass(frozen=True)
class PixioConfig:
    """Config for Pixio (MAE) objective."""

    decoder_depth: int = 8
    """Number of decoder transformer blocks."""
    decoder_dim: int = 512
    """Decoder embedding dimension."""
    decoder_heads: int = 8
    """Number of decoder attention heads."""
    mask_ratio: float = 0.75
    """Fraction of patches to mask."""
    block_size: int = 2
    """Mask in blocks of this many patches (1 = random patch masking)."""
    seed: int = 42
    """Base seed for masking RNG. Combined with worker_id for per-worker seeding."""
    source_rank: int = 0
    """Low-rank bottleneck for source prediction head. 0 = disabled."""
    source_weight: float = 1.0
    """Loss weight for source prediction."""


Config = SupervisedConfig | LeJEPAConfig | PixioConfig


class Objective(abc.ABC):
    """Abstract base class for training objectives."""

    @abc.abstractmethod
    def wrap_dataset(self, base_ds: birdjepa.data.Dataset) -> birdjepa.data.Dataset:
        """Wrap a dataset for this objective (e.g., add multi-view augmentation)."""
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(
        self,
        batch: dict,
        encoder: eqx.Module,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], Float[Array, "b d"], Int[Array, " b"]]:
        """Compute objective losses.

        Args:
            batch: Dict with at least "data" and "target" keys.
            encoder: Transformer encoder.
            key: PRNG key for forward pass.

        Returns:
            (losses, embeddings, targets) where losses is a dict of named losses.
        """
        ...


# -----------------------------------------------------------------------------
# Supervised
# -----------------------------------------------------------------------------


# (sam) TODO: Use jaxtyped
def softmax_cross_entropy(
    logits: Float[Array, "b c"], labels: Int[Array, " b"]
) -> Float[Array, ""]:
    """Cross-entropy loss for classification."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(one_hot * log_probs, axis=-1).mean()


# (sam) TODO: Use jaxtyped
class Supervised(eqx.Module, Objective):
    """Supervised classification objective."""

    encoder_cfg: birdjepa.nn.transformer.Config = eqx.field(static=True)
    head: eqx.nn.Linear

    def __init__(
        self,
        encoder_cfg: birdjepa.nn.transformer.Config,
        *,
        n_classes: int,
        key: PRNGKeyArray,
    ):
        self.encoder_cfg = encoder_cfg
        self.head = eqx.nn.Linear(encoder_cfg.embed_dim, n_classes, key=key)

    def wrap_dataset(self, base_ds: birdjepa.data.Dataset) -> birdjepa.data.Dataset:
        """Supervised doesn't need to wrap the dataset."""
        return base_ds

    def __call__(
        self,
        batch: dict,
        encoder: eqx.Module,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], Float[Array, "b d"], Int[Array, " b"]]:
        """Compute supervised loss.

        Args:
            batch: Dict with "data" (B, H, W) and "target" (B,).
            encoder: Transformer encoder.
            key: PRNG key for encoder forward pass.

        Returns:
            (losses, embeddings, targets) where losses has "ce" key.
        """
        data_bhw = batch["data"]
        target_b = batch["target"]

        # Patchify and encode
        x_bnk, grid_bn2 = birdjepa.nn.transformer.patchify(data_bhw, self.encoder_cfg)
        out = encoder(x_bnk, grid=grid_bn2, key=key)

        # Use mean of CLS tokens for classification
        emb_bd = out["cls"].mean(axis=1)  # (B, D)

        # Classification loss
        logits = jax.vmap(self.head)(emb_bd)
        ce_loss = softmax_cross_entropy(logits, target_b)

        losses = {"ce": ce_loss}
        return losses, emb_bd, target_b


# -----------------------------------------------------------------------------
# LeJEPA
# -----------------------------------------------------------------------------


class MultiViewDataset:
    """Wraps a base dataset to return multiple views per sample.

    Each call to base_ds[idx] should return a different random crop/augmentation.
    """

    def __init__(self, base_ds: birdjepa.data.Dataset, n_views: int):
        self.ds = base_ds
        self.n_views = n_views
        self.n_classes = base_ds.n_classes

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        items = [self.ds[idx] for _ in range(self.n_views)]
        views = jnp.stack([item["data"] for item in items])
        first = items[0]
        return {"views": views, "target": first["target"], "index": first["index"]}


class SIGReg(eqx.Module):
    """Sketched Isotropic Gaussian Regularization.

    Encourages embeddings to follow a standard Gaussian distribution
    using characteristic function matching.
    """

    t: Float[Array, " k"]
    phi: Float[Array, " k"]
    weights: Float[Array, " k"]

    def __init__(self, n_knots: int = 17):
        t = jnp.linspace(0, 3, n_knots)
        dt = 3 / (n_knots - 1)
        weights = jnp.full((n_knots,), 2 * dt)
        weights = weights.at[0].set(dt)
        weights = weights.at[-1].set(dt)
        window = jnp.exp(-(t**2) / 2.0)
        self.t = t
        self.phi = window
        self.weights = weights * window

    def __call__(
        self, proj_vbd: Float[Array, "v b d"], *, key: PRNGKeyArray
    ) -> Float[Array, ""]:
        """Compute SIGReg loss."""
        d = proj_vbd.shape[-1]
        A_dk = jax.random.normal(key, (d, 256))
        A_dk = A_dk / jnp.linalg.norm(A_dk, axis=0, keepdims=True)
        x_t = (proj_vbd @ A_dk)[..., None] * self.t  # (v, b, 256, k)
        err = (jnp.cos(x_t).mean(axis=-3) - self.phi) ** 2 + jnp.sin(x_t).mean(
            axis=-3
        ) ** 2
        statistic = (err @ self.weights) * proj_vbd.shape[-2]
        return statistic.mean()


class LeJEPA(eqx.Module, Objective):
    """LeJEPA: Joint-Embedding Predictive Architecture with SIGReg."""

    encoder_cfg: birdjepa.nn.transformer.Config = eqx.field(static=True)
    cfg: LeJEPAConfig = eqx.field(static=True)

    proj_linear1: eqx.nn.Linear
    proj_norm1: eqx.nn.LayerNorm
    proj_linear2: eqx.nn.Linear
    proj_norm2: eqx.nn.LayerNorm
    proj_linear3: eqx.nn.Linear
    sigreg: SIGReg

    def __init__(
        self,
        encoder_cfg: birdjepa.nn.transformer.Config,
        cfg: LeJEPAConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.encoder_cfg = encoder_cfg
        self.cfg = cfg

        keys = jax.random.split(key, 3)
        # Projection head: Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm -> ReLU -> Linear
        # Using LayerNorm instead of BatchNorm for simplicity (no running stats needed)
        self.proj_linear1 = eqx.nn.Linear(encoder_cfg.embed_dim, 2048, key=keys[0])
        self.proj_norm1 = eqx.nn.LayerNorm(2048)
        self.proj_linear2 = eqx.nn.Linear(2048, 2048, key=keys[1])
        self.proj_norm2 = eqx.nn.LayerNorm(2048)
        self.proj_linear3 = eqx.nn.Linear(2048, cfg.proj_dim, key=keys[2])
        self.sigreg = SIGReg()

    def _project(self, x: Float[Array, " d"]) -> Float[Array, " p"]:
        """Project a single embedding through the projection head."""
        x = self.proj_linear1(x)
        x = self.proj_norm1(x)
        x = jax.nn.relu(x)
        x = self.proj_linear2(x)
        x = self.proj_norm2(x)
        x = jax.nn.relu(x)
        x = self.proj_linear3(x)
        return x

    def wrap_dataset(self, base_ds: birdjepa.data.Dataset) -> birdjepa.data.Dataset:
        """Wrap dataset with multi-view augmentation."""
        return MultiViewDataset(base_ds, n_views=self.cfg.n_views)

    def __call__(
        self,
        batch: dict,
        encoder: eqx.Module,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], Float[Array, "n d"], Int[Array, " n"]]:
        """Compute LeJEPA loss.

        Args:
            batch: Dict with "views" (B, V, H, W) and "target" (B,).
            encoder: Transformer encoder.
            key: PRNG key for SIGReg and encoder.

        Returns:
            (losses, embeddings, targets) where losses has "inv" and "sigreg" keys.
        """
        enc_key, sigreg_key = jax.random.split(key)

        views_bvhw = batch["views"]
        target_b = batch["target"]
        b, v, h, w = views_bvhw.shape

        # Flatten views for encoder: (B, V, H, W) -> (B*V, H, W)
        views_flat = views_bvhw.reshape(b * v, h, w)

        # Patchify and encode
        x_bnk, grid_bn2 = birdjepa.nn.transformer.patchify(views_flat, self.encoder_cfg)
        out = encoder(x_bnk, grid=grid_bn2, key=enc_key)

        # Use mean of CLS tokens: (n, n_cls, D) -> (n, D) where n = b*v
        emb_nd = out["cls"].mean(axis=1)

        # Project through projection head
        proj_np = jax.vmap(self._project)(emb_nd)  # (n, proj_dim)
        proj_vbp = proj_np.reshape(b, v, -1).transpose(1, 0, 2)  # (v, b, proj_dim)

        # LeJEPA losses
        # Invariance loss: embeddings from different views should match
        mean_proj = proj_vbp.mean(axis=0)  # (b, p)
        inv_loss = ((mean_proj - proj_vbp) ** 2).mean()

        # SIGReg loss: encourage standard Gaussian distribution
        sigreg_loss = self.sigreg(proj_vbp, key=sigreg_key)

        # Repeat targets for each view
        target_n = jnp.repeat(target_b, v)

        losses = {
            "inv": inv_loss * (1 - self.cfg.lamb),
            "sigreg": sigreg_loss * self.cfg.lamb,
        }
        return losses, emb_nd, target_n


# -----------------------------------------------------------------------------
# Pixio (MAE-style)
# -----------------------------------------------------------------------------


class PixioDecoder(eqx.Module):
    """Lightweight decoder for MAE reconstruction.

    CLS token participates in decoder attention to learn better representations.
    """

    encoder_cfg: birdjepa.nn.transformer.Config = eqx.field(static=True)
    cfg: PixioConfig = eqx.field(static=True)

    embed: eqx.nn.Linear
    blocks: list[birdjepa.nn.transformer.Block]
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    pos_embed: Float[Array, "1 n d"]

    def __init__(
        self,
        encoder_cfg: birdjepa.nn.transformer.Config,
        cfg: PixioConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.encoder_cfg = encoder_cfg
        self.cfg = cfg

        keys = jax.random.split(key, cfg.decoder_depth + 3)

        # Project encoder dim to decoder dim
        self.embed = eqx.nn.Linear(encoder_cfg.embed_dim, cfg.decoder_dim, key=keys[0])

        # Decoder blocks
        decoder_cfg = birdjepa.nn.transformer.Config(
            input_h=encoder_cfg.input_h,
            input_w=encoder_cfg.input_w,
            patch_h=encoder_cfg.patch_h,
            patch_w=encoder_cfg.patch_w,
            embed_dim=cfg.decoder_dim,
            depth=cfg.decoder_depth,
            n_heads=cfg.decoder_heads,
        )
        self.blocks = [
            birdjepa.nn.transformer.Block(decoder_cfg, key=keys[i + 1])
            for i in range(cfg.decoder_depth)
        ]
        self.norm = eqx.nn.LayerNorm(cfg.decoder_dim)

        # Project to pixel space
        kernel_size = encoder_cfg.patch_h * encoder_cfg.patch_w
        self.head = eqx.nn.Linear(cfg.decoder_dim, kernel_size, key=keys[-2])

        # Positional embeddings for decoder (patches only)
        self.pos_embed = (
            jax.random.truncated_normal(
                keys[-1], -2, 2, (1, encoder_cfg.n_patches, cfg.decoder_dim)
            )
            * 0.02
        )

    def __call__(
        self,
        cls_bd: Float[Array, "b d"],
        x_bnd: Float[Array, "b n d"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "b d"], Float[Array, "b n k"]]:
        """Decode to pixel predictions with CLS participating in attention.

        Args:
            cls_bd: CLS token from encoder (B, encoder_dim).
            x_bnd: Encoder output + mask tokens (B, n_patches, encoder_dim).
            key: Optional PRNG key for dropout.

        Returns:
            cls_out: Decoded CLS token (B, decoder_dim) for downstream use.
            pred_bnk: Pixel predictions (B, n_patches, patch_h * patch_w).
        """
        # Project to decoder dim
        cls = jax.vmap(self.embed)(cls_bd)[:, None, :]  # (B, 1, decoder_dim)
        x = jax.vmap(jax.vmap(self.embed))(x_bnd)  # (B, N, decoder_dim)

        # Add positional embeddings (patches only)
        x = x + self.pos_embed

        # Concatenate CLS + patches for attention
        x = jnp.concatenate([cls, x], axis=1)  # (B, 1+N, decoder_dim)

        # Decoder blocks
        if key is not None:
            keys = jax.random.split(key, len(self.blocks))
        else:
            keys = [None] * len(self.blocks)

        for block, block_key in zip(self.blocks, keys):
            x = block(x, key=block_key)

        x = jax.vmap(jax.vmap(self.norm))(x)

        # Split CLS and patches
        cls_out = x[:, 0]  # (B, decoder_dim)
        patches = x[:, 1:]  # (B, N, decoder_dim)

        return cls_out, jax.vmap(jax.vmap(self.head))(patches)


class Pixio(eqx.Module, Objective):
    """Pixio: Enhanced MAE with deeper decoder and block masking.

    Reference: "In Pursuit of Pixel Supervision for Visual Pre-training"
    https://arxiv.org/abs/2512.15715
    """

    encoder_cfg: birdjepa.nn.transformer.Config = eqx.field(static=True)
    cfg: PixioConfig = eqx.field(static=True)

    decoder: PixioDecoder
    mask_token: Float[Array, "1 1 d"]

    def __init__(
        self,
        encoder_cfg: birdjepa.nn.transformer.Config,
        cfg: PixioConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.encoder_cfg = encoder_cfg
        self.cfg = cfg

        dec_key, mask_key = jax.random.split(key)
        self.decoder = PixioDecoder(encoder_cfg, cfg, key=dec_key)
        self.mask_token = (
            jax.random.normal(mask_key, (1, 1, encoder_cfg.embed_dim)) * 0.02
        )

    def wrap_dataset(self, base_ds: birdjepa.data.Dataset) -> birdjepa.data.Dataset:
        """Pixio doesn't need to wrap the dataset."""
        return base_ds

    def __call__(
        self,
        batch: dict,
        encoder: eqx.Module,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, Array], Float[Array, "b d"], Int[Array, " b"]]:
        """Compute Pixio (MAE) loss.

        Args:
            batch: Dict with "data" (B, H, W) and "target" (B,).
            encoder: Transformer encoder.
            key: PRNG key for masking and forward pass.

        Returns:
            (losses, embeddings, targets) where losses has "mse" key.
        """
        mask_key, enc_key, dec_key = jax.random.split(key, 3)

        data_bhw = batch["data"]
        target_b = batch["target"]
        b = data_bhw.shape[0]

        # 1. Patchify
        x_bnk, grid_bn2 = birdjepa.nn.transformer.patchify(data_bhw, self.encoder_cfg)
        n_patches = x_bnk.shape[1]

        # 2. Generate mask (True = masked)
        mask_bn = make_block_mask(
            b,
            self.encoder_cfg.n_patches_h,
            self.encoder_cfg.n_patches_w,
            self.cfg.block_size,
            self.cfg.mask_ratio,
            key=mask_key,
        )

        # 3. Compute visible count
        n_visible = int(n_patches * (1 - self.cfg.mask_ratio))

        # 4. Gather visible patches using argsort trick for static shapes
        # Sort by mask value (False=0 comes first), then take first n_visible
        sort_indices = jnp.argsort(mask_bn.astype(jnp.int32), axis=1)
        visible_indices = sort_indices[:, :n_visible]

        # Gather visible patches and grid coords
        x_visible = jnp.take_along_axis(x_bnk, visible_indices[:, :, None], axis=1)
        grid_visible = jnp.take_along_axis(
            grid_bn2, visible_indices[:, :, None], axis=1
        )

        # 5. Encode visible patches only
        out = encoder(x_visible, grid=grid_visible, key=enc_key)
        enc_visible = out["patches"]

        # Mean of CLS tokens
        cls_bd = out["cls"].mean(axis=1)  # (B, D)

        # 6. Prepare decoder input: insert mask tokens at masked positions
        # Start with zeros, add mask tokens, then scatter visible patches
        dec_input = jnp.zeros((b, n_patches, self.encoder_cfg.embed_dim))
        dec_input = dec_input + self.mask_token
        # Then scatter visible patches
        for i in range(n_visible):
            idx = visible_indices[:, i]  # (B,)
            vals = enc_visible[:, i]  # (B, D)
            dec_input = dec_input.at[jnp.arange(b), idx].set(vals)

        # 7. Decode to pixel predictions
        _, pred_bnk = self.decoder(cls_bd, dec_input, key=dec_key)

        # 8. Compute MSE loss on masked patches only (with per-patch normalization)
        target_bnk = x_bnk
        mean = target_bnk.mean(axis=-1, keepdims=True)
        var = target_bnk.var(axis=-1, keepdims=True)
        target_bnk = (target_bnk - mean) / jnp.sqrt(var + 1e-6)

        # Masked MSE: only compute loss on masked patches
        mask_bnk = mask_bn[:, :, None]  # (B, N, 1)
        diff_sq = (pred_bnk - target_bnk) ** 2
        masked_diff = diff_sq * mask_bnk
        mse_loss = masked_diff.sum() / (mask_bn.sum() * pred_bnk.shape[-1])

        losses = {"mse": mse_loss}

        # 9. Return CLS embedding for probe
        emb_bd = cls_bd

        return losses, emb_bd, target_b


@beartype.beartype
def make_objective(
    cfg: Config,
    model_cfg: birdjepa.nn.transformer.Config,
    dataset: birdjepa.data.Dataset,
    *,
    key: chex.PRNGKey,
) -> Objective:
    """Factory for JAX objectives."""
    if isinstance(cfg, SupervisedConfig):
        return Supervised(model_cfg, n_classes=dataset.n_classes, key=key)
    elif isinstance(cfg, LeJEPAConfig):
        return LeJEPA(model_cfg, cfg, key=key)
    elif isinstance(cfg, PixioConfig):
        return Pixio(model_cfg, cfg, key=key)
    else:
        raise ValueError(f"Unknown objective config: {type(cfg)}")


@beartype.beartype
def make_block_mask(
    batch_size: int,
    n_h: int,
    n_w: int,
    block_size: int,
    mask_ratio: float,
    *,
    key: chex.PRNGKey,
) -> Array:
    """Generate block mask with adjustment for exact mask count.

    1. Divide grid into blocks and randomly mask blocks
    2. Adjust to hit exact target count (add individual patches)

    Args:
        batch_size: Number of samples in batch.
        n_h: Number of patches in height.
        n_w: Number of patches in width.
        block_size: Size of masking blocks (2 = 2x2 blocks).
        mask_ratio: Target fraction of patches to mask.
        key: PRNG key for reproducibility.

    Returns:
        Boolean array (B, n_h * n_w) where True = masked.
    """
    n_patches = n_h * n_w
    n_masked_target = int(n_patches * mask_ratio)

    n_blocks_h = (n_h + block_size - 1) // block_size
    n_blocks_w = (n_w + block_size - 1) // block_size
    n_blocks = n_blocks_h * n_blocks_w

    max_patches_per_block = block_size * block_size
    n_blocks_masked = min(n_masked_target // max_patches_per_block, n_blocks)

    keys = jax.random.split(key, batch_size)

    def make_single_mask(sample_key: chex.PRNGKey) -> Array:
        block_key, extra_key = jax.random.split(sample_key)

        # Shuffle blocks and select first n_blocks_masked
        block_perm = jax.random.permutation(block_key, n_blocks)
        masked_block_indices = block_perm[:n_blocks_masked]

        # Create block mask (n_blocks,) then reshape
        block_mask_flat = (
            jnp.zeros(n_blocks, dtype=bool).at[masked_block_indices].set(True)
        )
        block_mask = block_mask_flat.reshape(n_blocks_h, n_blocks_w)

        # Map each patch to its block
        patch_block_h = jnp.arange(n_h) // block_size
        patch_block_w = jnp.arange(n_w) // block_size

        # Create 2D mask by indexing block_mask
        patch_mask_2d = block_mask[patch_block_h[:, None], patch_block_w[None, :]]
        patch_mask = patch_mask_2d.flatten()

        # Add extra patches to reach exact target using priority-based selection.
        # Already-masked patches get priority -inf, unmasked get random priority.
        # Sort descending and mark top n_masked_target as masked.
        priorities = jax.random.uniform(extra_key, (n_patches,))
        priorities = jnp.where(patch_mask, jnp.inf, priorities)
        sorted_indices = jnp.argsort(-priorities)
        new_mask = (
            jnp
            .zeros(n_patches, dtype=bool)
            .at[sorted_indices[:n_masked_target]]
            .set(True)
        )

        return new_mask

    return jax.vmap(make_single_mask)(keys)
