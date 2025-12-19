"""
Registry for audio backbones used in benchmarking.
"""

import dataclasses
import logging

import beartype
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

logger = logging.getLogger(__name__)


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class EncodedAudioBatch:
    """Output of AudioBackbone.audio_encode()."""

    features: Float[Tensor, "batch dim"]
    """Pooled audio-level features."""


class AudioBackbone(torch.nn.Module):
    """
    A frozen audio model that embeds batches of spectrograms into feature vectors.
    """

    def audio_encode(
        self, batch: Float[Tensor, "batch time mels"]
    ) -> EncodedAudioBatch:
        """Encode a batch of spectrograms."""
        raise NotImplementedError

    def make_audio_transform(self):
        """Return the preprocessing function (waveform -> spectrogram)."""
        raise NotImplementedError


_registry: dict[str, type[AudioBackbone]] = {}


@beartype.beartype
def register(name: str, cls: type[AudioBackbone]):
    """Register an audio backbone class."""
    if name in _registry:
        logger.warning("Overwriting '%s' in registry.", name)
    _registry[name] = cls


@beartype.beartype
def load(org: str, ckpt: str) -> AudioBackbone:
    """Load a registered audio backbone."""
    if org not in _registry:
        raise ValueError(f"'{org}' not in registry. Available: {list(_registry)}")
    return _registry[org](ckpt)


def list_backbones() -> list[str]:
    """List registered backbone names."""
    return list(_registry)


# Bird-MAE wrapper
class BirdMAEBackbone(AudioBackbone):
    """Wrapper around Bird-MAE for the benchmark registry."""

    def __init__(self, ckpt: str):
        super().__init__()
        from birdjepa.nn import bird_mae

        self.model = bird_mae.load(ckpt)
        self._transform = bird_mae.transform
        self._ckpt = ckpt

    @torch.inference_mode()
    def audio_encode(
        self, batch: Float[Tensor, "batch time mels"]
    ) -> EncodedAudioBatch:
        # batch: [B, 512, 128] -> add channel dim -> [B, 1, 512, 128]
        x = batch[:, None, :, :]
        out = self.model(x)
        # out["pooled"] is [B, dim] - mean pooling over patch tokens
        return EncodedAudioBatch(features=out["pooled"])

    def make_audio_transform(self):
        return self._transform


# Register Bird-MAE
register("bird-mae", BirdMAEBackbone)
