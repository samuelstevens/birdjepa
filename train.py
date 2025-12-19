"""
https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md
"""

import dataclasses
import logging
import pathlib
import typing as tp

import beartype
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision.ops import MLP
from torchvision.transforms import v2

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logger = logging.getLogger("birdjepa")


@dataclasses.dataclass(frozen=True)
class Config:
    """Training configuration for LeJEPA."""

    lamb: float = 0.02
    """Lambda for loss weighting between sigreg and invariance."""
    V: int = 4
    """Number of augmented views per image."""
    proj_dim: int = 16
    """Projection head output dimension."""
    lr: float = 2e-3
    """Learning rate for encoder."""
    bsz: int = 256
    """Batch size."""
    n_workers: int = 8
    """Number of dataloader workers."""
    epochs: int = 800
    """Number of training epochs."""
    device: tp.Literal["cuda", "cpu", "mps"] = "cuda"
    """Device to train on."""
    # Slurm
    slurm_acct: str = ""
    """Slurm account string. Empty means run locally."""
    slurm_partition: str = ""
    """Slurm partition."""
    n_hours: float = 24.0
    """Slurm job length in hours."""
    log_to: pathlib.Path = pathlib.Path("./logs")
    """Where to log Slurm job stdout/stderr."""


@beartype.beartype
class SIGReg(torch.nn.Module):
    def __init__(self, knots: int = 17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTEncoder(nn.Module):
    embed_dim: int = 192

    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.1,
            img_size=32,
        )
        self.proj = MLP(
            self.embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d
        )

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, split, V=1):
        self.V = V
        self.ds = load_dataset("uoft-cs/cifar100", split=split)
        self.aug = v2.Compose([
            v2.RandomResizedCrop(32, scale=(0.08, 1.0)),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))]),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.test = v2.Compose([
            v2.Resize(32),
            v2.CenterCrop(32),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["img"].convert("RGB")
        transform = self.aug if self.V > 1 else self.test
        return torch.stack([transform(img) for _ in range(self.V)]), item["fine_label"]

    def __len__(self):
        return len(self.ds)


def worker_fn(cfg: Config):
    logging.basicConfig(level=logging.INFO, format=log_format)

    if cfg.device == "cuda":
        assert torch.cuda.is_available(), "CUDA not available but --device=cuda"
    if cfg.device == "mps":
        assert torch.backends.mps.is_available(), "MPS not available but --device=mps"

    wandb.init(project="birdjepa", config=dataclasses.asdict(cfg))
    torch.manual_seed(0)

    train_ds = HFDataset("train", V=cfg.V)
    test_ds = HFDataset("test", V=1)
    train = DataLoader(
        train_ds,
        batch_size=cfg.bsz,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.n_workers,
    )
    test = DataLoader(test_ds, batch_size=256, num_workers=cfg.n_workers)

    # modules and loss
    net = ViTEncoder(proj_dim=cfg.proj_dim).to(cfg.device)
    probe = nn.Sequential(
        nn.LayerNorm(ViTEncoder.embed_dim), nn.Linear(ViTEncoder.embed_dim, 100)
    ).to(cfg.device)
    sigreg = SIGReg().to(cfg.device)
    # Optimizer and scheduler
    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled=cfg.device == "cuda")
    # Training
    for epoch in range(cfg.epochs):
        net.train(), probe.train()
        for vs, y in tqdm.tqdm(train, total=len(train)):
            with autocast(cfg.device, dtype=torch.bfloat16):
                vs = vs.to(cfg.device, non_blocking=True)
                y = y.to(cfg.device, non_blocking=True)
                emb, proj = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                y_rep, yhat = y.repeat_interleave(cfg.V), probe(emb.detach())
                probe_loss = F.cross_entropy(yhat, y_rep)
                loss = lejepa_loss + probe_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            wandb.log({
                "train/probe": probe_loss.item(),
                "train/lejepa": lejepa_loss.item(),
                "train/sigreg": sigreg_loss.item(),
                "train/inv": inv_loss.item(),
            })

        # Evaluation
        net.eval(), probe.eval()
        correct = 0
        with torch.inference_mode():
            for vs, y in test:
                vs = vs.to(cfg.device, non_blocking=True)
                y = y.to(cfg.device, non_blocking=True)
                with autocast(cfg.device, dtype=torch.bfloat16):
                    correct += (probe(net(vs)[0]).argmax(1) == y).sum().item()
        wandb.log({"test/acc": correct / len(test_ds), "test/epoch": epoch})
    wandb.finish()
