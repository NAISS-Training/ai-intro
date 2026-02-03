import os
from argparse import ArgumentParser

import lightning as L
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.strategies import ModelParallelStrategy
from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from my_torch_models import CNN, VisionTransformer


parser = ArgumentParser()
parser.add_argument("--strategy", choices=["ddp", "fsdp"], default="ddp", help="Which parallel strategy to use")
parser.add_argument("--precision", choices=["32-true", "16-mixed"], default="32-true", help="What precision to use")
parser.add_argument("--accum-batches", type=int, default=1, help="How many accumulated batches per backwards pass")


def get_train_dataloader(batch_size: int, num_workers: int) -> DataLoader:
    # Set-up dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])
    dataset = CIFAR10(
        root="/mimer/NOBACKUP/Datasets/CIFAR/",
        train=True,
        download=False,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0)
    )


# Lightning set-up
class VisionClassifier(L.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # N.B. we don't use .to(device) here, that is handled by Lightning
        x, y = batch
        y_pred = self.model(x)

        loss = cross_entropy(y_pred, y)
        with torch.no_grad():
            acc = (y_pred.argmax(dim=1) == y).float().mean().item()

        self.log("loss", loss, prog_bar=True)
        self.log("acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = AdamW(self.model.parameters(), lr=1e-3)
        return optimizer


def main():
    args = parser.parse_args()


    # Set precision for matrix multiplication
    torch.set_float32_matmul_precision("high")

    # Select model
    seed_everything(30174, workers=True)  # IMPORTANT! Set seed before initializing the model
    model = VisionTransformer(num_classes=10, embed_dim=512, num_heads=512, depth=64)

    # Training part with profiling
    trainer = L.Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=int(os.environ["SLURM_GPUS_ON_NODE"]),
        num_nodes=int(os.environ["SLURM_JOB_NUM_NODES"]),
        strategy=args.strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches,
    )
    trainer.fit(
        model=VisionClassifier(model),
        train_dataloaders=get_train_dataloader(
            num_workers=8,
            batch_size=512 // (trainer.world_size * args.accum_batches),
        ),
    )


if __name__ == "__main__":
    main()
