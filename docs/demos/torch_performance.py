from argparse import ArgumentParser

import lightning as L
import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from my_torch_models import CNN, VisionTransformer


parser = ArgumentParser()
parser.add_argument("--model", required=True, choices=["cnn", "vit"], help="Which model to use")
parser.add_argument("--num-workers", type=int,  default=0, help="Number of dataloader workers")
parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
parser.add_argument("--fp32-matmul-precision", choices=["highest", "high", "medium"], default="high", help="See, https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html")
parser.add_argument("--precision", choices=["32-true", "16-mixed"], default="32-true")
parser.add_argument("--profiler", required=False, choices=["pytorch", "simple", "advanced"], help="Select profiler if any")
parser.add_argument("--channels-last", action="store_true", help="Activate channels last format for CNN")


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


class ChannelsLastCallback(L.pytorch.callbacks.Callback):
    def setup(self, trainer, pl_module, stage = None):
        pl_module.to(memory_format=torch.channels_last)

    def teardown(self, trainer, pl_module, stage = None):
        pl_module.to(memory_format=torch.contiguous_format)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        batch[0] = batch[0].to(memory_format=torch.channels_last)


def main():
    args = parser.parse_args()

    # Set precision for matrix multiplication
    torch.set_float32_matmul_precision(args.fp32_matmul_precision)

    # Select model
    if args.model == "cnn":
        model = CNN(num_classes=10)
    elif args.model == "vit":
        model = VisionTransformer(num_classes=10)
    else:
        raise RuntimeError(f"Unknown model '{args.model}'")

    callbacks = []
    if args.channels_last:
        # Parse --channels-last flag
        callbacks.append(ChannelsLastCallback())

    # Training part with profiling
    trainer = L.Trainer(
        max_epochs=2,  # use fast_dev_run for even faster runs
        precision=args.precision,
        profiler=args.profiler,
        callbacks=callbacks,
    )
    trainer.fit(
        model=VisionClassifier(model),
        train_dataloaders=get_train_dataloader(
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        ),
    )


if __name__ == "__main__":
    main()
