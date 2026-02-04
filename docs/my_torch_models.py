import torch
from torch import nn


class CNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()
        conv_kws = {
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
        }
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, **conv_kws), nn.ReLU(),
            nn.Conv2d(64, 64, **conv_kws), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, **conv_kws), nn.ReLU(),
            nn.Conv2d(128, 128, **conv_kws), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, **conv_kws), nn.ReLU(),
            nn.Conv2d(256, 256, **conv_kws), nn.ReLU(),
            nn.Conv2d(256, 256, **conv_kws), nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(start_dim=-3),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        img_size: tuple[int, int] = (32, 32),
        embed_dim: int = 256,
        patch_size: int = 4,
        num_heads: int = 8,
        depth: int = 4,
    ):
        super().__init__()

        # Embeddings
        self.patch_embedding = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size[-2] // patch_size) * (img_size[-1] // patch_size) 
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim),
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)        

        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=depth,
        )

        # Classification head
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embedding
        x = self.encoder(x)
        x = x.mean(dim=1)  # token pooling
        x = self.classification_head(x)
        return x
