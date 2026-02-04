import tensorflow as tf
from tensorflow.keras import layers, initializers
from tensorflow.keras.models import Model, Sequential


class CNN(Model):
    def __init__(self, num_classes: int):
        super().__init__()

        conv_kwargs = {
            "kernel_size": 3,
            "strides": 1,
            "padding": "same",
            "activation": "relu",
        }

        # CNN backbone
        self.cnn = Sequential(
            [
                layers.Conv2D(64, **conv_kwargs),
                layers.Conv2D(64, **conv_kwargs),
                layers.MaxPooling2D(2),
                layers.Conv2D(128, **conv_kwargs),
                layers.Conv2D(128, **conv_kwargs),
                layers.MaxPooling2D(2),
                layers.Conv2D(256, **conv_kwargs),
                layers.Conv2D(256, **conv_kwargs),
                layers.Conv2D(256, **conv_kwargs),
                layers.GlobalMaxPooling2D(),
            ]
        )

        # MLP head
        self.mlp = Sequential(
            [
                layers.Dense(1024, activation="relu"),
                layers.Dense(1024, activation="relu"),
                layers.Dense(num_classes),
            ]
        )

    def call(self, x, training=False):
        x = self.cnn(x, training=training)
        x = self.mlp(x, training=training)
        return x


class VisionTransformer(Model):
    def __init__(
        self,
        num_classes: int,
        *,
        img_size: tuple[int, int] = (32, 32),
        patch_size: int = 4,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 4,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # Patch embedding
        self.patch_embed = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
        )

        # Positional embeddings
        self.pos_embedding = self.add_weight(
            shape=(1, self.num_patches, embed_dim),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="pos_embedding",
        )

        # Transformer blocks
        self.encoder_layers = []
        for _ in range(depth):
            self.encoder_layers.append(
                {
                    "ln1": layers.LayerNormalization(epsilon=1e-6),
                    "mha": layers.MultiHeadAttention(
                        num_heads=num_heads,
                        key_dim=embed_dim,
                    ),
                    "ln2": layers.LayerNormalization(epsilon=1e-6),
                    "ffn1": layers.Dense(embed_dim * 4, activation="gelu"),
                    "ffn2": layers.Dense(embed_dim),
                }
            )

        # Classification head
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.head = layers.Dense(num_classes)

    def call(self, x, training=False):
        # Patch embedding
        x = self.patch_embed(x)
        x = tf.reshape(x, [tf.shape(x)[0], -1, self.embed_dim])
        x = x + self.pos_embedding

        # Transformer encoder
        for layer in self.encoder_layers:
            y = layer["ln1"](x, training=training)
            y = layer["mha"](y, y, training=training)
            x = x + y

            y = layer["ln2"](x, training=training)
            y = layer["ffn1"](y)
            y = layer["ffn2"](y)
            x = x + y

        # Classification head
        x = self.norm(x, training=training)
        x = tf.reduce_mean(x, axis=1)
        x = self.head(x)
        return x
