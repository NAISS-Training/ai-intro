import os
import pickle
from argparse import ArgumentParser
from glob import iglob

import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.distribute import MultiWorkerMirroredStrategy
from tensorflow.distribute.cluster_resolver import SlurmClusterResolver
from tensorflow.keras import mixed_precision
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import AdamW

from my_tf_models import CNN, VisionTransformer


parser = ArgumentParser()
parser.add_argument("--dtype-policy", type=mixed_precision.Policy, help="Which dtype policy to use")  # fmt: skip


def get_train_dataset(
    *,
    batch_size: int,
    datadir: str = "/mimer/NOBACKUP/Datasets/CIFAR/cifar-10-batches-py",
) -> Dataset:
    # Prepare base dataset
    x, y = [], []
    for file in iglob(os.path.join(datadir, "data_batch_*")):
        with open(file, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        x.append(d[b"data"])
        y.extend(d[b"labels"])

    x = np.concatenate(x).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(y)

    dataset = Dataset.from_tensor_slices((x, y))

    # Prepare dataset for training
    def transform(x, y):
        mean = tf.constant([0.4914, 0.4822, 0.4465])
        std = tf.constant([0.2470, 0.2435, 0.2616])

        x = tf.cast(x, tf.float32)
        return (x - mean) / std, y

    dataset = dataset.shuffle(
        buffer_size=dataset.cardinality(),
        reshuffle_each_iteration=True,
    )
    dataset = dataset.map(transform)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def main():
    args = parser.parse_args()

    # Select dtype policy
    # N.B. needs to be done before layer creation
    if args.dtype_policy:
        mixed_precision.set_global_policy(args.dtype_policy)

    # Data parallelism set-up
    cluster_resolver = SlurmClusterResolver()
    strategy = MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver)
    with strategy:
        model = VisionTransformer(
            num_classes=10, embed_dim=512, num_heads=512, depth=64
        )
    model.compile(
        optimizer=AdamW(learning_rate=1e-3),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy(name="acc")],
    )

    dataset = get_train_dataset(batch_size=512)  # global batch size
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    # Run the training
    history = model.fit(
        dist_dataset,
        epochs=5,
    )


if __name__ == "__main__":
    main()
