import os
import pickle
from argparse import ArgumentParser
from glob import iglob

import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.keras import mixed_precision
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import AdamW

from my_tf_models import CNN, VisionTransformer


parser = ArgumentParser()
parser.add_argument("--model", required=True, choices=["cnn", "vit"], help="Which model to use")  # fmt: skip
parser.add_argument("--use-prefetch", action="store_true", help="Enable prefetching data")  # fmt: skip
parser.add_argument("--batch-size", type=int, default=512, help="Batch size")  # fmt: skip
parser.add_argument("--dtype-policy", type=mixed_precision.Policy, help="Which dtype policy to use")  # fmt: skip
parser.add_argument("--profile", action="store_true", help="Enable profiling")


def get_train_dataset(
    *,
    batch_size: int,
    use_prefetch: bool = False,
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
        mean=tf.constant([0.4914, 0.4822, 0.4465])
        std=tf.constant([0.2470, 0.2435, 0.2616])

        x = tf.cast(x, tf.float32)
        return (x - mean) / std, y

    dataset = dataset.shuffle(
        buffer_size=dataset.cardinality(),
        reshuffle_each_iteration=True,
    )
    dataset = dataset.map(transform)
    dataset = dataset.batch(batch_size)
    if use_prefetch:
        dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def main():
    args = parser.parse_args()

    # Select dtype policy
    # N.B. needs to be done before layer creation
    if args.dtype_policy:
        mixed_precision.set_global_policy(args.dtype_policy)

    # Select model
    if args.model == "cnn":
        model = CNN(num_classes=10)
    elif args.model == "vit":
        model = VisionTransformer(num_classes=10)
    else:
        raise RuntimeError(f"Unknown model '{args.model}'")

    # Prepare model, optimizer and metrics
    model.compile(
        optimizer=AdamW(learning_rate=1e-3),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy(name="acc")],
    )

    # Set-up profiler callback (profiling batches 10 to 15)
    callbacks = []
    if args.profile:
        callbacks.append(tf.keras.callbacks.TensorBoard(profile_batch="10, 15"))

    # Run the training
    history = model.fit(
        get_train_dataset(
            batch_size=args.batch_size,
            use_prefetch=args.use_prefetch,
        ),
        epochs=2,
        #steps_per_epoch=20,  # for even shorter dev runs
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
