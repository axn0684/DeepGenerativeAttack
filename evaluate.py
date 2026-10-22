"""Train writer-dependent verifiers and measure FAR/FRR under synthetic attacks.

For each writer, a frozen ImageNet backbone with a small dense head is
trained on the first 2/3 of their genuine signatures and skilled forgeries,
then tested on the remaining 1/3 and on the synthetic forgeries of those
held-out forgeries.

With --retrain (the countermeasure), one synthetic forgery per training
forgery is added to the training set as a negative example.

Example:
    python evaluate.py --data data --synthetic data/cgan_CEDAR_forg.npy --out results/cgan.csv
    python evaluate.py --data data --synthetic data/cgan_CEDAR_forg.npy --retrain --out results/cgan_retrain.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from tensorflow import keras

BACKBONES = {
    "densenet201": keras.applications.DenseNet201,
    "resnet152v2": keras.applications.ResNet152V2,
    "xception": keras.applications.Xception,
}
# Synthetic sample (out of n_generated) used as an extra negative when retraining.
RETRAIN_SAMPLE = 1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"), help="folder with CEDAR_genuine.npy and CEDAR_forg.npy")
    p.add_argument("--synthetic", type=Path, required=True, help="synthetic forgeries .npy from preprocess.py")
    p.add_argument("--backbone", choices=BACKBONES, default="densenet201")
    p.add_argument("--retrain", action="store_true", help="add synthetic forgeries to training (countermeasure)")
    p.add_argument("--writers", type=int, default=55)
    p.add_argument("--samples", type=int, default=24, help="signatures per writer")
    p.add_argument("--n-generated", type=int, default=9, help="synthetic images per forgery")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5, help="decision threshold for FAR/FRR")
    p.add_argument("--out", type=Path, default=Path("results.csv"))
    return p.parse_args()


def build_model(backbone):
    base = BACKBONES[backbone](weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False  # use the backbone as a fixed feature extractor
    x = keras.layers.Flatten()(base.output)
    x = keras.layers.Dense(256, activation="relu")(x)
    out = keras.layers.Dense(2, activation="sigmoid")(x)
    model = keras.Model(base.input, out)
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9),
        metrics=["accuracy"],
    )
    return model


def split_writer(genuine, forgeries, synthetic, w, args):
    """Return train/test sets for writer `w` (0-based). Label 1 = genuine."""
    n, g = args.samples, args.n_generated
    n_train = n * 2 // 3
    real = genuine[w * n:(w + 1) * n]
    forg = forgeries[w * n:(w + 1) * n]
    syn = synthetic[w * n * g:(w + 1) * n * g].reshape(n, g, *synthetic.shape[1:])

    x_train = [real[:n_train], forg[:n_train]]
    y_train = [np.ones(n_train), np.zeros(n_train)]
    if args.retrain:
        x_train.append(syn[:n_train, RETRAIN_SAMPLE])
        y_train.append(np.zeros(n_train))

    x_test = np.concatenate([real[n_train:], forg[n_train:]])
    y_test = np.concatenate([np.ones(n - n_train), np.zeros(n - n_train)])
    x_syn = syn[n_train:].reshape(-1, *synthetic.shape[1:])

    def scale(x):
        return x.astype("float32") / 255.0

    y_train = keras.utils.to_categorical(np.concatenate(y_train), 2)
    return scale(np.concatenate(x_train)), y_train, scale(x_test), y_test, scale(x_syn)


def main():
    args = parse_args()
    genuine = np.load(args.data / "CEDAR_genuine.npy")
    forgeries = np.load(args.data / "CEDAR_forg.npy")
    synthetic = np.load(args.synthetic)

    rows = []
    for w in range(args.writers):
        print(f"----- writer {w + 1}/{args.writers} -----")
        x_train, y_train, x_test, y_test, x_syn = split_writer(genuine, forgeries, synthetic, w, args)
        model = build_model(args.backbone)
        model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=0)

        # A signature is accepted as genuine when its genuine-class score exceeds the threshold.
        accepted = model.predict(x_test, verbose=0)[:, 1] > args.threshold
        syn_accepted = model.predict(x_syn, verbose=0)[:, 1] > args.threshold
        far = 100 * accepted[y_test == 0].mean()
        frr = 100 * (~accepted[y_test == 1]).mean()
        syn_far = 100 * syn_accepted.mean()
        rows.append((w + 1, far, frr, syn_far))
        print(f"FAR {far:.2f}%  FRR {frr:.2f}%  synthetic FAR {syn_far:.2f}%")

    means = np.mean([r[1:] for r in rows], axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["writer", "human_far", "human_frr", "synthetic_far"])
        writer.writerows(rows)
        writer.writerow(["mean", *means])
    print(f"Mean FAR {means[0]:.2f}%  FRR {means[1]:.2f}%  synthetic FAR {means[2]:.2f}%  -> {args.out}")


if __name__ == "__main__":
    main()
