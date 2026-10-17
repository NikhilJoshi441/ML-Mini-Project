"""Run the comparable Phase 4 experiments on one fixed dataset split."""

import argparse
import csv
import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

import cnn
import massageData
import transfer_learning


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder", default="data/numpy_bitmap_3/")
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--fine_tune_layers", type=int, default=20)
    parser.add_argument("--include_transfer", action="store_true")
    return parser


def set_flags(module, args, seed):
    module.FLAGS = module.create_parser().parse_args([
        "--data_folder", args.data_folder,
        "--max_iter", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--seed", str(seed),
    ])
    if module is transfer_learning:
        module.FLAGS.transfer_model = "MobileNet"
        module.FLAGS.input_size = args.input_size
        module.FLAGS.fine_tune_layers = args.fine_tune_layers
        module.FLAGS.tune_source_model = False


def load_fixed_split(folder):
    data = massageData.massageData(folder=folder)
    encoder = LabelEncoder().fit(data.y_train)
    num_classes = len(encoder.classes_)
    labels = (
        to_categorical(encoder.transform(data.y_train), num_classes=num_classes),
        to_categorical(encoder.transform(data.y_dev), num_classes=num_classes),
        to_categorical(encoder.transform(data.y_test), num_classes=num_classes),
    )
    arrays = (
        data.X_train.reshape((-1, 28, 28, 1)),
        data.X_dev.reshape((-1, 28, 28, 1)),
        data.X_test.reshape((-1, 28, 28, 1)),
    )
    return arrays, labels


def fit_and_measure(name, model, arrays, labels, epochs, batch_size):
    x_train, x_dev, x_test = arrays
    y_train, y_dev, y_test = labels
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    start = time.time()
    history = model.fit(
        x_train, y_train, validation_data=(x_dev, y_dev),
        epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks,
    )
    elapsed = time.time() - start
    dev = model.evaluate(x_dev, y_dev, verbose=0, return_dict=True)
    test = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    accuracy_key = "accuracy"
    top3_key = "top_3_accuracy"
    return {
        "model": name,
        "dev_top1": dev[accuracy_key],
        "dev_top3": dev[top3_key],
        "test_top1": test[accuracy_key],
        "test_top3": test[top3_key],
        "training_time_secs": elapsed,
        "parameter_count": model.count_params(),
        "train_dev_accuracy_gap": (
            history.history["accuracy"][-1] -
            history.history["val_accuracy"][-1]
        ),
    }


def measure_trained_model(name, model, arrays, labels, elapsed):
    dev = model.evaluate(arrays[1], verbose=0, return_dict=True)
    test = model.evaluate(arrays[2], verbose=0, return_dict=True)
    return {
        "model": name,
        "dev_top1": dev["accuracy"],
        "dev_top3": dev["top_3_accuracy"],
        "test_top1": test["accuracy"],
        "test_top3": test["top_3_accuracy"],
        "training_time_secs": elapsed,
        "parameter_count": model.count_params(),
        "train_dev_accuracy_gap": "",
    }


def run_one(seed, args, arrays, labels):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    rows = []

    set_flags(cnn, args, seed)
    for version, name in ((1, "cnn_v1"), (6, "cnn_v6")):
        model = cnn.model() if version == 1 else cnn.model_v6()
        rows.append(fit_and_measure(name, model, arrays, labels, args.epochs, args.batch_size))

    return rows


def run_transfer_one(seed, args, arrays, labels):
    rows = []
    set_flags(transfer_learning, args, seed)
    for backbone in transfer_learning.available_models:
        transfer_learning.FLAGS.transfer_model = backbone
        image_size, base, preprocess = transfer_learning.get_source_model(backbone)
        datasets = [
            transfer_learning.create_transfer_dataset(
                x, y, image_size, preprocess, args.batch_size,
                shuffle=index == 0
            )
            for index, (x, y) in enumerate(zip(arrays, labels))
        ]
        start = time.time()
        model, _, _ = transfer_learning.transfer_learning(
            datasets[0], datasets[1],
            base, tune_source_model=False,
        )
        rows.append(measure_trained_model(
            backbone, model, datasets, labels, time.time() - start
        ))
    return rows


def main():
    args = create_parser().parse_args()
    arrays, labels = load_fixed_split(args.data_folder)
    rows = []
    for seed in args.seeds:
        rows.extend(run_one(seed, args, arrays, labels))
        if args.include_transfer:
            rows.extend(run_transfer_one(seed, args, arrays, labels))

    fieldnames = ["model", "dev_top1_mean", "dev_top1_std",
                  "dev_top3_mean", "dev_top3_std", "test_top1_mean",
                  "test_top1_std", "test_top3_mean", "test_top3_std",
                  "training_time_secs_mean", "training_time_secs_std",
                  "parameter_count", "train_dev_accuracy_gap_mean",
                  "train_dev_accuracy_gap_std"]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for name in sorted({row["model"] for row in rows}):
            group = [row for row in rows if row["model"] == name]
            def values(key):
                return np.asarray([
                    float(row[key]) for row in group if row[key] != ""
                ])
            summary = {"model": name}
            for key in ("dev_top1", "dev_top3", "test_top1", "test_top3",
                        "training_time_secs", "train_dev_accuracy_gap"):
                data = values(key)
                summary[key + "_mean"] = data.mean() if len(data) else ""
                summary[key + "_std"] = data.std() if len(data) else ""
            summary["parameter_count"] = group[0]["parameter_count"]
            writer.writerow(summary)

    print("Wrote {} experiment rows to {}".format(len(rows), args.output))


if __name__ == "__main__":
    main()
