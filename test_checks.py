"""Regression and smoke tests for the classifier fixes."""

import os
import sys
import tempfile
import unittest
import csv

import numpy as np

import massageData


class TestChecks(unittest.TestCase):
    def test_splits_are_distinct_and_normalized(self):
        with tempfile.TemporaryDirectory() as folder:
            for class_index in range(2):
                values = np.arange(200 * 784, dtype=np.int32).reshape(200, 784)
                values = ((values + class_index) % 256).astype(np.uint8)
                np.save(os.path.join(folder, "class{}.npy".format(class_index)), values)

            data = massageData.massageData(folder=folder + os.sep)
            X_dev, _ = data.getDev()
            X_test, _ = data.getTest()

            self.assertIsNot(X_dev, X_test)
            self.assertFalse(np.array_equal(X_dev, X_test))
            self.assertGreaterEqual(X_dev.min(), 0.0)
            self.assertLessEqual(X_dev.max(), 1.0)
            self.assertGreaterEqual(X_test.min(), 0.0)
            self.assertLessEqual(X_test.max(), 1.0)

    def test_binarize_returns_zero_or_one(self):
        result = massageData.massageData.__new__(massageData.massageData).binarize(
            np.array([[0, 1, 255]], dtype=np.uint8))
        np.testing.assert_array_equal(result, [[0.0, 1.0, 1.0]])

    def test_model_versions_have_distinct_architectures(self):
        try:
            import cnn
        except ImportError as error:
            self.skipTest("CNN dependencies unavailable: {}".format(error))

        with tempfile.TemporaryDirectory() as folder:
            for class_index in range(5):
                open(os.path.join(folder, "class{}.npy".format(class_index)), "wb").close()
            cnn.FLAGS.data_folder = folder
            models = [cnn.model_v2(), cnn.model_v3(), cnn.model_v4(), cnn.model_v5()]

        architectures = [model.to_json() for model in models]
        self.assertEqual(len(architectures), len(set(architectures)))

    def test_cnn_main_dispatches_all_versions(self):
        import cnn

        with tempfile.TemporaryDirectory() as data_folder, tempfile.TemporaryDirectory() as results_folder:
            for class_index in range(3):
                values = np.full((12, 784), class_index * 80, dtype=np.uint8)
                np.save(os.path.join(data_folder, "class{}.npy".format(class_index)), values)

            original_argv = sys.argv
            architectures = []
            try:
                for version in range(1, 7):
                    sys.argv = [
                        "cnn.py", "--data_folder", data_folder,
                        "--results_folder", results_folder, "--max_iter", "1",
                        "--batch_size", "4", "--model_version", str(version),
                    ]
                    cnn.main()
                    with open(
                        os.path.join(results_folder, "simple_cnn_keras_model"),
                        "r"
                    ) as model_file:
                        architectures.append(model_file.read())
            finally:
                sys.argv = original_argv

            self.assertEqual(len(architectures), 6)
            self.assertEqual(len(set(architectures)), 6)

    def test_experiment_runner_one_epoch_on_tiny_data(self):
        import experiment_runner

        with tempfile.TemporaryDirectory() as data_folder, tempfile.TemporaryDirectory() as output_folder:
            for class_index in range(3):
                values = np.full(
                    (12, 784), class_index * 80, dtype=np.uint8
                )
                np.save(
                    os.path.join(data_folder, "class{}.npy".format(class_index)),
                    values
                )

            output = os.path.join(output_folder, "results.csv")
            original_argv = sys.argv
            try:
                sys.argv = [
                    "experiment_runner.py",
                    "--data_folder", data_folder,
                    "--output", output,
                    "--seeds", "7",
                    "--epochs", "1",
                    "--batch_size", "4",
                ]
                experiment_runner.main()
            finally:
                sys.argv = original_argv

            with open(output, newline="") as summary_file:
                summary_rows = list(csv.DictReader(summary_file))
            with open(
                os.path.join(output_folder, "results_raw.csv"), newline=""
            ) as raw_file:
                raw_rows = list(csv.DictReader(raw_file))

            self.assertEqual(
                {row["model"] for row in summary_rows},
                {
                    "cnn_v1",
                    "cnn_v6",
                    "cnn_v6_augmentation",
                    "cnn_v6_augmentation_label_smoothing_01",
                },
            )
            self.assertEqual(len(raw_rows), 4)

    def test_transfer_learning_one_epoch_with_small_backbone(self):
        import transfer_learning
        from tensorflow.keras.applications.mobilenet import MobileNet

        with tempfile.TemporaryDirectory() as data_folder, tempfile.TemporaryDirectory() as results_folder:
            for class_index in range(3):
                values = np.full((6, 784), class_index * 80, dtype=np.uint8)
                np.save(os.path.join(data_folder, "class{}.npy".format(class_index)), values)

            flags = transfer_learning.create_parser().parse_args([])
            flags.data_folder = data_folder
            flags.results_folder = results_folder
            flags.max_iter = 1
            flags.batch_size = 3
            flags.input_size = 32
            flags.seed = 7
            transfer_learning.FLAGS = flags
            data = massageData.massageData(folder=data_folder)
            encoder = transfer_learning.LabelEncoder().fit(data.y_train)
            labels = [
                transfer_learning.to_categorical(
                    encoder.transform(values), num_classes=3
                )
                for values in (data.y_train, data.y_dev)
            ]
            arrays = [
                values.reshape((-1, 28, 28, 1))
                for values in (data.X_train, data.X_dev)
            ]
            backbone = MobileNet(
                input_shape=(32, 32, 3), weights=None, include_top=False
            )
            train_dataset = transfer_learning.create_transfer_dataset(
                arrays[0], labels[0], (32, 32),
                transfer_learning.keras.applications.mobilenet.preprocess_input,
                3, shuffle=True
            )
            dev_dataset = transfer_learning.create_transfer_dataset(
                arrays[1], labels[1], (32, 32),
                transfer_learning.keras.applications.mobilenet.preprocess_input,
                3
            )
            model, head_history, fine_tune_history = transfer_learning.transfer_learning(
                train_dataset, dev_dataset, backbone, tune_source_model=False
            )

            self.assertEqual(len(head_history.history["loss"]), 1)
            self.assertIsNone(fine_tune_history)
            self.assertEqual(model.output_shape[-1], 3)


if __name__ == "__main__":
    unittest.main()
