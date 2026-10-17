"""Focused correctness checks for the Phase 1 data and model fixes."""

import os
import tempfile
import unittest

import numpy as np

import massageData


class Phase1Checks(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
