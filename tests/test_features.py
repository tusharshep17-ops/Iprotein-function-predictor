from __future__ import annotations

import unittest

from protfunc.features import ProteinFeatureExtractor, sequence_descriptors


class FeatureTests(unittest.TestCase):
    def test_descriptor_shape(self) -> None:
        descriptors = sequence_descriptors("ACDEFGHIKLMNPQRSTVWY")
        self.assertEqual(descriptors.shape, (28,))

    def test_feature_extractor_returns_rows(self) -> None:
        sequences = [
            "ACDEFGHIKLMNPQRSTVWYACDEFGHIK",
            "YWVTSRQPNMLKIHGFEDCAYWVTSRQPN",
            "GGGGAAAAVVVVLLLLMMMMFFFFYYYYTT",
            "KKKKRRRRHHHHEEEEEDDDDNQSTCNQS",
        ]
        extractor = ProteinFeatureExtractor(max_features=200, min_df=1)
        matrix = extractor.fit_transform(sequences)
        self.assertEqual(matrix.shape[0], len(sequences))
        self.assertGreater(matrix.shape[1], 28)


if __name__ == "__main__":
    unittest.main()

