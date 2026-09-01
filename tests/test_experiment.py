import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import (  # noqa: E402
    HIDDEN_SIZE,
    BodySchemaGRU,
    concept_direction,
    episode_inputs,
    orthogonal_controls,
    rollout,
    seed_all,
)


class PhantomSchemaTests(unittest.TestCase):
    def test_no_availability_mask_enters_controller(self):
        x, availability = episode_inputs("told_removal", 1, 99)
        self.assertEqual(x.shape[1], 9)
        self.assertEqual(availability.shape[1], 3)
        # Inputs contain command, response, and declaration—not the truth mask.
        self.assertFalse(torch.equal(x[:, 6:], availability))

    def test_removal_produces_real_negative_sensor_evidence(self):
        x, _ = episode_inputs("told_removal", 2, 123)
        command = x[30, 2]
        response = x[30, 5]
        self.assertGreater(abs(float(command)), 1e-4)
        self.assertLess(abs(float(response)), 0.05)
        self.assertEqual(float(x[30, 8]), -1.0)

    def test_patch_controls_are_orthogonal_and_norm_matched(self):
        direction = torch.arange(1, HIDDEN_SIZE + 1, dtype=torch.float32)
        controls = orthogonal_controls(direction, 8, 10)
        for control in controls:
            self.assertAlmostEqual(float(torch.dot(control, direction)), 0.0, places=3)
            self.assertAlmostEqual(float(control.norm()), float(direction.norm()), places=4)

    def test_rollout_is_deterministic(self):
        seed_all(7)
        model = BodySchemaGRU().eval()
        first = rollout(model, "told_removal", 0, 555)
        second = rollout(model, "told_removal", 0, 555)
        self.assertEqual(first, second)

    def test_concept_direction_has_expected_shape(self):
        seed_all(17)
        model = BodySchemaGRU().eval()
        direction = concept_direction(model, 17, 1)
        self.assertEqual(tuple(direction.shape), (HIDDEN_SIZE,))
        self.assertTrue(torch.isfinite(direction).all())


if __name__ == "__main__":
    unittest.main()
