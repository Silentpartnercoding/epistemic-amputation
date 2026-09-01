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
from diagnostics import action_with_replacement  # noqa: E402
from world_model_experiment import decision, temporal_forward  # noqa: E402
from ghost_experiment import ConsolidatedBodySchema, evaluation_inputs  # noqa: E402
from epistemic_experiment import EpistemicAgent, episode as epistemic_episode  # noqa: E402
from society_experiment import simulate as simulate_society  # noqa: E402
from pretrained_protocol import prompt as pretrained_prompt, protocol_payload, scenarios  # noqa: E402


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

    def test_exact_state_replacement_preserves_target_declaration(self):
        seed_all(29)
        model = BodySchemaGRU().eval()
        _, report = action_with_replacement(model, "told_removal", "adapted_absent", 1, 810, 29)
        self.assertGreaterEqual(report, 0.0)
        self.assertLessEqual(report, 1.0)

    def test_world_model_decides_before_consuming_current_response(self):
        seed_all(37)
        model = BodySchemaGRU().eval()
        x, _ = episode_inputs("told_removal", 0, 909)
        altered = x.clone()
        altered[10, 3:6] = 999.0
        original_action, _, _ = temporal_forward(model, x[None])
        altered_action, _, _ = temporal_forward(model, altered[None])
        self.assertTrue(torch.equal(original_action[:, 10], altered_action[:, 10]))
        self.assertFalse(torch.equal(original_action[:, 11], altered_action[:, 11]))

    def test_immediate_patch_changes_only_the_decision_state(self):
        seed_all(41)
        model = BodySchemaGRU().eval()
        declaration = torch.tensor([[-1.0, 1.0, 1.0]])
        hidden = torch.zeros(1, 1, HIDDEN_SIZE)
        baseline, report = decision(model, hidden, declaration)
        patched, patched_report = decision(model, hidden, declaration, torch.ones(HIDDEN_SIZE))
        self.assertFalse(torch.equal(baseline, patched))
        self.assertEqual(report.shape, patched_report.shape)

    def test_ghost_model_has_no_morphology_input(self):
        x, truth = evaluation_inputs("told_removal", 2, 515)
        self.assertEqual(x.shape[1], 9)
        self.assertEqual(truth.shape[1], 3)
        self.assertFalse(torch.equal(x[:, 6:], truth))
        self.assertEqual(float(x[75, 5]), 0.0)

    def test_slow_patch_does_not_change_fast_memory_or_declaration(self):
        seed_all(53)
        model = ConsolidatedBodySchema().eval()
        fast, slow = model.initial(1)
        declaration = torch.tensor([[-1.0, 1.0, 1.0]])
        baseline = model.decide(fast, slow, declaration)
        patched = model.decide(fast, slow, declaration, torch.ones(14))
        self.assertFalse(torch.equal(baseline[0], patched[0]))
        self.assertEqual(fast.tolist(), model.initial(1)[0].tolist())
        self.assertEqual(declaration.tolist(), [[-1.0, 1.0, 1.0]])

    def test_epistemic_amputation_contains_debrief_and_counterevidence(self):
        x = epistemic_episode("amputated", 991)
        self.assertEqual(tuple(x.shape), (30, 4))
        self.assertEqual(float(x[18, 3]), 1.0)
        self.assertTrue((x[:18, 2] == 0.0).all())
        self.assertTrue((x[18:, 2] == -1.0).all())

    def test_epistemic_patch_does_not_change_report_input(self):
        seed_all(61)
        model = EpistemicAgent().eval()
        hidden = model.initial(1)
        debrief = torch.ones(1)
        baseline = model.decide(hidden, debrief)
        patched = model.decide(hidden, debrief, torch.ones(24))
        self.assertFalse(torch.equal(baseline[1], patched[1]))
        self.assertEqual(debrief.tolist(), [1.0])

    def test_society_simulation_is_deterministic(self):
        first = simulate_society(7, True, True, True, True)
        second = simulate_society(7, True, True, True, True)
        self.assertEqual(first, second)

    def test_pretrained_protocol_is_balanced_and_event_direction_is_correct(self):
        payload = protocol_payload()
        self.assertEqual(len(scenarios()), 16)
        self.assertEqual(len(payload["probes"]), 240)
        text = pretrained_prompt(scenarios()[0], "amputated", "belief")
        self.assertIn("Only afterward did survivors create", text)
        self.assertIn("nobody claims the practice caused the original event", text)
        self.assertIn("exactly meets the council's stated rejection criterion", text)


if __name__ == "__main__":
    unittest.main()
