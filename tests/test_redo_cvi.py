import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import redo_cvi_protocol as p  # noqa: E402


class RedoCopiedIndependentTests(unittest.TestCase):
    def test_split_sizes(self):
        rows = p.scenarios()
        self.assertEqual(len(rows), 120)
        self.assertEqual(sum(r["split"] == "calibration" for r in rows), 24)
        self.assertEqual(len({r["id"] for r in rows}), 120)

    def test_formation_precedes_defeater_and_only_defeater_varies(self):
        row = p.scenarios()[5]
        for level in p.FORMATION_LEVELS:
            texts = {c: p.prompt(row, c, level) for c in p.CONDITIONS}
            head = p.base_story(row) + "\n\n" + p.formation(row, level)
            for c, text in texts.items():
                self.assertTrue(text.startswith(head))
                self.assertTrue(text.endswith(p.question(row)))

    def test_copied_and_independent_share_sources_sentence(self):
        row = p.scenarios()[0]
        a = p.defeater(row, "copied").split(". ")[0]
        b = p.defeater(row, "independent").split(". ")[0]
        self.assertEqual(a, b)

    def test_letters_counterbalanced(self):
        rows = p.scenarios()
        self.assertEqual(sum(r["yes_letter"] == "A" for r in rows), 60)

    def test_analysis_recovers_planted_effect_and_null(self):
        rng = random.Random(1)
        records = []
        for row in p.scenarios():
            if row["split"] != "heldout":
                continue
            base = 3.0 + rng.gauss(0, 0.5)
            values = {"null": base, "independent": base - 3.0, "copied": base - 1.0, "sham": base - 0.2,
                      "irrelevant": base - 0.1}
            for c, v in values.items():
                records.append({"scenario": row["id"], "split": "heldout", "level": 2, "condition": c,
                                "yes_minus_no": v + rng.gauss(0, 0.3)})
        out = p.analyze(records, 2)
        self.assertEqual(out["verdict"], "SUPPORTED")
        for r in records:
            if r["condition"] == "copied":
                twin = next(x for x in records if x["scenario"] == r["scenario"] and x["condition"] == "independent")
                r["yes_minus_no"] = twin["yes_minus_no"] + rng.gauss(0, 0.3)
        self.assertEqual(p.analyze(records, 2)["verdict"], "NOT_SUPPORTED")

    def test_calibration_picks_lowest_level_reaching_target(self):
        records = []
        for row in p.scenarios():
            if row["split"] != "calibration":
                continue
            for level in p.FORMATION_LEVELS:
                records.append({"scenario": row["id"], "split": "calibration", "level": level, "condition": "null",
                                "yes_minus_no": 0.8 * level})
        self.assertEqual(p.calibrate(records)["chosen_level"], 3)


if __name__ == "__main__":
    unittest.main()
