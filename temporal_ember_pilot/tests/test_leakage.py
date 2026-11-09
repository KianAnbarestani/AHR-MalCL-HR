import hashlib
import unittest

from temporal_ember_pilot.stream import WINDOW_IDS, build_manifest, window_partitions


def _row(name, month, family, *, split=None):
    digest = hashlib.sha256(name.encode()).hexdigest()
    return {
        "sha256": digest,
        "appeared": f"2018-{month:02d}",
        "label": 1,
        "avclass": family,
        "_split_override": split,
    }


class LeakageTests(unittest.TestCase):
    def test_six_windows_and_seed_independent_hash_split(self):
        rows = [_row("jan-a", 1, "alpha"), _row("feb-a", 2, "beta"), _row("mar-a", 3, "alpha")]
        first = build_manifest(rows)
        second = build_manifest(rows)
        self.assertEqual(first["window_ids"], list(WINDOW_IDS))
        self.assertEqual(first["rows"], second["rows"])
        self.assertEqual({row["window_id"] for row in first["rows"]}, {"2018-01", "2018-02_03"})
        self.assertTrue(all(row["split"] in {"train", "heldout"} for row in first["rows"]))

    def test_conflicting_identity_is_quarantined_but_exact_duplicate_collapsed(self):
        row = _row("same", 1, "alpha")
        exact_duplicate = dict(row)
        conflict = dict(row, avclass="different")
        manifest = build_manifest([row, exact_duplicate, conflict])
        self.assertEqual(len(manifest["rows"]), 0)
        self.assertEqual(manifest["audit_counts"]["exact_duplicates_collapsed"], 1)
        self.assertEqual(manifest["audit_counts"]["conflicting_identities_quarantined"], 1)

    def test_january_initial_cohort_uses_january_training_support_only(self):
        rows = []
        for family, count in [("jan_a", 3), ("jan_b", 2), ("future_z", 20)]:
            month = 1 if family != "future_z" else 2
            rows.extend(_row(f"{family}-{i}", month, family) for i in range(count))
        manifest = build_manifest(rows, initial_family_limit=2, admission_family_limit=1)
        admitted = manifest["admissions"]["2018-01"]
        jan_train_support = manifest["family_support_by_window"]["2018-01"]["train"]
        self.assertLessEqual(set(admitted), set(jan_train_support))
        self.assertNotIn("future_z", admitted)
        self.assertEqual(len(admitted), 2)

    def test_no_future_family_backfill_and_update_inputs_are_causal(self):
        rows = []
        for family, month in [("old", 1), ("skipped", 2), ("new_a", 3), ("later", 4)]:
            rows.extend(_row(f"{family}-{i}", month, family) for i in range(16))
        manifest = build_manifest(rows, initial_family_limit=1, admission_family_limit=1)
        self.assertNotIn("skipped", manifest["admitted_families"])
        before_apr = window_partitions(manifest, "2018-04_05")
        lookup = {row["sha256"]: row for row in manifest["rows"]}
        prior_ids = set(before_apr["train_ids"] + before_apr["scoring_ids"])
        self.assertTrue(all(lookup[sid]["window_index"] <= 2 for sid in prior_ids))
        self.assertNotIn("later", before_apr["known_families_before"])
        self.assertTrue(set(before_apr["train_ids"]).isdisjoint(before_apr["scoring_ids"]))
        self.assertTrue(all(lookup[sid]["window_index"] <= 2 for sid in before_apr["train_ids"]))

    def test_holdout_and_future_rows_never_enter_training_or_scaler_fit_ids(self):
        rows = []
        for month, family in [(1, "old"), (2, "old"), (3, "new"), (5, "future")]:
            rows.extend(_row(f"{family}-{month}-{i}", month, family) for i in range(24))
        manifest = build_manifest(rows, initial_family_limit=1, admission_family_limit=1)
        febmar = window_partitions(manifest, "2018-02_03")
        train = set(febmar["train_ids"])
        heldout = {row["sha256"] for row in manifest["rows"] if row["window_id"] == "2018-02_03" and row["split"] == "heldout"}
        future = {row["sha256"] for row in manifest["rows"] if row["window_index"] > 1}
        self.assertTrue(train.isdisjoint(heldout))
        self.assertEqual(febmar["scaler_fit_ids"], febmar["fit_ids"])
        self.assertTrue(set(febmar["scaler_fit_ids"]).isdisjoint(future))
        self.assertTrue(set(febmar["evaluation_ids"]).isdisjoint(train))


if __name__ == "__main__":
    unittest.main()
