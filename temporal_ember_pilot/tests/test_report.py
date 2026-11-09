import hashlib
import tempfile
import unittest
from pathlib import Path

from temporal_ember_pilot.report import _export_report_records
from temporal_ember_pilot.stream import WINDOW_IDS, manifest_sha256
from temporal_ember_pilot.config import ARMS, EXPERIMENT_ID, SEEDS


def _metric(value, count, family_counts=None):
    return {
        "macro_f1": value,
        "sample_count": count,
        "scoring_ids_sha256": hashlib.sha256(f"score-{count}".encode()).hexdigest(),
        "per_family_sample_counts": family_counts or {"f_old": count},
        "per_family_f1": {family: 0.5 for family in (family_counts or {"f_old": count})},
    }


def _manifest():
    admissions = {
        WINDOW_IDS[0]: ["f_old"],
        WINDOW_IDS[1]: ["f_new1", "f_new2"],
        WINDOW_IDS[2]: ["f_new3"],
        WINDOW_IDS[3]: ["f_new4"],
        WINDOW_IDS[4]: ["f_new5"],
        WINDOW_IDS[5]: ["f_oct"],
    }
    support = {}
    rows = []
    for index, window_id in enumerate(WINDOW_IDS):
        previous_families = admissions[WINDOW_IDS[index - 1]] if index > 1 else []
        support[window_id] = {
            "train": {family: 40 + index for family in admissions[window_id]},
            "heldout": {
                **{family: 8 if family == "f_new3" else 40 for family in admissions[window_id]},
                **{family: 8 if family == "f_new3" else 40 for family in previous_families},
            },
        }
        for row_index in range(25 + index):
            rows.append({
                "sha256": hashlib.sha256(f"scoring-{index}-{row_index}".encode()).hexdigest(),
                "window_index": index,
                "family": "f_old",
                "split": "heldout",
            })
    value = {
        "schema_version": 1,
        "manifest_sha256": "",
        "window_ids": list(WINDOW_IDS),
        "windows": [{"id": window_id, "months": [window_id]} for window_id in WINDOW_IDS],
        "admissions": admissions,
        "family_admission_index": {family: i for i, window_id in enumerate(WINDOW_IDS) for family in admissions[window_id]},
        "family_support_by_window": support,
        "rows": rows,
    }
    value["manifest_sha256"] = manifest_sha256(value)
    return value


def _records(manifest):
    records = []
    for arm in ARMS:
        for seed in SEEDS:
            rows = []
            for index, window_id in enumerate(WINDOW_IDS):
                scoring_ids = sorted(
                    row["sha256"] for row in manifest["rows"]
                    if row["window_index"] == index and row["split"] == "heldout"
                    and row["family"] in (set(manifest["admissions"][WINDOW_IDS[0]]) if index == 0 else {
                        family for prior_window in WINDOW_IDS[:index]
                        for family in manifest["admissions"][prior_window]
                    })
                )
                score_id_sha = hashlib.sha256("\n".join(scoring_ids).encode("ascii")).hexdigest()
                known = None if index == 0 else _metric(
                    (0.80 if arm == "full_hybrid" else 0.70 if arm == "anchor_only" else 0.60)
                    + index / 100 + seed / 1000,
                    25 + index,
                )
                if known is not None:
                    known["scoring_ids_sha256"] = score_id_sha
                jan = _metric(0.45 + seed / 1000, 25) if index == 0 else None
                if jan is not None:
                    jan["scoring_ids_sha256"] = score_id_sha
                acquisition = None
                if index <= 1:
                    acquisition = {
                        "status": "not_applicable_initial_cohort" if index == 0 else "no_prior_admission",
                        "macro_f1": None, "sample_count": 0,
                        "per_family_sample_counts": {}, "per_family_f1": {},
                    }
                else:
                    families = manifest["admissions"][WINDOW_IDS[index - 1]]
                    family_counts = {
                        family: manifest["family_support_by_window"][window_id]["heldout"].get(family, 0)
                        for family in families
                    }
                    count = sum(family_counts.values())
                    acquisition = {
                        "status": "exploratory_sparse_support" if any(v < 30 for v in family_counts.values()) else "measured",
                        "macro_f1": 0.25 + index / 100,
                        "sample_count": count,
                        "scoring_ids_sha256": hashlib.sha256(f"acq-{window_id}".encode()).hexdigest(),
                        "per_family_sample_counts": family_counts,
                        "per_family_f1": {family: 0.25 for family in families},
                    }
                old = _metric(0.50 + index / 100, 25 + index, {"f_old": 25 + index}) if index > 0 else None
                if old is not None:
                    old["scoring_ids_sha256"] = score_id_sha
                rows.append({
                    "window_id": window_id,
                    "window_index": index,
                    "pre_update": known,
                    "january_post_fit": jan,
                    "old_family_retention": old,
                    "new_family_acquisition": acquisition,
                    "scored_sample_count": 25 + index,
                    "heldout_sample_count": 50 + index,
                    "unscored_unknown_sample_count": 25,
                    "scoring_ids_sha256": score_id_sha,
                    "classifier_updates": 7 + index,
                    "planned_pool": {"full_hybrid": 400, "current": 200, "anchor": 100, "generated": 100},
                    "realized_pool": {
                        "total": 200 + (100 if arm in {"full_hybrid", "anchor_only"} else 0) + (75 if arm == "full_hybrid" else 0),
                        "current": 200,
                        "anchor": 100 if arm in {"full_hybrid", "anchor_only"} else 0,
                        "generated": 75 if arm == "full_hybrid" else 0,
                    },
                    "planned_replay": {
                        "anchor": 100 if arm in {"full_hybrid", "anchor_only"} else 0,
                        "generated": 100 if arm == "full_hybrid" else 0,
                    },
                    "realized_replay": {
                        "anchor": 100 if arm in {"full_hybrid", "anchor_only"} else 0,
                        "generated": 75 if arm == "full_hybrid" else 0,
                    },
                    "generated_replay_output_count": 100 if arm == "full_hybrid" else 0,
                    "gan_compute_seconds": 1.0 if arm == "full_hybrid" else 0.0,
                    "gan_wall_seconds": 1.5 if arm == "full_hybrid" else 0.0,
                })
            records.append({
                "schema_version": 1,
                "status": "completed",
                "experiment_id": EXPERIMENT_ID,
                "run_id": f"{arm}_seed{seed}",
                "arm": arm,
                "seed": seed,
                "manifest_sha256": manifest["manifest_sha256"],
                "schedule_sha256": "b" * 64,
                "final_checkpoint_sha256": "c" * 64,
                "window_records": rows,
            })
    return records


class ReportTests(unittest.TestCase):
    def test_primary_is_equal_window_paired_difference_with_each_seed_shown(self):
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as tmp:
            report = _export_report_records(manifest, _records(manifest), tmp)
        primary = report["primary_result"]
        self.assertEqual(primary["window_ids"], list(WINDOW_IDS[1:]))
        self.assertEqual(set(primary["per_seed"]), {str(seed) for seed in SEEDS})
        for seed in SEEDS:
            values = primary["per_seed"][str(seed)]["window_differences"]
            self.assertEqual(len(values), 5)
            expected = sum(values.values()) / 5
            self.assertAlmostEqual(primary["per_seed"][str(seed)]["mean_difference"], expected)
            self.assertAlmostEqual(expected, 0.10)
        self.assertAlmostEqual(primary["mean_of_seed_means"], 0.10)
        self.assertEqual(primary["window_weighting"], "equal")

    def test_january_postfit_is_separate_and_acquisition_reports_sparse_and_october_unavailable(self):
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as tmp:
            report = _export_report_records(manifest, _records(manifest), tmp)
        first = report["runs"][0]["windows"][0]
        self.assertIsNone(first["pre_update_macro_f1"])
        self.assertIsNotNone(first["january_postfit_macro_f1"])
        acquired = report["acquisition"]
        sparse = [row for row in acquired if row["family"] == "f_new3"]
        self.assertTrue(sparse)
        self.assertTrue(all(row["status"] == "exploratory_sparse_support" for row in sparse))
        october = [row for row in acquired if row["admission_window"] == WINDOW_IDS[5]]
        self.assertEqual({row["status"] for row in october}, {"unavailable_no_following_window"})
        self.assertTrue(all(row["sample_count"] is None for row in october))

    def test_report_exports_counts_budget_and_frozen_manifest(self):
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as tmp:
            report = _export_report_records(manifest, _records(manifest), tmp)
            self.assertTrue((Path(tmp) / "temporal_report.json").exists())
            self.assertTrue((Path(tmp) / "window_metrics.csv").exists())
            self.assertTrue((Path(tmp) / "new_family_acquisition.csv").exists())
            self.assertTrue((Path(tmp) / "frozen_split_manifest.json.gz").exists())
        self.assertEqual(len(report["runs"]), 9)
        self.assertEqual(len(report["runs"][0]["windows"]), 6)
        self.assertEqual(report["runs"][0]["windows"][1]["planned_classifier_updates"], 8)
        self.assertEqual(report["runs"][0]["windows"][1]["realized_replay"], {"anchor": 100, "generated": 75})

    def test_rejects_different_scoring_sets_and_incomplete_matrix(self):
        manifest = _manifest()
        rows = _records(manifest)
        rows[0]["window_records"][2]["scoring_ids_sha256"] = "d" * 64
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "scoring[- ]set"):
            _export_report_records(manifest, rows, tmp)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "exactly"):
            _export_report_records(manifest, _records(manifest)[:-1], tmp)

    def test_rejects_anchor_budget_mismatch_between_paired_hybrid_arms(self):
        manifest = _manifest()
        rows = _records(manifest)
        anchor = next(row for row in rows if row["run_id"] == "anchor_only_seed42")
        anchor["window_records"][2]["planned_replay"]["anchor"] += 1
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "anchor budgets differ"):
            _export_report_records(manifest, rows, tmp)


if __name__ == "__main__":
    unittest.main()
