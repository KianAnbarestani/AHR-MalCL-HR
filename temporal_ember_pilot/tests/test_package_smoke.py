import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

from temporal_ember_pilot.checkpoint import load_checkpoint, save_checkpoint
from temporal_ember_pilot.engine import StopRequested, capture_rng_state, run_window
from temporal_ember_pilot.report import export_report
from temporal_ember_pilot.stream import WINDOWS, WINDOW_IDS, build_manifest, manifest_sha256, window_partitions


class PackageSmokeTests(unittest.TestCase):
    def _manifest(self):
        rows = []
        for index, window in enumerate(WINDOWS):
            families = ["old_family"] + ([] if index == 0 else [f"new_family_{index}"])
            month = window["months"][0]
            for family in families:
                for sample_index in range(120):
                    rows.append({
                        "sha256": hashlib.sha256(f"smoke:{index}:{family}:{sample_index}".encode()).hexdigest(),
                        "appeared": month,
                        "label": 1,
                        "avclass": family,
                    })
        manifest = build_manifest(rows, initial_family_limit=1, admission_family_limit=1)
        self.assertEqual(manifest["admissions"][WINDOW_IDS[0]], ["old_family"])
        return manifest

    def _result_records(self, manifest):
        records = []
        arms = ("full_hybrid", "anchor_only", "current_only")
        seeds = (42, 43, 44)
        lookup = {row["sha256"]: row["family"] for row in manifest["rows"]}
        for arm in arms:
            for seed in seeds:
                window_rows = []
                for index, window_id in enumerate(WINDOW_IDS):
                    parts = window_partitions(manifest, window_id)
                    score_ids = parts["scoring_ids"]
                    digest = hashlib.sha256("\n".join(score_ids).encode("ascii")).hexdigest()
                    score_counts = {family: 0 for family in parts["known_families_after"] if index == 0}
                    if index > 0:
                        score_counts = {family: 0 for family in parts["known_families_before"]}
                    for sample_id in score_ids:
                        score_counts[lookup[sample_id]] += 1
                    metric = {
                        "macro_f1": 0.7 if arm == "full_hybrid" else 0.6,
                        "sample_count": len(score_ids),
                        "scoring_ids_sha256": digest,
                        "per_family_sample_counts": score_counts,
                        "per_family_f1": {family: 0.5 for family in score_counts},
                    }
                    acquisition = {"status": "not_applicable_initial_cohort", "macro_f1": None,
                                   "sample_count": 0, "per_family_sample_counts": {}, "per_family_f1": {}}
                    if index == 1:
                        acquisition["status"] = "no_prior_admission"
                    if index > 1:
                        families = manifest["admissions"][WINDOW_IDS[index - 1]]
                        heldout = manifest["family_support_by_window"][window_id]["heldout"]
                        family_counts = {family: int(heldout.get(family, 0)) for family in families}
                        acquisition = {
                            "status": "exploratory_sparse_support" if any(count < 30 for count in family_counts.values()) else "measured",
                            "macro_f1": 0.5,
                            "sample_count": sum(family_counts.values()),
                            "scoring_ids_sha256": hashlib.sha256(f"acquisition-{window_id}".encode()).hexdigest(),
                            "per_family_sample_counts": family_counts,
                            "per_family_f1": {family: 0.5 for family in families},
                        }
                    old_ids = [sample_id for sample_id in score_ids if lookup[sample_id] == "old_family"]
                    old_metric = None
                    if index > 0:
                        old_metric = {
                            "macro_f1": 0.5,
                            "sample_count": len(old_ids),
                            "scoring_ids_sha256": hashlib.sha256("\n".join(sorted(old_ids)).encode("ascii")).hexdigest(),
                            "per_family_sample_counts": {"old_family": len(old_ids)},
                            "per_family_f1": {"old_family": 0.5},
                        }
                    heldout_count = sum(manifest["family_support_by_window"][window_id]["heldout"].values())
                    window_rows.append({
                        "window_id": window_id,
                        "pre_update": None if index == 0 else metric,
                        "january_post_fit": metric if index == 0 else None,
                        "old_family_retention": old_metric,
                        "new_family_acquisition": acquisition,
                        "scored_sample_count": len(score_ids),
                        "heldout_sample_count": heldout_count,
                        "unscored_unknown_sample_count": heldout_count - len(score_ids),
                        "scoring_ids_sha256": digest,
                        "classifier_updates": 3,
                        "planned_pool": {"full_hybrid": 256, "current": 128, "anchor": 64, "generated": 64},
                        "planned_replay": {"anchor": 64 if arm != "current_only" else 0,
                                           "generated": 64 if arm == "full_hybrid" else 0},
                        "realized_pool": {"total": 220, "current": 128, "anchor": 64, "generated": 28},
                        "realized_replay": {"anchor": 64 if arm != "current_only" else 0,
                                             "generated": 28 if arm == "full_hybrid" else 0},
                        "generated_replay_output_count": 28 if arm == "full_hybrid" else 0,
                        "classifier_sampled_source_counts": {"current": 100, "anchor": 40, "generated": 20},
                        "gan_compute_seconds": 0.01 if arm == "full_hybrid" else 0.0,
                        "gan_wall_seconds": 0.02 if arm == "full_hybrid" else 0.0,
                        "generation_seconds": 0.01 if arm == "full_hybrid" else 0.0,
                    })
                records.append({
                    "status": "completed", "experiment_id": "ahrmalcl_temporal_ember18v2_newfamily_v1",
                    "run_id": f"{arm}_seed{seed}", "arm": arm, "seed": seed,
                    "manifest_sha256": manifest["manifest_sha256"], "schedule_sha256": "a" * 64,
                    "final_checkpoint_sha256": hashlib.sha256(f"checkpoint-{arm}-{seed}".encode()).hexdigest(),
                    "window_records": window_rows,
                })
        return records

    def test_checkpoint_resume_and_report_export_smoke(self):
        import numpy as np
        import torch

        def start_state():
            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            return {
                "stream_cursor": {"next_window_index": 1, "last_completed_window": 0,
                                  "window_id": "2018-01", "phase": "completed", "batch_index": 0},
                "training_state": {"model": model.state_dict()},
                "optimizer_state": {"model": optimizer.state_dict()},
                "scaler_state": {"fit_count": 12},
                "anchor_state": {"ids": ["anchor"]},
                "class_map": {"old_family": 0},
                "gan_state": {"generated": [1.0]},
                "numpy_rng_generators": {},
                "rng_state": capture_rng_state(),
                "step": 0,
            }

        def train_batch(state, window, batch_index):
            model = torch.nn.Linear(2, 1)
            model.load_state_dict(state["training_state"]["model"])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            optimizer.load_state_dict(state["optimizer_state"]["model"])
            x = torch.randn(3, 2)
            y = torch.randn(3, 1)
            loss = torch.nn.functional.mse_loss(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            state["training_state"]["model"] = model.state_dict()
            state["optimizer_state"]["model"] = optimizer.state_dict()
            state["step"] += 1
            return state

        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            boundary = root / "continuous" / "window_0.pkl"
            boundary.parent.mkdir()
            save_checkpoint(boundary, start_state())
            continuous = run_window(load_checkpoint(boundary), {"window_id": WINDOW_IDS[1], "window_index": 1},
                                   updates=3, train_batch=train_batch,
                                   checkpoint_path=boundary.parent / "window_1.pkl")
            resume_dir = root / "resumed"
            resume_dir.mkdir()
            save_checkpoint(resume_dir / "window_0.pkl", start_state())
            with self.assertRaises(StopRequested):
                run_window(load_checkpoint(resume_dir / "window_0.pkl"),
                           {"window_id": WINDOW_IDS[1], "window_index": 1}, updates=3,
                           train_batch=train_batch, checkpoint_path=resume_dir / "window_1.pkl",
                           stop_after_batches=1)
            resumed = run_window(load_checkpoint(resume_dir / "window_0.pkl"),
                                 {"window_id": WINDOW_IDS[1], "window_index": 1}, updates=3,
                                 train_batch=train_batch, checkpoint_path=resume_dir / "window_1.pkl")
            self.assertEqual(continuous["stream_cursor"], resumed["stream_cursor"])
            self.assertEqual(continuous["step"], resumed["step"])
            for key, tensor in continuous["training_state"]["model"].items():
                self.assertTrue(torch.equal(tensor, resumed["training_state"]["model"][key]))
            self.assertEqual(continuous["scaler_state"], resumed["scaler_state"])
            self.assertEqual(continuous["anchor_state"], resumed["anchor_state"])
            self.assertEqual(continuous["class_map"], resumed["class_map"])
            report = export_report(manifest, self._result_records(manifest), root / "report")
            self.assertEqual(report["primary_result"]["window_count"], 5)
            self.assertEqual(len(report["runs"]), 9)
            self.assertTrue((root / "report" / "temporal_report.json").exists())
            self.assertTrue((root / "report" / "frozen_split_manifest.json.gz").exists())

    def test_notebook_has_four_ordered_steps_and_training_disabled(self):
        notebook_path = Path(__file__).parents[1] / "temporal_ember_pilot.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        markdown = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "markdown"]
        headings = [line.strip() for cell in markdown for line in cell.splitlines() if line.startswith("## ")]
        self.assertEqual(headings, ["## 1. Setup", "## 2. Audit", "## 3. Run/Resume", "## 4. Report"])
        code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell.get("source", [])), f"pilot_notebook_cell_{index}", "exec")
        self.assertIn("ENABLE_TRAINING = False", code)
        self.assertIn("CONFIRM_EXPERIMENT_ID", code)
        self.assertIn("ahrmalcl_temporal_ember18v2_newfamily_v1", code)

    def test_vendored_verified_sources_match_their_recorded_hashes(self):
        root = Path(__file__).parents[1]
        source_manifest = json.loads((root / "verified_source_manifest.json").read_text(encoding="utf-8"))
        for entry in source_manifest["files"]:
            vendored_path = entry.get("vendored_path")
            if not vendored_path:
                continue
            digest = hashlib.sha256((Path(__file__).parents[2] / vendored_path).read_bytes()).hexdigest()
            self.assertTrue(entry["byte_identical"])
            self.assertEqual(digest, entry["vendored_sha256"])


if __name__ == "__main__":
    unittest.main()
