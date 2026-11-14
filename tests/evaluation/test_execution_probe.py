"""Diagnostic selection fails closed and cannot export source results."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "execution_probe", Path(__file__).resolve().parents[2] / "scripts" / "probe_grade_execution.py"
)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class ExecutionProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.path = self.root / "results/runs/run/completions/test__base__n16.jsonl"
        self.path.parent.mkdir(parents=True)
        self.row = {"item_id": "q1", "sample": 13, "text": "PRIVATE SQL"}
        self.path.write_text(json.dumps(self.row) + "\n", encoding="utf-8")
        self.review = {
            "drift_diagnostics": [
                {
                    "source": "run/test__base__n16",
                    "differences": [{"item_id": "q1", "sample": 13, "fields": {"executed": {}}}],
                }
            ]
        }

    def test_original_bytes_are_preserved_and_hashed(self) -> None:
        before = self.path.read_bytes()
        cases = PROBE.select_cases(self.root, self.review)
        self.assertEqual(cases[0]["completion"], self.row)
        self.assertEqual(len(cases[0]["source_sha256"]), 64)
        self.assertEqual(self.path.read_bytes(), before)

    def test_missing_case_refuses(self) -> None:
        self.review["drift_diagnostics"][0]["differences"][0]["sample"] = 12
        with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
            PROBE.select_cases(self.root, self.review)

    def test_ambiguous_case_refuses(self) -> None:
        self.path.write_text((json.dumps(self.row) + "\n") * 2, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing or ambiguous"):
            PROBE.select_cases(self.root, self.review)

    def test_duplicate_diagnostic_refuses(self) -> None:
        self.review["drift_diagnostics"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            PROBE.select_cases(self.root, self.review)

    def test_bad_source_refuses_before_access(self) -> None:
        for source in ("../bad", "a/../bad", "a\\b/file", "a/", "/file"):
            with self.subTest(source=source):
                self.review["drift_diagnostics"][0]["source"] = source
                with self.assertRaisesRegex(ValueError, "Invalid"):
                    PROBE.select_cases(self.root, self.review)

    def test_empty_diagnostic_is_not_a_success(self) -> None:
        with self.assertRaisesRegex(ValueError, "No drifted"):
            PROBE.select_cases(self.root, {"drift_diagnostics": []})

    def test_execution_export_excludes_rows_and_sql(self) -> None:
        result = SimpleNamespace(
            executed=False,
            error="timeout",
            elapsed_ms=5000,
            row_count=0,
            truncated=False,
            rows=[["private"]],
            sql="private",
        )
        public = PROBE.summarize_execution(result)
        self.assertEqual(set(public), {"executed", "error", "elapsed_ms", "row_count", "truncated"})
        self.assertNotIn("private", json.dumps(public))


if __name__ == "__main__":
    unittest.main()
