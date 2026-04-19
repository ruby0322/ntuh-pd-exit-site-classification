import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "summarize_results.py"


class SummarizeResultsCliTest(unittest.TestCase):
    def test_generates_frontier_summary_and_markdown_brief(self):
        sample_tsv = textwrap.dedent(
            """\
            commit\tmc_acc\tbin_acc\tmemory_gb\tstatus\tdescription
            aaa1111\t0.286017\t0.811441\t0.3\tkeep\te01_baseline_sgd
            bbb2222\t0.675847\t0.936441\t0.5\tkeep\te30_mnv3_pos125
            ccc3333\t0.635593\t0.951271\t0.5\tkeep\te34_mnv3_backbone15e4_pos125
            ddd4444\t0.701271\t0.944915\t0.5\tdiscard\te41_no_vflip
            eee5555\t0.692797\t0.917373\t0.5\tdiscard\te44_schednone
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            results_path = tmpdir_path / "results.tsv"
            json_path = tmpdir_path / "analysis_summary.json"
            md_path = tmpdir_path / "analysis_summary.md"
            results_path.write_text(sample_tsv, encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--results",
                    str(results_path),
                    "--json-out",
                    str(json_path),
                    "--md-out",
                    str(md_path),
                ],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )

            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())

            summary = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["current_best"]["description"], "e34_mnv3_backbone15e4_pos125")
            self.assertAlmostEqual(summary["current_best"]["bin_acc"], 0.951271)
            self.assertEqual(summary["best_mc_side_run"]["description"], "e41_no_vflip")
            self.assertEqual(summary["best_mc_side_run"]["status"], "DISCARD")
            self.assertEqual(summary["counts"]["keep"], 3)
            self.assertEqual(summary["counts"]["discard"], 2)
            self.assertEqual(len(summary["frontier_history"]), 3)
            self.assertEqual(summary["promising_near_misses"][0]["description"], "e41_no_vflip")
            self.assertTrue(
                any("schednone" in hint for hint in summary["next_hypothesis_hints"]),
                msg=summary["next_hypothesis_hints"],
            )

            markdown = md_path.read_text(encoding="utf-8")
            self.assertIn("Current screening frontier", markdown)
            self.assertIn("e34_mnv3_backbone15e4_pos125", markdown)
            self.assertIn("e41_no_vflip", markdown)


if __name__ == "__main__":
    unittest.main()
