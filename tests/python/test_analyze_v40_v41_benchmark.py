import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "analyze" / "analyze_v40_v41_benchmark.py"
SPEC = importlib.util.spec_from_file_location("v40_v41_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AnalyzeV40V41BenchmarkTests(unittest.TestCase):
    def test_parse_and_summarize_paired_rounds(self) -> None:
        text = """\
round=0 version=v40 nodes=100 us=100 nps=1000000
round=0 version=v41 nodes=90 us=75 nps=1200000
round=1 version=v40 nodes=100 us=110 nps=909090
round=1 version=v41 nodes=90 us=90 nps=1000000
summary version=v40 depth=7
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.log"
            path.write_text(text, encoding="utf-8")
            rounds = MODULE.parse_rounds(path)

        result = MODULE.summarize(rounds, replicates=100, seed=7)
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["bootstrap_seed"], 7)
        self.assertEqual(result["v40"]["nodes"], 200)
        self.assertEqual(result["v41"]["nodes"], 180)
        self.assertAlmostEqual(result["node_ratio"], 0.9)
        self.assertAlmostEqual(result["time_ratio"], 165 / 210)
        self.assertAlmostEqual(result["nps_ratio"], (180 / 165) / (200 / 210))

    def test_rejects_incomplete_round(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.log"
            path.write_text(
                "round=0 version=v40 nodes=100 us=100 nps=1000000\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not a complete pair"):
                MODULE.parse_rounds(path)


if __name__ == "__main__":
    unittest.main()
