from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RobotMoonAtomicWrapperTests(unittest.TestCase):
    def test_failed_converter_does_not_replace_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.binpack"
            output = root / "output.cbin"
            converter = root / "failing_converter.py"
            source.write_bytes(b"input")
            output.write_bytes(b"known-good")
            converter.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "sys.stdout.buffer.write(b'partial')\n"
                "raise SystemExit(7)\n",
                encoding="utf-8",
            )
            converter.chmod(0o755)

            process = subprocess.run(
                [
                    sys.executable,
                    "tools/robotmoon_binpack_zst_to_cbin.py",
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--converter",
                    str(converter),
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(process.returncode, 7)
            self.assertEqual(output.read_bytes(), b"known-good")
            self.assertFalse((root / "output.cbin.tmp").exists())


if __name__ == "__main__":
    unittest.main()
