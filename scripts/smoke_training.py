"""Execute the disposable-fixture training-engine tests; never an ML experiment."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
result = subprocess.run([sys.executable, "-m", "unittest", "tests.test_training_engine"], cwd=root)
raise SystemExit(result.returncode)
