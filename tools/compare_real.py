"""006 vs 008 on real validation records only (normal + dense). Usage: --script tools/compare_real.py"""
import subprocess
import sys

subprocess.run([sys.executable, "/root/tools/compare_models.py", "--real", "runs/006_ambiguity-bigger-model",
                "runs/008_synthetic-decoys"], check=True)
