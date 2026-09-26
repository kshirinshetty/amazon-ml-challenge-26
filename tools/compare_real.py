"""Compare runs on real validation records only (normal + dense), using the current features.
Usage: modal run modal_app.py --script tools/compare_real.py   (edit RUNS)"""
import subprocess
import sys

RUNS = ["runs/010_multipass-synth", "runs/012_keys-synth-safecap"]
subprocess.run([sys.executable, "/root/tools/compare_models.py", "--real", *RUNS], check=True)
