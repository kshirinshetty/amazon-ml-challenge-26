"""Label-free per-country comparison of the submission candidates (run: modal run modal_app.py --script tools/compare.py)."""
import subprocess
import sys

subprocess.run([sys.executable, "/root/tools/france_diag.py", "runs/003_decoy-features/output", "runs/005_deeper-retrieval/output",
                "runs/006_ambiguity-bigger-model/output", "runs/006_ambiguity-bigger-model/output_unseen0.5"], check=True)
