"""Measure the declined statistic file's value, through the frozen runner.

runner.run reloads solution/features.py from disk each call, so the variant has
to be written to disk. solution/ is ALWAYS restored, including on Ctrl-C.
"""
import shutil, statistics as st, subprocess, sys
from pathlib import Path
ROOT = Path("/Users/zhisong/TikTok TechJam 2026")
SP = Path("/private/tmp/claude-501/-Users-zhisong-TikTok-TechJam-2026/9f148176-2dc4-426f-9bce-86ba4c910874/scratchpad")
sys.path.insert(0, str(ROOT))
D = str(ROOT / "KuaiRand/KuaiRand-Pure/data")

def restore():
    subprocess.run(["git", "checkout", "--", "solution/features.py"], cwd=ROOT, check=False)
    print("solution/features.py restored", flush=True)

from solution import runner
try:
    out = {}
    for label, src in (("reference (official 5 fields)", SP / "features_reference.py"),
                       ("+ platform statistic fields",   SP / "features_with_stat.py")):
        shutil.copy2(src, ROOT / "solution/features.py")
        vals = []
        for seed in range(3):
            r = runner.run(D, config={"seed": seed}, seed=seed)
            if r["status"] != "ok":
                print(f"  seed {seed} FAILED: {r['error'].splitlines()[0][:120]}", flush=True)
                continue
            vals.append(float(r["metrics"]["primary"]))
            print(f"  {label} seed {seed}: {vals[-1]:.4f}", flush=True)
        out[label] = vals
        print(f"{label:32s} mean {st.mean(vals):.4f}\n", flush=True)
    a, b = out["reference (official 5 fields)"], out["+ platform statistic fields"]
    print(f"DELTA from the declined file: {st.mean(b) - st.mean(a):+.4f}")
finally:
    restore()
