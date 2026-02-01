# find_npz_debug.py
import os, glob
import numpy as np

CANDIDATES = [
    "clipes_npz",
    "clips_npz",
    "./clipes_npz",
    "./clips_npz",
]

print("CWD:", os.getcwd())
print("Directory listing (top):", [x for x in os.listdir(".") if not x.startswith(".")][:30])

def try_dir(d):
    paths = sorted(glob.glob(os.path.join(d, "*.npz")))
    print(f"\nDIR='{d}' -> {len(paths)} npz files")
    if paths:
        print("  first 5:", [os.path.basename(p) for p in paths[:5]])
        # try loading one
        z = np.load(paths[0], allow_pickle=True)
        keys = list(z.keys())
        print("  loaded sample OK. keys:", keys)
        if "X" in z:
            print("  X shape:", z["X"].shape, "dtype:", z["X"].dtype)
        if "label" in z:
            print("  label:", str(z["label"]))
    return len(paths)

total = 0
for d in CANDIDATES:
    if os.path.isdir(d):
        total += try_dir(d)
    else:
        print(f"\nDIR='{d}' does not exist")

print("\nTotal found across candidates:", total)