import os
import re

FNAME_RE = re.compile(r"^me_([a-zA-Z]+)_\d+_\d+\.npz$")

for f in os.listdir("clips_npz")[:10]:
    m = FNAME_RE.match(f)
    print(f, "->", m.group(1) if m else "NO MATCH")