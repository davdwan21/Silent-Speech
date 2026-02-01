import numpy as np, glob
p = glob.glob("clips_npz/*.npz")[0]
z = np.load(p)
print(p, z.files)
for k in z.files:
    print(k, z[k].shape)
