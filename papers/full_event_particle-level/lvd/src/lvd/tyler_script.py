import numpy as np

npz_filename = "lvd/data/Herwig_Zjet_pTZ-200GeV_7.npz"
name = "Herwig_Zjet_pTZ-200GeV_7"

npy_arr = np.load(npz_filename)

print(f"==== {name} ====\n\tShape: {len(npy_arr.files)}")
print(npy_arr.files + "\n\n")

# create a sample array
