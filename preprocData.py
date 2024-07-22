import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
nstart = 3100
nend = 3200
gap = 50

# Coarse
istart, iend = 16, 64
jstart, jend = 44, 79
kstart, kend = 21, 98

#fine
# istart, iend = 16, 114
# jstart, jend = 44, 112
# kstart, kend = 21, 174


# Calculate number of datasets
ndataset = (nend - nstart) // gap - 1

# Dimensions
x_dim = kend - kstart  # Swap x and z dimensions
y_dim = jend - jstart
z_dim = iend - istart  # Swap x and z dimensions

# Initialize arrays
data = np.zeros((ndataset, 3, 4, x_dim, y_dim, z_dim))  # 4 for u, v, w, p

# Create a directory for saving images
output_dir = 'midsection_images'
os.makedirs(output_dir, exist_ok=True)

# Loop through datasets
for iDataset, fnameID in enumerate(range(nstart + 2 * gap, nend + gap, gap)):
    print(iDataset, fnameID)
    current_file = f"./out.{fnameID:07d}.hdf5"
    previous_file1 = f"./out.{fnameID - gap:07d}.hdf5"
    previous_file2 = f"./out.{fnameID - 2 * gap:07d}.hdf5"

    for i, file in enumerate([previous_file2, previous_file1, current_file]):
        with h5py.File(file, 'r') as f:
            u = np.swapaxes(np.array(f['u']), 0, 2)[kstart:kend, jstart:jend, istart:iend]
            v = np.swapaxes(np.array(f['v']), 0, 2)[kstart:kend, jstart:jend, istart:iend]
            w = np.swapaxes(np.array(f['w']), 0, 2)[kstart:kend, jstart:jend, istart:iend]
            p = np.swapaxes(np.array(f['p']), 0, 2)[kstart:kend, jstart:jend, istart:iend]

        # Debug prints to check the shapes
        print(f"Shape of u, v, w, p from {file}: {u.shape}, {v.shape}, {w.shape}, {p.shape}")

        # Check for dimension mismatches
        if u.shape != (x_dim, y_dim, z_dim) or v.shape != (x_dim, y_dim, z_dim) or \
           w.shape != (x_dim, y_dim, z_dim) or p.shape != (x_dim, y_dim, z_dim):
            print(f"Dimension mismatch at dataset {iDataset} (file ID {fnameID})")
            continue

        data[iDataset, i, 0] = u
        data[iDataset, i, 1] = v
        data[iDataset, i, 2] = w
        data[iDataset, i, 3] = p

    # Plotting can be added here if needed

# Save arrays
np.save("data.npy", data)
print("Data saved. Shape:", data.shape)
print("Max and min values:", np.max(data), np.min(data))