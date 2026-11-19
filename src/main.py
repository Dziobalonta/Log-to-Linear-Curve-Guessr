from pathlib import Path

import cv2
import numpy as np
import OpenEXR

def read_exr(path: Path) -> np.ndarray:
    """EXR -> float32 numpy array (Height, Width, [R,G,B])"""

    with OpenEXR.File(str(path)) as f:
        # Reads the header to learn the size and channels
        # Reads all the rows using the offset table
        # Rearranges the values from planar B, G, R into one NumPy array where each pixel's three values sit together, in R, G, B order
        # np.array a[0, 0] = [0.33381, 0.31467, 0.26377]
        return f.channels()["RGB"].pixels

log_img = read_exr(Path("test_img/bmpcc6kg2/log/A001_01050605_C463_log.exr"))
lin_img = read_exr(Path("test_img/bmpcc6kg2/lin/A001_01050605_C463_lin.exr"))


print(type(log_img))     # <class 'numpy.ndarray'>
print(log_img.shape)     # (3456, 6144, 3)
print(log_img.dtype)     # float32
print(log_img[0, 1])     # top-left pixel: 3 numbers (R, G, B)
print(log_img[0, 0, 1])  # just its green value
print(log_img.max())     # brightest value in the whole image

# 1. Flatten both images into lists of pixels: (21233664, 3)
log_all = log_img.reshape(-1, 3)
lin_all = lin_img.reshape(-1, 3)

# 2. Keep only "good" pixels
good = (log_all.min(axis=1) > 0.1) & (log_all.max(axis=1) < 0.95) & (lin_all.min(axis=1) > 0)

# 3. Pick 5000 of them at random
rng = np.random.default_rng(0)
idx = rng.choice(np.flatnonzero(good), size=5000, replace=False)
# Picking the same pixels from both images, creating smaller copy
log_px = log_all[idx]
lin_px = lin_all[idx]

# Converting to match to match OpenCV's ccm module, expects float64 and shape (N, 1, 3)
src = log_px.reshape(-1, 1, 3).astype(np.float64)

# OpenCV expects reference colours between 0 and 1
# Preparing scaling factor
SCALE = lin_px.max() * 1.01
# Converting and scaling
ref = (lin_px / SCALE).reshape(-1, 1, 3).astype(np.float64)

# Build the model
model = cv2.ccm.ColorCorrectionModel(src, ref, cv2.ccm.COLOR_SPACE_SRGBL)
model.setColorSpace(cv2.ccm.COLOR_SPACE_SRGB)
model.setLinearization(cv2.ccm.LINEARIZATION_COLORLOGPOLYFIT)
model.setLinearizationDegree(7)

# 4. Fit, then read back the linearized values
model.compute()
result = model.getSrcLinearRGB().reshape(-1, 3) * SCALE   # undo the scaling

err = np.abs(np.log2(result / lin_px))   # 0 = perfect, 1 = off by a factor of 2

print("Max error:", err.max())