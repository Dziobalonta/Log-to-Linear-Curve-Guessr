# FILE USED ONLY FOR TESTING THE LIBRARIES AND FUNCTIONS IN OPENCV, NOT NEEDED

from pathlib import Path
import cv2
import numpy as np
import OpenEXR
import matplotlib.pyplot as plt

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

# Flatten both images into lists of pixels: (21233664, 3)
log_all = log_img.reshape(-1, 3)
lin_all = lin_img.reshape(-1, 3)

# Keep only "good" pixels
good = (log_all.min(axis=1) > 0.1) & (log_all.max(axis=1) < 0.95) & (lin_all.min(axis=1) > 0)

# Pick 5000 of them at random
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

# Fit, then read back the linearized values
model.compute()
result = model.getSrcLinearRGB().reshape(-1, 3) * SCALE   # undo the scaling

# Save the fitted model to a file
fs = cv2.FileStorage("model.yml", cv2.FILE_STORAGE_WRITE)
model.write(fs)
fs.release()

# Read the three polynomials back
fs = cv2.FileStorage("model.yml", cv2.FILE_STORAGE_READ)
linear = fs.getNode("ColorCorrectionModel").getNode("linear")
coefs = [linear.getNode(k).getNode("p").getNode("p").mat().ravel() for k in ("pr", "pg", "pb")]
fs.release()

np.set_printoptions(precision=4, suppress=True)   # 4 decimals, no 1e+02 notation
# Print the polynomials in a human-readable form
for name, p in zip("RGB", coefs):
    terms = " + ".join(f"{a:.4f}·x^{i}" for i, a in enumerate(p))
    print(f"{name}(x) = {terms}")

err = np.abs(np.log2(result / lin_px))   # 0 = perfect, 1 = off by a factor of 2

print("Max error:", err.max())

# PLOTTING

# Official Blackmagic Film Gen 5 constants
A = 0.08692876065491224
B = 0.005494072432257808
C = 0.5300133392291939
D = 8.283605932402494
E = 0.09246575342465753
LOG_CUT = D * 0.005 + E          # = 0.134, below this the curve is a straight line

def gen5_to_linear(y):
    """Convert ONE log value to linear.
    Taken from Blackmagic's official Gen 5 LUTs, documentation."""
    if y >= LOG_CUT:
        return np.exp((y - C) / A) - B
    else:
        return (y - E) / D

log_values = np.linspace(0, 1, 500)          # 500 points from 0 to 1
linear_values = [gen5_to_linear(y) for y in log_values]

def poly_to_linear(p, y):
    """Convert ONE log value to linear with an OpenCV polynomial (coefficients a0, ..., a7)."""
    x = np.log(y)                    # the polynomial works on ln(log value)
    result = 0
    for i, a in enumerate(p):        # a0*x^0 + a1*x^1 + ... + a7*x^7
        result += a * x**i
    return np.exp(result) * SCALE    # back from ln, undo the scaling

plt.plot(log_values, linear_values, color="gray", linewidth=1, linestyle='dashed', label="Gen 5 (official)")

poly_values = np.linspace(0.1, 1, 500)      # start at 0.1, because ln(0) doesn't exist
for name, color, p in zip("RGB", ["red", "green", "blue"], coefs):
    values = [poly_to_linear(p, y) for y in poly_values]
    plt.plot(poly_values, values, color=color, label=f"{name} polynomial")

plt.ylim(0, 250)
plt.legend()
plt.xlabel("log value (from the camera)")
plt.ylabel("linear value")
plt.title("Blackmagic Film Gen 5: log -> linear")
plt.grid(alpha=0.3)
plt.show()

# APPLYING THE MODEL TO AN IMAGE
def gen5_to_linear_image(img):
    """Official Gen 5 for a WHOLE image: same formula, np.where instead of if."""
    return np.where(img >= LOG_CUT, np.exp((img - C) / A) - B, (img - E) / D)

def poly_to_linear_image(img):
    """OpenCV polynomials for a WHOLE image: one polynomial per channel."""
    out = np.empty(img.shape, dtype=np.float32)
    for c in range(3):                                   # 0 = R, 1 = G, 2 = B
        channel = np.clip(img[:, :, c], 1e-6, None)      # ln needs values > 0
        out[:, :, c] = poly_to_linear(coefs[c], channel)
    return out

lin_official = gen5_to_linear_image(log_img)   # ~0.3 s
lin_poly = poly_to_linear_image(log_img)       # ~4 s

# Show them side by side
def to_display(lin):
    """Linear values look too dark on a screen, so clip to 0..1 and apply gamma 2.2."""
    return np.clip(lin, 0, 1) ** (1 / 2.2)

diff = np.abs(np.log2(np.clip(lin_poly, 1e-6, None) / np.clip(lin_official, 1e-6, None))).max(axis=2)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(to_display(lin_official)); axes[0].set_title("Official Gen 5 formula")
axes[1].imshow(to_display(lin_poly));     axes[1].set_title("OpenCV polynomial")
im = axes[2].imshow(diff, cmap="magma", vmin=0, vmax=0.1); axes[2].set_title("Difference (stops)")
fig.colorbar(im, ax=axes[2])
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
