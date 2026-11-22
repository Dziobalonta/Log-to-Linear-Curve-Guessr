# FILE USED ONLY FOR TESTING THE LIBRARIES AND FUNCTIONS IN OPENCV, NOT NEEDED

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

files = sorted(glob.glob("test_img/bmpcc6kg2/bracketing/png/*.png"))

images = []
for f in files:
    img = cv2.imread(f) # 8-bit, BGR order
    if img is None:
        raise FileNotFoundError(f)
    images.append(img)

print(len(images), "images,", images[0].shape, images[0].dtype)
# 22 images, (3456, 6144, 3) uint8

speeds = [25, 30, 50, 60, 100, 125, 200, 250, 500, 1000, 2000] # S023 - S033
times = np.array([1 / s for s in speeds], dtype=np.float32)

response = cv2.createCalibrateDebevec().process(images, times)

np.set_printoptions(precision=4, suppress=True) # 4 decimals, no 1e-03 notation

print(response.shape, response.dtype) # (256, 1, 3) float32

for level in [0, 30, 50, 80, 100, 128, 150, 183, 200, 255]:
    print(f"level {level:3d} (log {level / 255:.3f}):  B,G,R = {response[level, 0]}")

levels = np.arange(0, 256)
log_values = levels / 255 # input: log value of each row
lin_values = response[levels, 0, :] # output: linear value, (144, 3), B G R
lin_values = lin_values[:, ::-1] # flip to R G B, which ccm expects

SCALE = lin_values.max() * 1.01
src = np.repeat(log_values[:, None], 3, axis=1).reshape(-1, 1, 3).astype(np.float64)
ref = (lin_values / SCALE).reshape(-1, 1, 3).astype(np.float64)

model = cv2.ccm.ColorCorrectionModel(src, ref, cv2.ccm.COLOR_SPACE_SRGBL)
model.setColorSpace(cv2.ccm.COLOR_SPACE_SRGB)
model.setLinearization(cv2.ccm.LINEARIZATION_COLORLOGPOLYFIT)
model.setLinearizationDegree(5)
model.compute()

fs = cv2.FileStorage("model.yml", cv2.FILE_STORAGE_WRITE)
model.write(fs)
fs.release()

fs = cv2.FileStorage("model.yml", cv2.FILE_STORAGE_READ)
linear = fs.getNode("ColorCorrectionModel").getNode("linear")
coefs = [linear.getNode(k).getNode("p").getNode("p").mat().ravel() for k in ("pr", "pg", "pb")]
fs.release()

# Official Blackmagic Film Gen 5 constants
A = 0.08692876065491224
B = 0.005494072432257808
C = 0.5300133392291939
D = 8.283605932402494
E = 0.09246575342465753
LOG_CUT = D * 0.005 + E # = 0.134, below this the curve is a straight line

def gen5_to_linear(y):
    """Convert ONE log value to linear.
    Taken from Blackmagic's official Gen 5 LUTs, documentation."""
    if y >= LOG_CUT:
        return np.exp((y - C) / A) - B
    else:
        return (y - E) / D

log_values = np.linspace(0, 1, 500) # 500 points from 0 to 1
linear_values = [gen5_to_linear(y) for y in log_values]

def poly_to_linear(p, y):
    """Convert ONE log value to linear with an OpenCV polynomial (coefficients a0, ..., a7)."""
    x = np.log(y) # the polynomial works on ln(log value)
    result = 0
    for i, a in enumerate(p): # a0*x^0 + a1*x^1 + ... + a7*x^7
        result += a * x**i
    return np.exp(result) * SCALE # back from ln, undo the scaling

plt.plot(log_values, linear_values, color="gray", linewidth=1, linestyle='dashed', alpha=0.5, label="Gen 5 (official)")

poly_values = np.linspace(1 / 255, 1, 300)     # only the range the fit had data for
for name, color, p in zip("RGB", ["red", "green", "blue"], coefs):
    values = np.array([poly_to_linear(p, y) for y in poly_values])
    k = gen5_to_linear(0.5) / poly_to_linear(p, 0.5)      # unknown scale: match Gen 5 at log 0.5
    plt.plot(poly_values, values * k, color=color, label=f"{name} polynomial")

plt.yscale("log")
plt.legend()
plt.xlabel("log value (from the camera)")
plt.ylabel("linear value (log scale)")
plt.title("Blackmagic Film Gen 5: log -> linear")
plt.grid(alpha=0.3)
plt.show()

# apply the curves to the middle image
middle = len(files) // 2                  # 11 images -> index 5 -> S028
print("middle image:", files[middle])
img = images[middle][:, :, ::-1]          # BGR -> RGB (matplotlib wants RGB)

# Step 1: build two lookup tables, 256 rows x 3 channels
lut_poly = np.zeros((256, 3), dtype=np.float32)
lut_gen5 = np.zeros((256, 3), dtype=np.float32)
for level in range(1, 256):               # skip 0, because ln(0) doesn't exist (row 0 stays 0)
    y = level / 255
    for c in range(3):                    # 0=R, 1=G, 2=B
        k = gen5_to_linear(0.5) / poly_to_linear(coefs[c], 0.5)
        lut_poly[level, c] = poly_to_linear(coefs[c], y) * k
        lut_gen5[level, c] = gen5_to_linear(y)

# Step 2: replace every pixel with its row from the table
lin_poly = np.zeros(img.shape, dtype=np.float32)
lin_gen5 = np.zeros(img.shape, dtype=np.float32)
for c in range(3):
    # for every pixel, use its value as the row number and read that row from the table
    lin_poly[:, :, c] = lut_poly[img[:, :, c], c]
    lin_gen5[:, :, c] = lut_gen5[img[:, :, c], c]

# Step 3: linear -> something a monitor can show
def to_display(lin):
    return np.clip(lin, 0, 1) ** (1 / 2.2)

# Step 4: how different are they? (in stops, only where the curve was measured)
stops = np.log2(lin_poly / lin_gen5)
good = (img > 40) & (img < 183)
print("difference in stops: median", np.median(np.abs(stops[good])), "max", np.abs(stops[good]).max())

fig, ax = plt.subplots(1, 3, figsize=(18, 4))
ax[0].imshow(img);                    ax[0].set_title("log (from the camera)")
ax[1].imshow(to_display(lin_poly));   ax[1].set_title("linear: OpenCV polynomial")
ax[2].imshow(to_display(lin_gen5));   ax[2].set_title("linear: official Gen 5")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.show()