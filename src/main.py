import sys
import os
import glob
import cv2
import numpy as np
import OpenEXR
from PIL import Image

EXPOSURE_FILE = "exposures.txt"  # optional file in the image folder

# ---------- Program Argument ----------

def read_folder_argument():
    """Returns the folder given on the command line, e.g. python src/main.py test_img/.../png"""

    if len(sys.argv) != 2:
        print("usage: python main.py <folder with images>")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print("not a folder:", folder)
        sys.exit(1)

    return folder

# ---------- Finding the Images ----------

def find_images(folder):
    """Return a sorted list of image files. PNG first, then TIFF, then EXR"""

    for extensions in [["png"], ["tif", "tiff"], ["exr"]]:

        files = []
        for ext in extensions:
            files += glob.glob(os.path.join(folder, "*." + ext))
            files += glob.glob(os.path.join(folder, "*." + ext.upper()))

        if len(files) > 0:
            return sorted(files)

    raise FileNotFoundError("no PNG, TIFF or EXR files in " + folder)

# ---------- Loading and Converting (to 8-bit) ----------

def read_exr(path):
    """Read an EXR file. Returns float32 array (Height, Width, [R,G,B])"""

    with OpenEXR.File(path) as f:
        channels = f.channels()

        if "RGB" not in channels:
            raise ValueError("no RGB channels in " + path)
        
        return channels["RGB"].pixels

def Convert_to_8bit(path):
    """Load one image and return it as 8-bit BGR"""
    ext = path.lower().split(".")[-1]

    if ext == "exr":
        img = read_exr(path) # float 0..1, RGB
        img = np.clip(img, 0, 1)
        img = np.round(img * 255).astype(np.uint8)
        img = img[:, :, ::-1] # RGB -> BGR, same as cv2.imread
        return np.ascontiguousarray(img)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED) # keep 16-bit if it is 16-bit
    if img is None:
        raise FileNotFoundError(path)

    if img.dtype == np.uint16: 
        img = np.round(img / 257).astype(np.uint8) # mapping values to 0-255

    if img.shape[2] == 4: # drop alpha if there is one
        img = img[:, :, :3]
    return img


def load_images(files):

    images = []

    for f in files:
        images.append(Convert_to_8bit(f))
        print("loaded", os.path.basename(f))

    return images

# ---------- Prepping Exposure Times ----------

def angle_to_seconds(angle, fps):
    """Convert shutter angle to seconds. 360 degrees = 1 second."""
    return (angle / 360.0) / fps

def exposure_from_metadata(path):
    """Return exposure time in seconds, or None if the file doesn't say."""
    ext = path.lower().split(".")[-1]

    if ext in ["tif", "tiff"]:
        exif = Image.open(path).getexif()
        exif_ifd = exif.get_ifd(0x8769) # the "Exif" sub-section
        value = exif_ifd.get(33434) # 33434 = ExposureTime
        if value is not None:
            return float(value)

    if ext == "exr":
        with OpenEXR.File(path) as f:
            header = f.header()
            angle =  header.get("camera_shutter_angle")
            fps = header.get("framesPerSecond")
            if angle is not None and fps is not None:
                angle = float(angle.replace("°", "")) # strip the degree symbol
                return angle_to_seconds(angle, float(fps))

    return None # Nothing found

def read_exposure_file(path):
    """Read exposures.txt. One value per line, in the same order as the sorted files:
         1/125      -> shutter speed in seconds
         180deg     -> shutter angle (then you also need a line 'fps 25')
       Lines starting with # are comments."""

    fps = None
    values = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            if line.startswith("fps"): # saving fps info
                fps = float(line.split()[1])
            else:
                values.append(line)

    times = []
    for v in values:
        if v.endswith("deg"): # when using degrees

            if fps is None:
                raise ValueError("Shutter angles need fps to to be converted to shutter speed.\n \
                Please append a line like 'fps 25' in " + path)

            times.append(angle_to_seconds(float(v[:-3]), fps))
        
        elif "/" in v: # when using shutter speed
            top, bottom = v.split("/")
            times.append(float(top) / float(bottom))
        else:
            times.append(float(v))

    return times

def get_exposure_times(folder, files):
    """Gets exposure for every file in folder. First it Tries the metadata first. If any file has none, uses exposures.txt"""
    times = []
    for f in files:
        times.append(exposure_from_metadata(f))

    if None not in times:
        print("exposure times from metadata")
    else:
        path = os.path.join(folder, EXPOSURE_FILE)

        if not os.path.isfile(path):
            raise FileNotFoundError("no exposure time in metadata and no " + path)
        
        times = read_exposure_file(path)
        print("exposure times from", path)

    if len(times) != len(files): # file amount and entries in file file could differ
        raise ValueError(f"{len(files)} images but {len(times)} exposure times")

    for f, t in zip(files, times):
        print(f"{os.path.basename(f)}  1/{1 / t:.0f} s")
    return np.array(times, dtype=np.float32)

# ---------- Creating color correction ----------
def calculate_correction(images, times):
    """Calculates LOG curve correction to linear using Debevec. Returns (256, 1, 3) float32, B G R Lookup Table"""

    return cv2.createCalibrateDebevec().process(images, times)

# ---------- Fitting a polynomial (OpenCV ccm) ----------

def fit_polynomial(lut, model_path, degree=7):
    """Fit OpenCV's plain polynomial to Debevec's table and save the model.
    linear = a0 + a1*y + a2*y^2 + ...,  y = log value (0..1)
    Returns {"R": [a0, a1, ...], "G": ..., "B": ...}, the scale is already included."""

    # <40 - deep shadows mostly noise
    # Gen 5's sensor clip it at log 0.718, which is level 183 in 8 bits
    levels = np.arange(40,184) 
    log_values = levels / 255 # input: log value of each row
    lin_values = lut[levels, 0, :] # output: linear value, B G R 
    lin_values = lin_values[:,::-1] # flip to R G B, which ccm expects

    scale = lin_values.max() * 1.01        # ccm wants values below 1
    src = np.repeat(log_values[:, None], 3, axis=1).reshape(-1, 1, 3).astype(np.float64)
    ref = (lin_values / scale).reshape(-1, 1, 3).astype(np.float64)

    model = cv2.ccm.ColorCorrectionModel(src, ref, cv2.ccm.COLOR_SPACE_SRGBL)
    model.setColorSpace(cv2.ccm.COLOR_SPACE_SRGB)
    model.setLinearization(cv2.ccm.LINEARIZATION_COLORPOLYFIT)
    model.setLinearizationDegree(degree)
    model.compute()

    # save the model
    fs = cv2.FileStorage(model_path, cv2.FILE_STORAGE_WRITE)
    model.write(fs)
    fs.release()

    # read the coefficients back from the saved file
    fs = cv2.FileStorage(model_path, cv2.FILE_STORAGE_READ)
    linear = fs.getNode("ColorCorrectionModel").getNode("linear")
    coefs = {}
    for name, key in [("R", "pr"), ("G", "pg"), ("B", "pb")]:
        coefs[name] = linear.getNode(key).getNode("p").mat().ravel() * scale # undo shrinking
    fs.release()

    return coefs

def print_equation(name, p):
    """Print the polynomial of one colour as an equation."""
    
    x = "y"

    text = f"{p[0]:.4f}"
    for i in range(1, len(p)):
        if p[i] >= 0:
            text += f" + {p[i]:.4f}*{x}^{i}"
        else:
            text += f" - {-p[i]:.4f}*{x}^{i}"

    print(f"{name}: linear = {text}")

# ---------- Applying color correction ----------

def choose_middle_image(times):
    """Return the index of the middle exposure (probably EV 0)"""
    order = np.argsort(times) # indexes sorted from shortest to longest time exposure time
    return order[len(order) // 2] # the one in the middle

def apply_correction(img, lut):
    """8-bit BGR image -> linear float32 BGR, using Debevec's lookup table."""

    linear = np.zeros(img.shape, dtype=np.float32)

    for c in range(3):# B, G, R
        # Needs to be in this bizzare way cuz numpy reads it faster than loops

        # for all the rows and colums pefrom this respectively for B, G, R
        # lut is the Debevec's response. It's just packaging, The middle one has size 1. There's only one item there, number 0. OpenCV adds it because it stores the table as a "1-pixel-wide image"

        linear[:, :, c] = lut[img[:, :, c], 0, c] # each pixel value = row number in the table
    return linear

def auto_exposure(linear):
    """Debevec doesn't know the real brightness. Make it middle grey(the median pixel = 0.18)"""
    return linear * (0.18 / np.median(linear))

# ---------- Saving ----------

def save_linear_exr(path, linear_bgr):
    """Save linear float data as an EXR (float32, ZIP compressed)."""
    rgb = np.ascontiguousarray(linear_bgr[:, :, ::-1])     # BGR -> RGB, EXR stores RGB
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, {"RGB": rgb}) as f:
        f.write(path)


def save_gamma22_tiff(path, linear_bgr):
    """Linear -> gamma 2.2 -> 16-bit TIFF, for viewing."""
    gamma = np.clip(linear_bgr, 0, 1) ** (1 / 2.2)         # cut above 1, then the power
    cv2.imwrite(path, np.round(gamma * 65535).astype(np.uint16))

# ---------- Main Function ----------
def main():
    folder = read_folder_argument()
    files = find_images(folder)
    print(len(files), "images found")

    times = get_exposure_times(folder, files)
    print()
    images = load_images(files)
    print(f"Converted to {images[0].shape}, {images[0].dtype}")
    print()

    print("Performing calculations...")
    lut = calculate_correction(images, times)
    print("lut:", lut.shape, lut.dtype)
    print()

    os.makedirs("output", exist_ok=True)
    print("Conversion function to linear image:")
    coefs = fit_polynomial(lut, os.path.join("output", "model.yml"))
    for name in "RGB":
        print_equation(name, coefs[name])
    print()

    middle = choose_middle_image(times)
    print("Applying correction for middle EV image:", os.path.basename(files[middle]))

    linear = auto_exposure(apply_correction(images[middle], lut))

    os.makedirs("output", exist_ok=True)
    name = os.path.splitext(os.path.basename(files[middle]))[0]   # "A001_01032221_S028" [1] - extension .tif
    save_linear_exr(os.path.join("output", name + "_lin.exr"), linear)
    save_gamma22_tiff(os.path.join("output", name + "_g22.tif"), linear)
    print("saved to output/")


if __name__ == "__main__":
    main()