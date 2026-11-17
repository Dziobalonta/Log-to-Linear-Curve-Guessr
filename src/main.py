from pathlib import Path

import cv2
import numpy as np
import OpenEXR

def read_exr(path: Path) -> np.ndarray:
    """EXR -> float32 array (H, W, 3), RGB order."""

    with OpenEXR.File(str(path)) as f:
        return f.channels()["RGB"].pixels

img = read_exr(Path("test_img/bmpcc6kg2/lin/A001_01050605_C463_lin.exr"))

print(type(img))     # <class 'numpy.ndarray'>
print(img.shape)     # (3456, 6144, 3)
print(img.dtype)     # float32
print(img[0, 1])     # top-left pixel: 3 numbers (R, G, B)
print(img[0, 0, 1])  # just its green value
print(img.max())     # brightest value in the whole image