# Log to Linear curve recovery

Given a bracket of log-encoded photos of the same scene with known exposure times, the program **guesses the camera's log -> linear transfer function** without knowing the manufacturer's formula. It then outputs:

- the recovered curve as a **polynomial** per colour channel (R, G, B),
- the middle exposure converted to **linear** (EXR) and to **gamma 2.2** (16-bit TIFF).

Built with OpenCV: `cv2.createCalibrateDebevec` for the curve, `cv2.ccm.ColorCorrectionModel` for the polynomial fit.

## How it works

1. **Load** all images in the folder (PNG, else TIFF, else EXR), converted to 8-bit, since Debevec only accepts 8-bit input.
2. **Exposure times** are read from metadata (TIFF EXIF `ExposureTime`, EXR shutter angle + fps), or from `exposures.txt` when the files have none.
3. **Debevec** compares the same pixels across exposures and recovers the response curve as a 256-entry lookup table (one row per 8-bit level, relative scale).
4. **Polynomial fit** (OpenCV ccm, `LINEARIZATION_COLORPOLYFIT`, degree 7) on the measured part of the table (levels 40–183):
   `linear = a0 + a1·y + a2·y² + … + a7·y⁷`, where `y` is the log value (0–1).
5. The **middle exposure** (assumed EV 0) is linearized with the lookup table and normalized so its median is 0.18 (middle grey). It is saved as linear EXR and as `clip(linear, 0, 1)^(1/2.2)`.

## Usage

```bash
uv sync
uv run src/main.py <folder with images>
# e.g.
uv run src/main.py test_img/bmpcc6kg2/bracketing/tiff
```

Output in `output/`: `model.yml` (OpenCV model), `<name>_lin.exr`, `<name>_g22.tif`.

If the images carry no exposure metadata, put an `exposures.txt` in the folder, one value per line in file-name order:

```
# shutter speeds
1/25
1/30
...
# or shutter angles (needs fps)
fps 25
180deg
```

## Test data and results

`analysis.ipynb` runs the same pipeline and compares the result against a reference linear EXR and the official formula (plots, error in stops, error map).

| Camera | Bracket | Check | Median error |
|---|---|---|---|
| Blackmagic Pocket 6K G2, Film Gen 5 | 11 shots, 1/25–1/2000 | vs DaVinci Resolve linear EXR | ≈ 0.03 stops |
| DJI Osmo Action 4, log profile | 19 shots, 1/25–1/1600 | self-consistency (shots 1 stop apart must differ ×2) | ≈ 0.007 stops |

Limits: the curve is only measured between sensor noise (dark end) and sensor clip (Blackmagic: log 0.718 = level 183). Outside that range it is extrapolated. The brightness scale is relative (Debevec cannot know absolute exposure).

Reference: *Blackmagic Generation 5 Color Science Technical Reference* (`doc/`), Inverse OETF.
