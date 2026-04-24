# Court Corner Projection and Court Overlay from Annotation

This document describes a minimal, concrete approach for using the provided annotation JSON to:

1. determine where the court corners land in the image,
2. generate a court mask in image space,
3. overlay court boundary and court lines onto the image.

This document intentionally does **not** cover tracking from video, solving for unknown calibration, or estimating distortion from footage. It only assumes that the JSON annotation already provides camera calibration.

---

## Goal

Given a JSON annotation with camera intrinsics, extrinsics, distortion coefficients, and image size, project known court geometry from court/world coordinates into image coordinates.

This lets you:

- locate the court corners in the image,
- rasterize the court polygon to label court pixels,
- draw any known court markings as projected curves/lines on top of the image.

---

## What is available in the annotation

The annotation provides a calibrated camera, including:

- `KK`: 3x3 intrinsic matrix,
- `R`: 3x3 rotation matrix,
- `T`: 3-vector translation,
- `kc`: distortion parameters,
- `alpha`: skew parameter,
- `img_width`, `img_height`.

For the attached JSON, the calibration block contains for example:

- `KK = [[1863.127930, 0, 811.5], [0, 1863.127930, 616.5], [0, 0, 1]]`
- `R` and `T` defining camera pose,
- `kc = [-0.121767, 0.0, 0.001442, 0.0, 0.0]`
- image size `1624 x 1234`.

Assume the world coordinate frame is already aligned to the court, with the court lying on plane `Z = 0`. The user previously described the origin as a court corner and axes aligned with court edges. The attached JSON supplies the camera parameters needed for projection. Source: `camcourt1_1513710529019.json`. fileciteturn4file0

---

## Court geometry you must define

The JSON gives the camera, not the court dimensions. You still need to define the court in world coordinates.

Use a consistent court coordinate system such as:

- origin at one court corner,
- `+X` along the long side,
- `+Y` along the short side,
- `Z = 0` on the floor plane.

Then define the 4 court corners as:

- `C0 = (0, 0, 0)`
- `C1 = (L, 0, 0)`
- `C2 = (L, W, 0)`
- `C3 = (0, W, 0)`

where `L` is court length and `W` is court width, in the **same units** as the calibration.

If the calibration is in millimeters, then court geometry must also be in millimeters.

Examples:

- NBA / FIBA-sized values depend on the dataset convention.
- Do not hardcode dimensions until you confirm what this dataset uses.

For the first implementation, it is enough to define the outer court rectangle plus any court lines you want to draw.

---

## Projection pipeline

For each world point `Pw = (X, Y, Z)`:

### 1. Transform world point into camera coordinates

Use the extrinsics:

```text
Pc = R * Pw + T
```

with:

```text
Pc = (Xc, Yc, Zc)
```

### 2. Perspective divide into normalized image coordinates

```text
x = Xc / Zc
y = Yc / Zc
```

These are ideal pinhole coordinates before distortion.

### 3. Apply lens distortion

Interpret `kc` in the common 5-parameter form:

```text
kc = [k1, k2, p1, p2, k3]
```

Compute:

```text
r2 = x^2 + y^2
radial = 1 + k1*r2 + k2*r2^2 + k3*r2^3
```

Then distorted normalized coordinates are:

```text
xd = x * radial + 2*p1*x*y + p2*(r2 + 2*x^2)
yd = y * radial + p1*(r2 + 2*y^2) + 2*p2*x*y
```

For this JSON:

```text
k1 = -0.121767
k2 = 0
p1 = 0.001442
p2 = 0
k3 = 0
```

### 4. Apply intrinsics to get pixels

If skew is zero or negligible:

```text
u = fx * xd + cx
v = fy * yd + cy
```

Using the provided `KK`:

```text
fx = 1863.127930
fy = 1863.127930
cx = 811.5
cy = 616.5
```

If you want to include skew `alpha` or a nonzero off-diagonal intrinsic term, use the full matrix multiplication instead.

---

## Court corners in the image

Once the 4 court corners are defined in world coordinates, project each corner using the pipeline above.

This yields 4 image points:

- `c0 = project(C0)`
- `c1 = project(C1)`
- `c2 = project(C2)`
- `c3 = project(C3)`

These are the image-space court corners.

Use them to:

- draw the court boundary polygon,
- fill the polygon to label court pixels,
- sanity-check that the calibration and court dimensions are correct.

---

## Court pixel labeling

To label the court pixels:

1. project the 4 court corners into the image,
2. form a polygon from those 4 projected points,
3. rasterize/fill the polygon into a binary mask.

The result is a mask where:

- `1` means the pixel corresponds to the court interior,
- `0` means outside the court rectangle.

Notes:

- This mask represents the projected court area, not visible wood only.
- Players, referees, and other objects may occlude the court in the image, but the geometric mask still marks where the court lies.
- Some projected points may fall outside the image bounds; clip the polygon to the image if needed.

---

## Overlaying court lines

To overlay court markings, define each marking in court/world coordinates and project sampled points along it.

### Straight lines

For a court line segment from `A` to `B`, define:

```text
P(t) = A + t * (B - A),   t in [0, 1]
```

Sample many values of `t`, project each point, then draw the resulting polyline in the image.

Examples:

- sidelines,
- baselines,
- half-court line,
- lane rectangle edges,
- free-throw line.

### Curved markings

For arcs/circles, parameterize them in court coordinates, sample points densely, project each sample, and draw the resulting polyline.

Examples:

- center circle,
- free-throw circle,
- three-point arc.

This works because the world-space geometry is known and the projection model already includes distortion.

---

## Why sampling is fine for overlay

Even though straight world lines have an analytic image projection under the camera model, numerically it is simplest to:

- sample world points along the court marking,
- project those points,
- draw the projected polyline.

This is accurate enough for visualization and mask creation as long as sampling is dense enough.

---

## Homography versus full camera projection

Because the court lies on a plane, you could also use a plane homography once the image is undistorted.

For the current task, do **not** use homography as the primary implementation. Use the full forward camera model directly because:

- the annotation already provides `K`, `R`, `T`, and `kc`,
- distortion is already part of the annotation,
- the direct model is the most faithful way to overlay the court on the raw image.

A homography can be derived later if you need court-to-image mapping on an undistorted image.

---

## Minimal implementation plan

### Inputs

- annotation JSON,
- chosen court dimensions and court coordinate convention,
- image.

### Outputs

- projected image-space court corners,
- binary court mask,
- overlay image with projected court lines.

### Steps

1. Parse `KK`, `R`, `T`, `kc`, `img_width`, `img_height`.
2. Build world-space court geometry.
3. Implement `project_world_to_image(Pw)` using extrinsics, perspective divide, distortion, and intrinsics.
4. Project the 4 outer court corners.
5. Fill the projected quadrilateral to obtain a court mask.
6. Parameterize court markings and project dense samples for each.
7. Draw the projected polylines on top of the image.

---

## Recommended API shape

```python
class CameraCalibration:
    K: np.ndarray      # 3x3
    R: np.ndarray      # 3x3
    T: np.ndarray      # 3,
    kc: np.ndarray     # 5,
    width: int
    height: int


def project_world_to_image(Pw, calib) -> np.ndarray:
    ...


def make_court_corners(L, W):
    return np.array([
        [0, 0, 0],
        [L, 0, 0],
        [L, W, 0],
        [0, W, 0],
    ], dtype=float)


def make_court_mask(image_shape, projected_corners):
    ...


def sample_segment(A, B, n=200):
    ...


def sample_circle_arc(center, radius, theta0, theta1, n=300):
    ...
```

---

## Important sanity checks

Before using the overlay downstream, verify:

1. **Unit consistency**  
   Court dimensions must use the same units as `T` and camera position.

2. **Axis convention**  
   Make sure the chosen world `X/Y` directions match the annotation's court frame.

3. **Corner ordering**  
   Use a consistent polygon order when filling the court mask.

4. **Projection visibility**  
   Some court geometry may project outside the frame; this is normal.

5. **Distortion enabled**  
   If the overlay looks slightly off near the edges, confirm the distortion step is included.

---

## Non-goals for this document

This document does not yet cover:

- recovering calibration from video,
- solving for homography from tracked features,
- estimating distortion from court lines,
- tracking players or projecting player footpoints to the court,
- undistorting the full image.

Those can be added later after the direct overlay path is working.

---

## Suggested first deliverable

Implement a script that:

1. loads one image and its matching JSON annotation,
2. projects the 4 outer court corners,
3. draws the court boundary polygon,
4. fills a court mask,
5. overlays at least these markings:
   - outer boundary,
   - half-court line,
   - center circle,
   - lane rectangle,
6. writes both the mask and overlay image to disk.

Once that works, it becomes the geometric base for later tracking work.
