# CamTools

Tools for handling pinhole camera parameters and plotting cameras.

## Build

```bash
mkdir build && cd build
cmake ..
make pip-install -j$(nproc)
python -c "import camtools as ct; print(ct.__version__)"
```

## Matrix conventions

```
K        : (3, 3) # Camera intrinsic matrix.
                  # [[fx,  s, cx],
                  #  [ 0, fy, cy],
                  #  [ 0,  0,  1]]
                  # x: goes from top-left to top-right.
                  # y: goes from top-left to bottom-left.
R        : (3, 3) # Rotation matrix.
Rc       : (3, 3) # Rc = R.T = R.inv().
t        : (3,)   # Translation.
T        : (4, 4) # Extrinsic matrix with (0, 0, 0, 1) row below.
                  # T = [R  | t
                  #      0  | 1]
                  # T projects world space coordinate to the camera space
                  # (a.k.a view space or eye space). The camera center in world
                  # space projected by T becomes [0, 0, 0, 1]^T, i.e. the camera
                  # space has its origin at the camera center.
                  # T @ [[|], = [[0],
                  #      [C],    [0],
                  #      [|],    [0],
                  #      [1]]    [1]]
P        : (3, 4) # World-to-pixel projection matrix. P = K @ [R | t] = K @ T[:3, :].
W2P      : (4, 4) # World-to-pixel projection matrix. It is P with (0, 0, 0, 1)
                  # row below. When using W2P @ point_homo, the last
                  # element is always 1, thus it is ignored.
pose     : (4, 4) # Camera pose. pose = T.inv(). pose[:3, :3] = R.T = Rc. pose[:3, 3] = C.
C        : (3,)   # Camera center.
```

## Coordinate conventions

### 3D to 2D projection

Project 3D point `[X, Y, Z, 1]` to 2D `[x, y, 1]` pixel, e.g. with
`pixels = ct.project.points_to_pixel(points, K, T)`.

```python
# 0 -------> 1 (x)
# |
# |
# v (y)

cols = pixels[:, 0]  # cols, width,  x, top-left to top-right
rows = pixels[:, 1]  # rows, height, y, top-left to bottom-left
cols = np.round(cols).astype(np.int32)
rows = np.round(rows).astype(np.int32)
cols[cols >= width] = width - 1
cols[cols < 0] = 0
rows[rows >= height] = height - 1
rows[rows < 0] = 0
```

It can be confusing to use `x, y, u, v`. Prefer `row` and `col`.


### UV coordinates

```python
# OpenGL convention:
# 1 (v)
# ^
# |
# |
# 0 -------> 1 (u)

# The following conversion accounts for pixel size
us = 1 / width *  (0.5 + cols)
vs = 1 / height * (0.5 + (height - rows - 1))
```

## Notes on vector vs. matrix

We choose to use 1D array for vector values like `t` and `C`.  For example, `t`
is of shape `(3, )` instead of `(3, 1)`.

```python
# The `@` operator can be directly used to dot a matrix and a vector
# - If both arguments are 2-D they are multiplied like conventional matrices.
# - If either argument is N-D, N > 2, it is treated as a stack of matrices
#   residing in the last two indexes and broadcast accordingly.
# - If the first argument is 1-D, it is promoted to a matrix by prepending a 1
#   to its dimensions. After matrix multiplication the prepended 1 is removed.
# - If the second argument is 1-D, it is promoted to a matrix by appending a 1
#   to its dimensions. After matrix multiplication the appended 1 is removed.

# t is (3, ) and it is promoted to be (3, 1).
C = - R.T @ t
```

## Unit tests

```bash
pytest . -s
pytest camtools -s
```


## TODO

- Full unit tests
- PyTorch/Numpy wrapper (e.g. with `eagerpy`)
