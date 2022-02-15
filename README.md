# CamTools

Tools for handling pinhole camera parameters and plotting cameras.

## Conventions

```
K        : (3, 3) # Camera matrix.
R        : (3, 3) # Rotation matrix.
Rc       : (3, 3) # Rc = R.T = R.inv().
t        : (3,)   # Translation.
T        : (4, 4) # Extrinsic matrix with (0, 0, 0, 1) row below.
                  # T = [R  | t
                  #      0  | 1]
                  #
                  # T projects world space coordinate to the camera space
                  # (a.k.a view space or eye space). The camera center in world
                  # space projected by T becomes [0, 0, 0, 1]^T, i.e. the camera
                  # space has its origin at the camera center.
                  # T @  [|  = [0
                  #       C     0
                  #       |     0
                  #       1]    1]
P        : (3, 4) # World-to-pixel projection matrix. P = K @ [R | t] = K @ T[:3, :].
world_mat: (4, 4) # World-to-pixel projection matrix. It is P with (0, 0, 0, 1)
                  # row below. When using world_mat @ point_homo, the last
                  # element is always 1, thus it is ignored.
pose     : (4, 4) # Camera pose. pose = T.inv(). pose[:3, :3] = R.T = Rc. pose[:3, 3] = C.
C        : (3,)   # Camera center.
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
