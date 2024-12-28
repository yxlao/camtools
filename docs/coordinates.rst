Camera Coordinate System
========================

A homogeneous point ``[X, Y, Z, 1]`` in the world coordinate can be projected to a
homogeneous point ``[x, y, 1]`` in the image (pixel) coordinate using the
following equation:

.. math::

   \lambda
   \left[\begin{array}{l}
   x \\
   y \\
   1
   \end{array}\right]=\left[\begin{array}{ccc}
   f_{x} & 0 & c_{x} \\
   0 & f_{y} & c_{y} \\
   0 & 0 & 1
   \end{array}\right]\left[\begin{array}{llll}
   R_{00} & R_{01} & R_{02} & t_{0} \\
   R_{10} & R_{11} & R_{12} & t_{1} \\
   R_{20} & R_{21} & R_{22} & t_{2}
   \end{array}\right]\left[\begin{array}{c}
   X \\
   Y \\
   Z \\
   1
   \end{array}\right].

We follow the standard OpenCV-style camera coordinate system as illustrated at
the beginning of the documentation.

Camera Coordinate
-----------------

Right-handed, with :math:`Z` pointing away from the camera towards the view direction
and :math:`Y` axis pointing down. Note that the OpenCV convention (camtools' default)
is different from the OpenGL/Blender convention, where :math:`Z` points towards the
opposite view direction, :math:`Y` points up and :math:`X` points right.

To convert between the OpenCV camera coordinates and the OpenGL-style coordinates,
use the conversion functions:

- ``ct.convert.T_opencv_to_opengl()``
- ``ct.convert.T_opengl_to_opencv()``
- ``ct.convert.pose_opencv_to_opengl()``
- ``ct.convert.pose_opengl_to_opencv()``

Image Coordinate
----------------

Starts from the top-left corner of the image, with :math:`x` pointing right
(corresponding to the image width) and :math:`y` pointing down (corresponding to
the image height). This is consistent with OpenCV.

Pay attention that the 0th dimension in the image array is the height (i.e., :math:`y`)
and the 1st dimension is the width (i.e., :math:`x`). That is:

- :math:`x` <=> ``u`` <=> width <=> column <=> the 1st dimension
- :math:`y` <=> ``v`` <=> height <=> row <=> the 0th dimension

Matrix Definitions
------------------

Camera Intrinsic Matrix (K)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``K`` is a ``(3, 3)`` camera intrinsic matrix:

.. code-block:: python

   K = [[fx,  s, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]]

Camera Extrinsic Matrix (T or W2C)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``T`` is a ``(4, 4)`` camera extrinsic matrix:

.. code-block:: python

   T = [[R  | t   = [[R00, R01, R02, t0],
         0  | 1]]    [R10, R11, R12, t1],
                     [R20, R21, R22, t2],
                     [  0,   0,   0,  1]]

- ``T`` is also known as the world-to-camera ``W2C`` matrix, which transforms a
  point in the world coordinate to the camera coordinate.
- ``T``'s shape is ``(4, 4)``, not ``(3, 4)``.
- ``T`` is the inverse of ``pose``, i.e., ``np.linalg.inv(T) == pose``.
- The camera center ``C`` in world coordinate is projected to ``[0, 0, 0, 1]`` in
  camera coordinate.

Rotation Matrix (R)
^^^^^^^^^^^^^^^^^^^

``R`` is a ``(3, 3)`` rotation matrix:

.. code-block:: python

   R = T[:3, :3]

- ``R`` is a rotation matrix. It is an orthogonal matrix with determinant 1, as
  rotations preserve volume and orientation.
  - ``R.T == np.linalg.inv(R)``
  - ``np.linalg.norm(R @ x) == np.linalg.norm(x)``, where ``x`` is a ``(3,)``
    vector.

Translation Vector (t)
^^^^^^^^^^^^^^^^^^^^^^

``t`` is a ``(3,)`` translation vector:

.. code-block:: python

   t = T[:3, 3]

- ``t``'s shape is ``(3,)``, not ``(3, 1)``.

Camera Pose Matrix (pose or C2W)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``pose`` is a ``(4, 4)`` camera pose matrix. It is the inverse of ``T``.

- ``pose`` is also known as the camera-to-world ``C2W`` matrix, which transforms a
  point in the camera coordinate to the world coordinate.
- ``pose`` is the inverse of ``T``, i.e., ``pose == np.linalg.inv(T)``.

Camera Center (C)
^^^^^^^^^^^^^^^^^

``C`` is the camera center:

.. code-block:: python

   C = pose[:3, 3]

- ``C``'s shape is ``(3,)``, not ``(3, 1)``.
- ``C`` is the camera center in world coordinate. It is also the translation
  vector of ``pose``.

Projection Matrix (P)
^^^^^^^^^^^^^^^^^^^^^

``P`` is a ``(3, 4)`` camera projection matrix:

- ``P`` is the world-to-pixel projection matrix, which projects a point in the
  homogeneous world coordinate to the homogeneous pixel coordinate.
- ``P`` is the product of the intrinsic and extrinsic parameters:

  .. code-block:: python

    # P = K @ [R | t]
    P = K @ np.hstack([R, t[:, None]])

- ``P``'s shape is ``(3, 4)``, not ``(4, 4)``.
- It is possible to decompose ``P`` into intrinsic and extrinsic matrices by QR
  decomposition.
- Don't confuse ``P`` with ``pose``. Don't confuse ``P`` with ``T``.
