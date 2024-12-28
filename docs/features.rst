Features
========

What can you do with CamTools?
------------------------------

1. Plot cameras
^^^^^^^^^^^^^^^

Useful for debugging 3D reconstruction and NeRFs!

.. code-block:: python

   import camtools as ct
   import open3d as o3d
   cameras = ct.camera.create_camera_frustums(Ks, Ts)
   o3d.visualization.draw_geometries([cameras])

.. raw:: html

   <p align="center">
      <img src="https://raw.githubusercontent.com/yxlao/camtools/main/camtools/assets/camera_frames.png" width="360" />
   </p>

2. Convert camera parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   pose = ct.convert.T_to_pose(T)     # Convert T to pose
   T    = ct.convert.pose_to_T(pose)  # Convert pose to T
   R, t = ct.convert.T_to_R_t(T)      # Convert T to R and t
   C    = ct.convert.pose_to_C(pose)  # Convert pose to camera center
   K, T = ct.convert.P_to_K_T(P)      # Decompose projection matrix P to K and T
                                      # And more...

3. Projection and ray casting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Project 3D points to pixels.
   pixels = ct.project.points_to_pixel(points, K, T)

   # Back-project depth image to 3D points.
   points = ct.project.im_depth_to_points(im_depth, K, T)

   # Ray cast a triangle mesh to depth image given the camera parameters.
   im_depth = ct.raycast.mesh_to_im_depth(mesh, K, T, height, width)

   # And more...

4. Image and depth I/O
^^^^^^^^^^^^^^^^^^^^^^

Strict type checks and range checks are enforced. The image and depth I/O
APIs are specifically designed to solve the following pain points:

- Is my image of type ``float32`` or ``uint8``?
- Does it have range ``[0, 1]`` or ``[0, 255]``?
- Is it RGB or BGR?
- Does my image have an alpha channel?
- When saving depth image as integer-based ``.png``, is it correctly scaled?

.. code-block:: python

   ct.io.imread()
   ct.io.imwrite()
   ct.io.imread_detph()
   ct.io.imwrite_depth()

5. Command-line tools
^^^^^^^^^^^^^^^^^^^^^

The ``ct`` command runs in terminal:

.. code-block:: bash

   # Crop image boarders.
   ct crop-boarders *.png --pad_pixel 10 --skip_cropped --same_crop

   # Draw synchronized bounding boxes interactively.
   ct draw-bboxes path/to/a.png path/to/b.png

   # For more command-line tools.
   ct --help

.. raw:: html

   <p align="center">
      <img src="https://user-images.githubusercontent.com/1501945/241416210-e11ff3bf-22e6-46c0-8ba0-d177a0015323.png" width="400" />
   </p>

6. And more
^^^^^^^^^^^

- Solve line intersections
- COLMAP tools
- Points normalization
- And more...
