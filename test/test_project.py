import numpy as np
import camtools as ct
import open3d as o3d


def test_points_to_pixels():
    K = np.array(
        [
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1],
        ]
    )
    R = ct.convert.roll_pitch_yaw_to_R(
        roll=np.pi / 6,
        pitch=-np.pi / 4,
        yaw=np.pi / 3,
    )
    # Position camera closer to origin
    t = np.array([0.1, 0.2, 0.3])
    T = ct.convert.R_t_to_T(R, t)

    # Plot cameras
    coords = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    frustum = ct.camera.create_camera_frustums(
        Ks=[K],
        Ts=[T],
        size=0.2,
        up_triangle=True,
    )
    o3d.visualization.draw_geometries([coords, frustum])
