import camtools as ct
import numpy as np
import open3d as o3d
import itertools


def main():
    K = np.array(
        [
            [554.256, 0.0, 320],
            [0.0, 554.256, 240],
            [0.0, 0.0, 1.0],
        ]
    )

    resolution = 12
    radius = 1.0
    thetas = np.linspace(0, np.pi, resolution, endpoint=False)[1:]
    phis = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    Ts = [
        ct.convert.spherical_to_T_towards_origin(radius, theta, phi)
        for (theta, phi) in itertools.product(thetas, phis)
    ]
    Ks = [K for _ in range(len(Ts))]

    cameras = ct.camera.create_camera_ray_frames(Ks=Ks, Ts=Ts, center_line=False)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries([cameras, axes])


if __name__ == "__main__":
    main()
