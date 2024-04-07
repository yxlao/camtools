import numpy as np
import camtools as ct

np.set_printoptions(formatter={"float": "{: 0.2f}".format})


def test_R_t_to_cameracenter():
    T = np.array(
        [
            [0.132521, 0.00567408, 0.991163, 0.0228366],
            [-0.709094, -0.698155, 0.0988047, 0.535268],
            [0.692546, -0.715923, -0.0884969, 16.0856],
            [0, 0, 0, 1],
        ]
    )
    R = T[:3, :3]
    t = T[:3, 3]
    expected_camera_center = [-10.7635, 11.8896, 1.348]
    camera_center = ct.convert.R_t_to_C(R, t)
    np.testing.assert_allclose(
        expected_camera_center, camera_center, rtol=1e-5, atol=1e-5
    )


def test_P_to_K_R_t():
    def P_to_K_R_t_manual(P):
        """
        https://ros-developer.com/tag/decomposeprojectionmatrix/
        https://www.cnblogs.com/shengguang/p/5932522.html
        """

        def HouseHolderQR(A):
            num_rows = A.shape[0]
            num_cols = A.shape[1]
            assert num_rows >= num_cols

            R = np.copy(A)
            Q = np.eye(num_rows)

            for col in range(num_cols):
                A_prime = R[col:num_rows, col:num_cols]
                y = A_prime[:, 0].reshape((-1, 1))
                y_norm = np.linalg.norm(y)
                e1 = np.eye(y.shape[0], 1)
                w = y + np.sign(y[0, 0]) * y_norm * e1
                w = w / np.linalg.norm(w)
                I = np.eye(A_prime.shape[0])
                I_2VVT = I - 2 * (w @ w.T)
                H = np.eye(num_rows)
                H[col:num_rows, col:num_rows] = I_2VVT
                R = H @ R
                Q = Q @ H

            return Q, R

        H_inf3x3 = P[:3, :3]
        h3x1 = P[:3, 3].reshape((3, 1))
        H_inf3x3_inv = np.linalg.inv(H_inf3x3)
        # np.linalg.qr(H_inf3x3_inv) may have different signs
        Q, R = HouseHolderQR(H_inf3x3_inv)
        K = np.linalg.inv(R)
        K = K / K[2, 2]
        R = -np.linalg.inv(Q)
        t = -1.0 * (-np.linalg.inv(Q) @ (-np.linalg.inv(H_inf3x3) @ h3x1))

        return K, R, t.squeeze()

    # Ground truth parameters
    focal_length = 1.5
    height = 600
    width = 800
    sensor_height = 10
    sensor_width = 10

    # Ground truth matrices
    K = np.array(
        [
            [focal_length * width / sensor_width, 0, width / 2],
            [0, focal_length * height / sensor_height, height / 2],
            [0, 0, 1],
        ]
    )
    R = ct.convert.roll_pitch_yaw_to_R(np.pi / 4, np.pi / 10, -np.pi / 6)
    t = np.array([[1.0], [2.1], [-1.4]]).squeeze()

    # World-to-pixel projection
    P = K @ np.hstack([R, t.reshape((3, 1))])

    # OpenCV decpomposition
    K_opencv, R_opencv, t_opencv = ct.convert.P_to_K_R_t(P)
    np.testing.assert_allclose(K, K_opencv, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(R, R_opencv, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(t, t_opencv, rtol=1e-5, atol=1e-5)

    # Manual decpomposition
    K_manual, R_manual, t_manual = P_to_K_R_t_manual(P)
    np.testing.assert_allclose(K, K_manual, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(R, R_manual, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(t, t_manual, rtol=1e-5, atol=1e-5)

    # Print info
    # print(f"> Camera {K.shape}:\n{K}")
    # print(f"> Rotation {R.shape}:\n{R}")
    # print(f"> Translation {t.shape}:\n{t}")
    # print(f"> Projection {P.shape}:\n{P}")


def test_convert_pose_opencv_opengl():

    def gen_random_pose():
        axis = np.random.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(0, 2 * np.pi)
        # Skew-symmetric matrix
        ss = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        RT = np.eye(3) + np.sin(angle) * ss + (1 - np.cos(angle)) * np.dot(ss, ss)
        c = np.random.uniform(-10, 10, size=(3,))
        pose = np.eye(4)
        pose[:3, :3] = RT
        pose[:3, 3] = c

        return pose

    for _ in range(10):
        pose = gen_random_pose()
        T = ct.convert.pose_to_T(pose)

        # Test convert pose bidirectionally
        pose_cv = np.copy(pose)
        pose_gl = ct.convert.pose_opencv_to_opengl(pose_cv)
        pose_cv_recovered = ct.convert.pose_opengl_to_opencv(pose_gl)
        pose_gl_recovered = ct.convert.pose_opencv_to_opengl(pose_cv_recovered)
        np.testing.assert_allclose(pose_cv, pose_cv_recovered, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(pose_gl, pose_gl_recovered, rtol=1e-5, atol=1e-5)

        # Test convert T bidirectionally
        T_cv = np.copy(T)
        T_gl = ct.convert.T_opencv_to_opengl(T_cv)
        T_cv_recovered = ct.convert.T_opengl_to_opencv(T_gl)
        T_gl_recovered = ct.convert.T_opencv_to_opengl(T_cv_recovered)
        np.testing.assert_allclose(T_cv, T_cv_recovered, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(T_gl, T_gl_recovered, rtol=1e-5, atol=1e-5)

        # Test T and pose are consistent across conversions
        np.testing.assert_allclose(
            pose_cv,
            ct.convert.T_to_pose(T_cv),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            pose_gl,
            ct.convert.T_to_pose(T_gl),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            pose_cv_recovered,
            ct.convert.T_to_pose(T_cv_recovered),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            pose_gl_recovered,
            ct.convert.T_to_pose(T_gl_recovered),
            rtol=1e-5,
            atol=1e-5,
        )


def test_convert_T_opencv_to_opengl():

    def gen_random_T():
        R = ct.convert.roll_pitch_yaw_to_R(
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-np.pi, np.pi),
        )
        t = np.random.uniform(-10, 10, size=(3,))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    for _ in range(10):
        T = gen_random_T()
        pose = ct.convert.T_to_pose(T)

        # Test convert T bidirectionally
        T_cv = np.copy(T)
        T_gl = ct.convert.T_opencv_to_opengl(T_cv)
        T_cv_recovered = ct.convert.T_opengl_to_opencv(T_gl)
        T_gl_recovered = ct.convert.T_opencv_to_opengl(T_cv_recovered)
        np.testing.assert_allclose(T_cv, T_cv_recovered, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(T_gl, T_gl_recovered, rtol=1e-5, atol=1e-5)

        # Test convert pose bidirectionally
        pose_cv = np.copy(pose)
        pose_gl = ct.convert.pose_opencv_to_opengl(pose_cv)
        pose_cv_recovered = ct.convert.pose_opengl_to_opencv(pose_gl)
        pose_gl_recovered = ct.convert.pose_opencv_to_opengl(pose_cv_recovered)
        np.testing.assert_allclose(pose_cv, pose_cv_recovered, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(pose_gl, pose_gl_recovered, rtol=1e-5, atol=1e-5)

        # Test T and pose are consistent across conversions
        np.testing.assert_allclose(
            T_cv,
            ct.convert.pose_to_T(pose_cv),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            T_gl,
            ct.convert.pose_to_T(pose_gl),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            T_cv_recovered,
            ct.convert.pose_to_T(pose_cv_recovered),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            T_gl_recovered,
            ct.convert.pose_to_T(pose_gl_recovered),
            rtol=1e-5,
            atol=1e-5,
        )
