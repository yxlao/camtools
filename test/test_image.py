import numpy as np
import camtools as ct

np.set_printoptions(formatter={"float": "{: 0.2f}".format})


def test_recover_rotated_pixels():
    w = 5
    h = 4

    src_cr = np.array([[3, 1]])
    rot0_cr = np.array([[3, 1]])
    rot90_cr = np.array([[1, 1]])
    rot180_cr = np.array([[1, 2]])
    rot270_cr = np.array([[2, 3]])

    np.testing.assert_allclose(
        src_cr, ct.image.recover_rotated_pixels(rot0_cr, (w, h), 0)
    )
    np.testing.assert_allclose(
        src_cr, ct.image.recover_rotated_pixels(rot90_cr, (w, h), 90)
    )
    np.testing.assert_allclose(
        src_cr, ct.image.recover_rotated_pixels(rot180_cr, (w, h), 180)
    )
    np.testing.assert_allclose(
        src_cr, ct.image.recover_rotated_pixels(rot270_cr, (w, h), 270)
    )

    src_cr = np.array([[3, 2]])
    rot0_cr = np.array([[3, 2]])
    rot90_cr = np.array([[2, 1]])
    rot180_cr = np.array([[1, 1]])
    rot270_cr = np.array([[1, 3]])
    np.testing.assert_allclose(
        src_cr, ct.image.recover_rotated_pixels(rot0_cr, (w, h), 0)
    )
    np.testing.assert_allclose(
        src_cr, ct.image.recover_rotated_pixels(rot90_cr, (w, h), 90)
    )
    np.testing.assert_allclose(
        src_cr, ct.image.recover_rotated_pixels(rot180_cr, (w, h), 180)
    )
    np.testing.assert_allclose(
        src_cr, ct.image.recover_rotated_pixels(rot270_cr, (w, h), 270)
    )

    src_cr = np.array([[0, 0]])
    rot0_cr = np.array([[0, 0]])
    rot90_cr = np.array([[0, 4]])
    rot180_cr = np.array([[4, 3]])
    rot270_cr = np.array([[3, 0]])
    np.testing.assert_allclose(
        src_cr, ct.image.recover_rotated_pixels(rot0_cr, (w, h), 0)
    )
    np.testing.assert_allclose(
        src_cr, ct.image.recover_rotated_pixels(rot90_cr, (w, h), 90)
    )
    np.testing.assert_allclose(
        src_cr, ct.image.recover_rotated_pixels(rot180_cr, (w, h), 180)
    )
    np.testing.assert_allclose(
        src_cr, ct.image.recover_rotated_pixels(rot270_cr, (w, h), 270)
    )
