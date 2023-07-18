import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from . import convert
from . import sanity


def ax3d(fig=None):
    # https://github.com/isl-org/StableViewSynthesis/tree/main/co
    if fig is None:
        fig = plt.gcf()
    return fig.add_subplot(111, projection="3d")


# TODO: adjust order of ax for other functions
def plot_points(points, color="k", ax=None, **kwargs):
    sanity.assert_shape_nx3(points)
    if ax is None:
        ax = plt.gca()
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], color=color)


# TODO: adjust order of ax for other functions
def plot_mesh(vertices, triangles, alpha=0.5, ax=None, **kwargs):
    sanity.assert_shape_nx3(vertices)
    sanity.assert_shape_nx3(triangles)

    if ax is None:
        ax = plt.gca()
    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=triangles,
        alpha=alpha,
        shade=True,
    )


def plot_sphere(radius=1, ax=None):
    if ax is None:
        ax = plt.gca()
    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.2, linewidth=0)


def plot_camera(
    R=np.eye(3),
    t=np.zeros((3,)),
    size=25,
    marker_C=".",
    color="b",
    linestyle="-",
    linewidth=0.1,
    label=None,
    txt=None,
    ax=None,
    **kwargs,
):
    # https://github.com/isl-org/StableViewSynthesis/tree/main/co
    if ax is None:
        ax = plt.gca()
    C0 = convert.R_t_to_C(R, t).ravel()
    C1 = (
        C0 + R.T.dot(np.array([[-size], [-size], [3 * size]], dtype=np.float32)).ravel()
    )
    C2 = (
        C0 + R.T.dot(np.array([[-size], [+size], [3 * size]], dtype=np.float32)).ravel()
    )
    C3 = (
        C0 + R.T.dot(np.array([[+size], [+size], [3 * size]], dtype=np.float32)).ravel()
    )
    C4 = (
        C0 + R.T.dot(np.array([[+size], [-size], [3 * size]], dtype=np.float32)).ravel()
    )

    if marker_C != "":
        ax.plot(
            [C0[0]],
            [C0[1]],
            [C0[2]],
            marker=marker_C,
            color=color,
            label=label,
            **kwargs,
        )
    ax.plot(
        [C0[0], C1[0]],
        [C0[1], C1[1]],
        [C0[2], C1[2]],
        color=color,
        label="_nolegend_",
        linestyle=linestyle,
        linewidth=linewidth,
        **kwargs,
    )
    ax.plot(
        [C0[0], C2[0]],
        [C0[1], C2[1]],
        [C0[2], C2[2]],
        color=color,
        label="_nolegend_",
        linestyle=linestyle,
        linewidth=linewidth,
        **kwargs,
    )
    ax.plot(
        [C0[0], C3[0]],
        [C0[1], C3[1]],
        [C0[2], C3[2]],
        color=color,
        label="_nolegend_",
        linestyle=linestyle,
        linewidth=linewidth,
        **kwargs,
    )
    ax.plot(
        [C0[0], C4[0]],
        [C0[1], C4[1]],
        [C0[2], C4[2]],
        color=color,
        label="_nolegend_",
        linestyle=linestyle,
        linewidth=linewidth,
        **kwargs,
    )
    ax.plot(
        [C1[0], C2[0], C3[0], C4[0], C1[0]],
        [C1[1], C2[1], C3[1], C4[1], C1[1]],
        [C1[2], C2[2], C3[2], C4[2], C1[2]],
        color=color,
        label="_nolegend_",
        linestyle=linestyle,
        linewidth=linewidth,
        **kwargs,
    )

    if txt is not None:
        ax.text(*C0, txt)


def plot_cameras(Rs, ts, size=25, linewidth=0.1, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    for idx, (R, t) in enumerate(zip(Rs, ts)):
        plot_camera(
            R=R, t=t, size=size, linewidth=linewidth, txt=f"{idx:02d}", ax=ax, **kwargs
        )


def axis_equal(ax=None):
    # https://github.com/isl-org/StableViewSynthesis/tree/main/co
    if ax is None:
        ax = plt.gca()
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


def axis_label(x="x", y="y", z="z", ax=None):
    # https://github.com/isl-org/StableViewSynthesis/tree/main/co
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
