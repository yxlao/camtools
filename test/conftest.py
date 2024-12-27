import pytest
import open3d as o3d


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "skip_no_o3d_display: skip test when no display is available"
    )


@pytest.fixture
def has_o3d_display():
    """
    Fixture to check if Open3D visualization display is available.
    """
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1, height=1, visible=False)
        if vis.get_view_control() is None:
            vis.destroy_window()
            return False
        vis.destroy_window()
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def skip_no_o3d_display(request, has_o3d_display):
    """
    Automatically skip tests marked with skip_no_o3d_display when Open3D visualization
    isn't available.
    """
    if request.node.get_closest_marker("skip_no_o3d_display"):
        if not has_o3d_display:
            pytest.skip("Test skipped: no Open3D display available")


@pytest.fixture
def visualize(request):
    """
    Fixture to control visualization in tests.
    Can be overridden via command line: pytest --visualize
    """
    return request.config.getoption("--visualize")


def pytest_addoption(parser):
    """
    Add visualize option to pytest command line arguments
    """
    parser.addoption(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable visualization during tests",
    )
