import pytest
import open3d as o3d


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "skip_no_display: skip test when no display is available"
    )


@pytest.fixture
def has_display():
    """F
    Fixture to check if display is available for Open3D visualization.
    """
    try:
        vis = o3d.visualization.Visualizer()
        vis.destroy_window()
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def skip_no_display(request, has_display):
    """
    Automatically skip tests marked with skip_no_display when display isn't
    available.
    """
    if request.node.get_closest_marker("skip_no_display"):
        if not has_display:
            pytest.skip("Test skipped: no display available")


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
