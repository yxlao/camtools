import pytest
import open3d as o3d


@pytest.fixture
def has_display():
    """Check if display is available for Open3D visualization."""
    try:
        # Try to create a visualizer to check if display is available
        vis = o3d.visualization.Visualizer()
        vis.destroy_window()
        return True
    except Exception:
        return False


@pytest.fixture
def visualize(request):
    """Fixture to control visualization in tests.
    Can be overridden via command line: pytest --visualize
    """
    return request.config.getoption("--visualize")


def pytest_addoption(parser):
    """Add visualize option to pytest command line arguments"""
    parser.addoption(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable visualization during tests",
    )
