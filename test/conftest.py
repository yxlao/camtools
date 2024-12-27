import pytest


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
