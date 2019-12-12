import pytest

from landlab import RasterModelGrid


@pytest.fixture
def tc():
    grid = RasterModelGrid((10, 10), spacing=25)
    grid.add_zeros('flow__depth', at='node')
    grid.add_zeros('topographic__elevation', at='node')
    grid.add_zeros('flow__horizontal_velocity', at='link')
    grid.add_zeros('flow__vertical_velocity', at='link')
    grid.add_zeros('flow__sediment_concentration', at='node')
    return tc(grid, cf=0.004, h_init=0.001)
