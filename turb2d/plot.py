import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from landlab.plot.imshow import imshow_grid
import numpy as np
from landlab.io.native_landlab import load_grid
import sys


def plot_result(grid, filename, variable_name, vmin=None, vmax=None):
    """Plot calculation results of TurbidityCurrent2D with topography

        Parameters
        -----------------
        grid : RasterModelGrid
            An object of the class RasterModelGrid
        filname : string
            A file name to save the figure
        variable_name : string
            A name of variable to visualize
    """

    plt.clf()

    imshow_grid(
        grid,
        variable_name,
        cmap='PuBu',
        grid_units=('m', 'm'),
        var_name=variable_name,
        var_units='m',
        vmin=vmin,
        vmax=vmax,
    )

    z = grid.at_node['topographic__elevation']

    elev = grid.node_vector_to_raster(z)
    X = grid.node_vector_to_raster(grid.node_x)
    Y = grid.node_vector_to_raster(grid.node_y)
    cs = plt.contour(X,
                     Y,
                     elev,
                     colors=['dimgray'],
                     levels=np.linspace(np.min(z), np.max(z), 10))
    cs.clabel(inline=True, fmt='%1i', fontsize=10)

    plt.savefig(filename)


if __name__ == '__main__':

    variable_name = sys.argv[1]

    for i in range(101):
        grid = load_grid('tc{:04d}.grid'.format(i))
        plot_result(
            grid,
            'tc{:04d}.png'.format(i),
            variable_name,
            vmin=np.min(grid.at_node[variable_name]),
            #     vmax=0.005,
            # )
            vmax=np.max(grid.at_node[variable_name]))
