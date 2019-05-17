TURB2D
========================

This is a code for simulation of turbidity currents on 2D grids.

---------------
Installation

python setup.py install

---------------

Example
---------
from landlab.io.esri_ascii import read_esri_ascii
import matplotlib.pyplot as plt
import numpy as np
from landlab import Component, FieldError, RasterModelGrid
from landlab.plot.imshow import imshow_grid

grid = RasterModelGrid((200, 100), spacing=10.0)
grid.add_zeros('flow__depth', at='node')
grid.add_zeros('topographic__elevation', at='node')
grid.add_zeros('flow__horizontal_velocity', at='link')
grid.add_zeros('flow__vertical_velocity', at='link')
initial_flow_region = (grid.node_x > 400.) & (grid.node_x < 600.) & (
    grid.node_y > 1400.) & (grid.node_y < 1600.)

grid.at_node['flow__depth'][initial_flow_region] = 20.0
grid.at_node['topographic__elevation'][
    grid.node_y > 1000] = (grid.node_y[grid.node_y > 1000] - 1000) * 0.15

dflow = DebrisFlow(
    grid,
    h_init=0.01,
    alpha=0.1, # Change this for stability of calculation
    flow_type='Voellmy', # choose 'water' or 'Voellmy'.
    Cf=0.004, # friction
    basal_friction_angle=0.0875, # kind of yield strength
    )

last = 20
for i in range(last):
    dflow.run_one_step(dt=10.0)
    plt.clf()
    imshow_grid(grid, 'flow__depth', cmap='Blues')
    plt.savefig('dflow{:04d}.png'.format(i))
    print("", end="\r")
    print("{:.1f}% finished".format((i + 1) / (last) * 100), end='\r')

