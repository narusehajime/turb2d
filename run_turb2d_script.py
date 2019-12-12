"""This is a script to run the model of TurbidityCurrent2D
"""

from turb2d import create_topography
from turb2d import create_init_flow_region, create_topography_from_geotiff
from turb2d import TurbidityCurrent2D
import time
from landlab.io.netcdf import write_netcdf, read_netcdf
from landlab.io.native_landlab import load_grid, save_grid
import os
import numpy as np

import ipdb
ipdb.set_trace()
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'

grid = create_topography(
    length=5000,
    width=2000,
    spacing=10,
    slope_outside=0.2,
    slope_inside=0.05,
    slope_basin_break=2000,
    canyon_basin_break=2200,
    canyon_center=1000,
    canyon_half_width=100,
)
# grid = create_topography_from_geotiff('depth500.tif',
#                                       xlim=[200, 800],
#                                       ylim=[400, 1200],
#                                       spacing=500)

create_init_flow_region(
    grid,
    initial_flow_concentration=0.01,
    initial_flow_thickness=100,
    initial_region_radius=100,
    initial_region_center=[1000, 4000],
)
# create_init_flow_region(
#     grid,
#     initial_flow_concentration=0.01,
#     initial_flow_thickness=500,
#     initial_region_radius=30000,
#     initial_region_center=[200000, 125000],
# )

# making turbidity current object
tc = TurbidityCurrent2D(grid,
                        Cf=0.004,
                        alpha=0.1,
                        kappa=0.25,
                        Ds=80 * 10**-6,
                        h_init=0.00001,
                        h_w=0.01,
                        C_init=0.00001,
                        implicit_num=20,
                        r0=1.5)

# start calculation
t = time.time()
write_netcdf('tc{:04d}.nc'.format(0), grid)
Ch_init = np.sum(tc.Ch)
last = 4

for i in range(6, last + 1):
    tc.run_one_step(dt=1.0)
    write_netcdf('tc{:04d}.nc'.format(i), grid)
    print("", end="\r")
    print("{:.1f}% finished".format(i / last * 100), end='\r')
    if np.sum(tc.Ch) / Ch_init < 0.01:
        break
save_grid(grid, 'tc{:04d}.nc'.format(i))
print('elapsed time: {} sec.'.format(time.time() - t))