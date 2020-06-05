"""This is a script to run the model of TurbidityCurrent2D
"""
import os
os.environ['MKL_NUM_THREADS'] = '6'
os.environ['OMP_NUM_THREADS'] = '6'
import numpy as np
from turb2d import create_topography
from turb2d import create_init_flow_region, create_topography_from_geotiff
from landlab import RasterModelGrid
from turb2d import TurbidityCurrent2D
import time
from landlab import FIXED_GRADIENT_BOUNDARY, FIXED_VALUE_BOUNDARY
from tqdm import tqdm

# grid = create_topography(
#     length=8000,
#     width=2000,
#     spacing=10,
#     slope_outside=0.2,
#     slope_inside=0.05,
#     slope_basin_break=1000,  #2000
#     canyon_basin_break=1200,  #2200
#     canyon_center=1000,
#     canyon_half_width=100,
# )
grid = create_topography_from_geotiff('depth500.tif',
                                      xlim=[200, 800],
                                      ylim=[400, 1200],
                                      spacing=500)

grid.status_at_node[grid.nodes_at_top_edge] = FIXED_GRADIENT_BOUNDARY
grid.status_at_node[grid.nodes_at_bottom_edge] = FIXED_GRADIENT_BOUNDARY
grid.status_at_node[grid.nodes_at_left_edge] = FIXED_GRADIENT_BOUNDARY
grid.status_at_node[grid.nodes_at_right_edge] = FIXED_GRADIENT_BOUNDARY

# inlet = np.where((grid.x_of_node > 800)
#                  & (grid.x_of_node < 1200) & (grid.y_of_node > 4970))
# inlet_link = np.where((grid.x_of_link > 800) & (grid.x_of_link < 1200)
#                       & (grid.y_of_link > 4970))

# grid.at_node['flow__depth'][inlet] = 30.0
# grid.at_node['flow__sediment_concentration'][inlet] = 0.01
# grid.at_node['flow__horizontal_velocity_at_node'][inlet] = 0.0
# grid.at_node['flow__vertical_velocity_at_node'][inlet] = -3.0
# grid.at_link['flow__horizontal_velocity'][inlet_link] = 0.0
# grid.at_link['flow__vertical_velocity'][inlet_link] = -3.0

# create_init_flow_region(
#     grid,
#     initial_flow_concentration=0.01,
#     initial_flow_thickness=100,
#     initial_region_radius=100,
#     initial_region_center=[1000, 4000],
# )

create_init_flow_region(
    grid,
    initial_flow_concentration=0.01,
    initial_flow_thickness=200,
    initial_region_radius=30000,
    initial_region_center=[100000, 125000],
)

# making turbidity current object
tc = TurbidityCurrent2D(
    grid,
    Cf=0.004,
    alpha=0.2,
    kappa=0.01,
    Ds=80 * 10**-6,
    h_init=0.0,
    p_w=10**(-5),
    C_init=0.0,
    implicit_num=100,
    r0=1.5,
    water_entrainment=False,
    suspension=True,
)

# start calculation
t = time.time()
tc.save_nc('tc{:04d}.nc'.format(0))
Ch_init = np.sum(tc.Ch)
last = 300

for i in tqdm(range(1, last + 1), disable=False):
    tc.run_one_step(dt=100.0)
    tc.save_nc('tc{:04d}.nc'.format(i))
    if np.sum(tc.Ch) / Ch_init < 0.01:
        break
tc.save_grid('tc{:04d}.nc'.format(i))
print('elapsed time: {} sec.'.format(time.time() - t))
