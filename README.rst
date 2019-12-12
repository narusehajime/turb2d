TURB2D
========================

This is a code for simulation of turbidity currents on 2D grids.

---------------
Installation

Here is an example of installation command. 

python setup.py install

Please note that Visual C++ or other cpp compiler is installed in your environemnts if you use windows. This is because this code uses Cython.

---------------
Usage

A simple usage of this program is as follows:

Example 01: Surge-like turbidity current in artificial canyon
---------
from turb2d import create_topography,
from turb2d import create_init_flow_region,
from turb2d import TurbidityCurrent2D
from landlab.io.netcdf import write_netcdf
from landlab.io.native_landlab import save_grid

# First, make a landlab grid with artificial topography
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

# Next, add initial flow region on the topography
create_init_flow_region(
    grid,
    initial_flow_concentration=0.01,
    initial_flow_thickness=100,
    initial_region_radius=100,
    initial_region_center=[1000, 4000],
)

# Create instance of TurbidityCurrent2D
tc = TurbidityCurrent2D(grid,
                        Cf=0.004,
                        alpha=0.1,
                        Ds=80 * 10**-6,
                        )

# Save the initial condition to a netcdf file which can be read by
# paraview
write_netcdf('tc{:04d}.nc'.format(0), grid)

# Start Calculation for 10 seconds
for i in range(10):
    tc.run_one_step(dt=1.0)
    write_netcdf('tc{:04d}.nc'.format(i), grid)
    print("", end="\r")
    print("{:.1f}% finished".format(i / last * 100), end='\r')

# Save the result
save_grid(grid, 'tc{:04d}.nc'.format(i))


Example 02: Use natural topography from GEOTIFF
----------------
from turb2d import create_topography_from_geotiff
from turb2d import create_init_flow_region
from turb2d import TurbidityCurrent2D
from landlab.io.netcdf import write_netcdf
from landlab.io.native_landlab import save_grid

grid = create_topography_from_geotiff('depth500.tif', spacing=500)

create_init_flow_region(
    grid,
    initial_flow_concentration=0.01,
    initial_flow_thickness=500,
    initial_region_radius=30000,
    initial_region_center=[200000, 125000],
)

# making turbidity current object
tc = TurbidityCurrent2D(grid,
                        Cf=0.004,
                        alpha=0.1,
                        Ds=80 * 10**-6,
                        )

write_netcdf('tc{:04d}.nc'.format(0), grid)

for i in range(10):
    tc.run_one_step(dt=1.0)
    write_netcdf('tc{:04d}.nc'.format(i), grid)
    print("", end="\r")
    print("{:.1f}% finished".format(i / last * 100), end='\r')
save_grid(grid, 'tc{:04d}.nc'.format(i))


-------------------------
Limitation and future implementation

Single grain-size only. Boundary condition are always 'open' (no gradient).
