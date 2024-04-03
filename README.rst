TURB2D
========================

This is a code for the simulation of turbidity currents on 2D grids.

---------------
Installation
Step 0. Since this program uses Cython, you need to install some C compiler before installing turb2d. For Windows, you can download Microsoft Visual Studio Community (https://visualstudio.microsoft.com/free-developer-offers/), which provides a free environment for compiling C code.


Step 1. Execute the following command in the downloaded turb2d directory (the directory where this file is located).

pip install turb2d

If you run this command, the installation of GDAL package will often fail. You can use pip to install the appropriate version of GDAL for your environment before installing turb2d, or you can use Anaconda to install GDAL. If you use conda, you may be able to install GDAL
may be installed with the following command.

conda install -c conda-forge gdal=3.6.2

Step 2. If the installation of turb2d is successful, you can run run_turb2d_script.py to try out the calculations. 

python run_turb2d_script.py

Calculation results will be output to tc*.nc files in the NetCDF format, which can be read by visualization software such as Paraview.
---------------
Usage

A simple usage of this program is as follows:

Example 01: Surge-like turbidity current in an artificial submarine canyon
---------
from turb2d import create_topography,
from turb2d import create_init_flow_region,
from turb2d import TurbidityCurrent2D
from landlab.io.native_landlab import save_grid

# First, make a Landlab grid with artificial topography
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

# Next, add the initial flow region on the topography
create_init_flow_region(
    grid,
    initial_flow_concentration=0.01,
    initial_flow_thickness=100,
    initial_region_radius=100,
    initial_region_center=[1000, 4000],
)

# Create an instance of TurbidityCurrent2D
tc = TurbidityCurrent2D(grid,
                        Cf=0.004,
                        alpha=0.1,
                        Ds=80 * 10**-6, # single grain size case
                        # For mixed grain size case
			            # Ds = [250e-6, 125e-6, 63e-6, 32e-6],
                        )

# Save the initial condition to a NetCDF file which can be read by
# paraview
tc.save_nc('tc{:04d}.nc'.format(0))

# Start Calculation for 10 seconds
for i in range(10):
    tc.run_one_step(dt=1.0)
    tc.save_nc('tc{:04d}.nc'.format(i + 1))
    print("", end="\r")
    print("{:.1f}% finished".format(i / last * 100), end='\r')

# Save the result
save_grid(grid, 'tc{:04d}.nc'.format(i))


Example 02: Use natural topography loaded from the GEOTIFF format file
----------------
from turb2d import create_topography_from_geotiff
from turb2d import create_init_flow_region
from turb2d import TurbidityCurrent2D
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
                        Ds=80 * 10**-6, # Single grain size
			# Ds = [250e-6, 125e-6, 63e-6, 32e-6], # Multiple grain size
                        )

tc.save_nc('tc{:04d}.nc'.format(0))

for i in range(10):
    tc.run_one_step(dt=1.0)
    tc.save_nc('tc{:04d}.nc'.format(i+1))
    print("", end="\r")
    print("{:.1f}% finished".format(i / last * 100), end='\r')
save_grid(grid, 'tc{:04d}.nc'.format(i))


-------------------------
Visualization

Use the visualization software to visualize the calculation results. Paraview can be used for this purpose.


