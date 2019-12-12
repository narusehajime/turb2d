import numpy as np
from landlab import Component, FieldError, RasterModelGrid
from landlab.utils.decorators import use_file_name_or_kwds
from landlab.grid.structured_quad import links
from landlab.io.native_landlab import save_grid
from cip import rcip_2d_M_advection, cip_2d_nonadvection
from cip import cip_2d_diffusion, shock_dissipation
from .sediment_func import get_es, get_ew, get_ws

import time
# from osgeo import gdal, gdalconst
import os
"""A component of landlab that simulates a turbidity current on Raster 2D grids

This component simulates turbidity currents using the 2-D numerical model of
shallow-water equations over topography on the basis of 3 equation model of
Parker et al. (1986). This component is based on the landlab component
 overland_flow that was written by Jordan Adams.

.. codeauthor:: Hajime Naruse

Examples
---------
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

"""


class TurbidityCurrent2D(Component):
    """Simulate a turbidity current using the CIP method.

    Landlab component that simulates turbidity current using the CIP method
    for solving the 2D shallow water equations.

    This component calculates flow depth, shear stress across any raster grid.
    Default input file is named "turbidity_current_input.txt" and is
    contained in the turb2d directory.

    The primary method of this class is :func:'run_one_step'
    """

    _name = 'TurbidityCurrent2D'

    _cite_as = """Naruse in prep."""

    _input_var_names = (
        'flow__depth',
        'flow__horizontal_velocity',
        'flow__vertical_velocity',
        'flow__sediment_concentration',
        'topographic__elevation',
        'bed__thickness',
        'flow__surface_elevation',
    )

    _output_var_names = (
        'flow__depth',
        'flow__horizontal_velocity',
        'flow__vertical_velocity',
        'flow__sediment_concentration',
        'topographic__elevation',
        'bed__thickness',
        'flow__surface_elevation',
    )

    _var_units = {
        'flow__depth': 'm',
        'flow__horizontal_velocity': 'm/s',
        'flow__vertical_velocity': 'm/s',
        'flow__sediment_concentration': '1',
        'topographic__elevation': 'm',
        'bed__thickness': 'm',
        'flow__surface_elevation': 'm',
    }

    _var_mapping = {
        'flow__depth': 'node',
        'flow__horizontal_velocity': 'link',
        'flow__vertical_velocity': 'link',
        'flow__sediment_concentration': 'node',
        'topographic__elevation': 'node',
        'bed__thickness': 'node',
        'flow__surface_elevation': 'node',
    }

    _var_doc = {
        'flow__depth': 'The depth of flow at each node.',
        'flow__horizontal_velocity':
        'Horizontal component of flow velocity at each link',
        'flow__vertical_velocity':
        'Vertical component of flow velocity at each link',
        'flow__sediment_concentration': 'Sediment concentration in flow',
        'topographic__elevation': 'The land surface elevation.',
        'bed__thickness': 'The bed thickness',
        'flow__surface_elevation': 'Elevation of flow surface',
    }

    @use_file_name_or_kwds
    def __init__(self,
                 grid,
                 default_fixed_links=False,
                 h_init=0.0001,
                 h_w=0.01,
                 alpha=0.1,
                 Cf=0.004,
                 g=9.81,
                 R=1.65,
                 Ds=100 * 10**-6,
                 lambda_p=0.4,
                 r0=1.5,
                 nu=1.010 * 10**-6,
                 kappa=0.25,
                 flow_type='3eq',
                 implicit_num=50,
                 C_init=0.00001,
                 **kwds):
        """Create a debris-flow component.

        Parameters
        ----------
        grid : RasterModelGrid
            A landlab grid.
        h_init : float, optional
            Thickness of initial thin layer of flow to prevent divide by zero
            errors (m).
        h_w : float, optional
            Thickness of flow to judge "wet" nodes and links (m).
        alpha : float, optional
            Time step coefficient
        Cf : float, optional
            Dimensionless Chezy friction coefficient.
        g : float, optional
            Acceleration due to gravity (m/s^2)
        R : float, optional
            Submerged specific density (rho_s/rho_f - 1) (1)
        Ds : float, optional
            Sediment diameter (m)
        lambda_p : float, optional
            Bed sediment porosity (1)
        nu : float, optional
            Kinematic viscosity of water (at 293K)
        nu_t: float, optional
            Eddy viscosity for horizontal velocity components
        flow_type : str, optional
            '3eq' for the three equation model
            '4eq' for the four equation model (not implemented)
        """
        super(TurbidityCurrent2D, self).__init__(grid, **kwds)

        # First we copy our grid
        self._grid = grid

        # Copy model parameters
        self.h_init = h_init
        self.alpha = alpha
        if type(Cf) is str:
            self.Cf = self.grid.at_link[Cf]
        else:
            self.Cf = Cf
        self.g = g
        self.R = R
        self.Ds = Ds
        self.flow_type = flow_type
        self.h_w = h_w
        self.nu = nu
        self.kappa = kappa
        self.r0 = r0
        self.lambda_p = lambda_p
        self.implicit_num = implicit_num
        self.C_init = C_init

        # Now setting up fields at nodes and links
        try:
            self.eta = grid.add_zeros(
                'topographic__elevation',
                at='node',
                units=self._var_units['topographic__elevation'])

        except FieldError:
            # Field was already set
            self.eta = self._grid.at_node['topographic__elevation']

        try:
            self.bed_thick = grid.add_zeros(
                'bed__thickness',
                at='node',
                units=self._var_units['bed__thickness'])
        except FieldError:
            # Field was already set
            self.bed_thick = self._grid.at_node['bed__thickness']

        try:
            self.h = grid.add_zeros('flow__depth',
                                    at='node',
                                    units=self._var_units['flow__depth'])
            self.C = grid.add_zeros(
                'flow__sediment_concentration',
                at='node',
                units=self._var_units['flow__sediment_concentration'])

        except FieldError:
            # Field was already set
            self.h = grid.at_node['flow__depth']
            self.C = grid.at_node['flow__sediment_concentration']

        try:
            grid.add_zeros('flow__surface_elevation',
                           at='node',
                           units=self._var_units['flow__surface_elevation'])
            grid.at_node['flow__surface_elevation'] = self.h + self.eta

        except FieldError:
            grid.at_node['flow__surface_elevation'] = self.h + self.eta

        try:
            self.u = grid.add_zeros(
                'flow__horizontal_velocity',
                at='link',
                units=self._var_units['flow__horizontal_velocity'])
            self.v = grid.add_zeros(
                'flow__vertical_velocity',
                at='link',
                units=self._var_units['flow__vertical_velocity'])

        except FieldError:
            # Field was already set
            self.u = grid.at_link['flow__horizontal_velocity']
            self.v = grid.at_link['flow__vertical_velocity']

        self.h += self.h_init
        self.C += self.C_init

        # For gradient of parameters at nodes and links
        try:
            self.dudx = grid.add_zeros(
                'flow_horizontal_velocity__horizontal_gradient', at='link')
            self.dudy = grid.add_zeros(
                'flow_horizontal_velocity__vertical_gradient', at='link')
            self.dvdx = grid.add_zeros(
                'flow_vertical_velocity__horizontal_gradient', at='link')
            self.dvdy = grid.add_zeros(
                'flow_vertical_velocity__vertical_gradient', at='link')

        except FieldError:
            self.dudx = grid.at_link[
                'flow_horizontal_velocity__horizontal_gradient']
            self.dudy = grid.at_link[
                'flow_horizontal_velocity__vertical_gradient']
            self.dvdx = grid.at_link[
                'flow_vertical_velocity__horizontal_gradient']
            self.dvdy = grid.at_link[
                'flow_vertical_velocity__vertical_gradient']

        # record active links
        self.active_link_ids = links.active_link_ids(self.grid.shape,
                                                     self.grid.status_at_node)
        self.horizontal_active_links = links.horizontal_active_link_ids(
            self.grid.shape, self.active_link_ids)
        self.horizontal_active_links = np.delete(
            self.horizontal_active_links,
            np.where(self.horizontal_active_links < 0))
        self.vertical_active_links = links.vertical_active_link_ids(
            self.grid.shape, self.active_link_ids)
        self.vertical_active_links = np.delete(
            self.vertical_active_links,
            np.where(self.vertical_active_links < 0))

        # Set up temporal parameters
        self.Ch = self.C * self.h
        self.u_node = np.zeros(grid.number_of_nodes)
        self.v_node = np.zeros(grid.number_of_nodes)
        self.h_link = np.zeros(grid.number_of_links)
        self.Ch_link = np.zeros(grid.number_of_links)

        self.U = np.zeros(grid.number_of_links)  # composite velocity
        # composite velocity at node
        self.U_node = np.zeros(grid.number_of_nodes)

        self.ew_node = np.zeros(grid.number_of_nodes)
        self.ew_link = np.zeros(grid.number_of_links)
        self.es = np.zeros(grid.number_of_nodes)
        self.nu_t = np.zeros(grid.number_of_links)

        self.G_h = np.zeros(grid.number_of_nodes)
        self.G_u = np.zeros(grid.number_of_links)
        self.G_v = np.zeros(grid.number_of_links)
        self.G_Ch = np.zeros(grid.number_of_nodes)
        self.G_eta = np.zeros(grid.number_of_nodes)
        self.G_eta_p = np.zeros(grid.number_of_nodes)
        self.G_eta_c = np.zeros(grid.number_of_nodes)

        self.h_temp = np.zeros(grid.number_of_nodes)
        self.h_prev = np.zeros(grid.number_of_nodes)
        self.h_link_temp = np.zeros(grid.number_of_links)
        self.u_temp = np.zeros(grid.number_of_links)
        self.u_node_temp = np.zeros(grid.number_of_nodes)
        self.dudx_temp = np.zeros(grid.number_of_links)
        self.dudy_temp = np.zeros(grid.number_of_links)
        self.v_temp = np.zeros(grid.number_of_links)
        self.v_node_temp = np.zeros(grid.number_of_nodes)
        self.dvdx_temp = np.zeros(grid.number_of_links)
        self.dvdy_temp = np.zeros(grid.number_of_links)
        self.Ch_temp = np.zeros(grid.number_of_nodes)
        self.Ch_prev = np.zeros(grid.number_of_nodes)
        self.Ch_p = np.zeros(grid.number_of_nodes)
        self.Ch_link_temp = np.zeros(grid.number_of_links)
        self.eta_temp = self.eta.copy()
        self.eta_p = np.zeros(grid.number_of_nodes)
        self.eta_init = self.eta.copy()
        self.U_temp = self.U.copy()
        self.U_node_temp = self.U_node.copy()

        self.horizontal_up_nodes = np.zeros(grid.number_of_nodes,
                                            dtype=np.int64)
        self.vertical_up_nodes = np.zeros(grid.number_of_nodes, dtype=np.int64)
        self.horizontal_down_nodes = np.zeros(grid.number_of_nodes,
                                              dtype=np.int64)
        self.vertical_down_nodes = np.zeros(grid.number_of_nodes,
                                            dtype=np.int64)

        self.horizontal_up_links = np.zeros(grid.number_of_links,
                                            dtype=np.int64)
        self.vertical_up_links = np.zeros(grid.number_of_links, dtype=np.int64)
        self.horizontal_down_links = np.zeros(grid.number_of_links,
                                              dtype=np.int64)
        self.vertical_down_links = np.zeros(grid.number_of_links,
                                            dtype=np.int64)

        self.east_link_at_node = np.zeros(grid.number_of_nodes, dtype=np.int64)
        self.north_link_at_node = np.zeros(grid.number_of_nodes,
                                           dtype=np.int64)
        self.west_link_at_node = np.zeros(grid.number_of_nodes, dtype=np.int64)
        self.south_link_at_node = np.zeros(grid.number_of_nodes,
                                           dtype=np.int64)
        self.west_node_at_horizontal_link = np.zeros(grid.number_of_links,
                                                     dtype=np.int64)
        self.east_node_at_horizontal_link = np.zeros(grid.number_of_links,
                                                     dtype=np.int64)
        self.south_node_at_vertical_link = np.zeros(grid.number_of_links,
                                                    dtype=np.int64)
        self.north_node_at_vertical_link = np.zeros(grid.number_of_links,
                                                    dtype=np.int64)

        # Calculate subordinate parameters
        self.ws = get_ws(self.R, self.g, self.Ds, self.nu)

        # Record initial topography
        self.eta_init[:] = self.eta[:]

        # Start time of simulation is at 0 s
        grid.at_grid['time__elapsed'] = 0.0
        self.elapsed_time = grid.at_grid['time__elapsed']

        self.dt = None
        self.dt_local = None
        self.first_step = True
        # self.first_stage_count = 0

        self.neighbor_flag = False
        self.default_fixed_links = default_fixed_links

    def calc_time_step(self):
        """Calculate time step
        """

        sqrt_RCgh = np.sqrt(self.R * self.C * self.g * self.h)

        dt_local = self.alpha * self._grid.dx \
            / np.amax(np.array([np.amax(np.abs(self.u_node) + sqrt_RCgh),
                                np.amax(np.abs(self.v_node) + sqrt_RCgh), 1.0]))

        if self.first_step is True:
            dt_local *= 0.1

        return dt_local

    def set_up_neighbor_arrays(self):
        """Create and initialize link neighbor arrays.

        Set up arrays of neighboring horizontal and vertical nodes that are
        needed for CIP solution.
        """

        # Find the neighbor nodes
        self.core_nodes = self.grid.core_nodes
        neighbor_nodes = self.grid.adjacent_nodes_at_node.copy()
        self.node_east = neighbor_nodes[:, 0]
        self.node_north = neighbor_nodes[:, 1]
        self.node_west = neighbor_nodes[:, 2]
        self.node_south = neighbor_nodes[:, 3]

        # Find the neighbor links
        self.active_links = self.grid.active_links
        neighbor_links = links.neighbors_at_link(
            self.grid.shape, np.arange(self.grid.number_of_links)).copy()
        self.link_east = neighbor_links[:, 0]
        self.link_north = neighbor_links[:, 1]
        # This is to fix a bug in links.neighbors_at_link
        self.link_north[self.link_north == self.grid.number_of_links] = -1
        self.link_west = neighbor_links[:, 2]
        self.link_south = neighbor_links[:, 3]

        # Find links connected to nodes, and nodes connected to links
        self.east_link_at_node = self.grid.links_at_node[:, 0].copy()
        self.north_link_at_node = self.grid.links_at_node[:, 1].copy()
        self.west_link_at_node = self.grid.links_at_node[:, 2].copy()
        self.south_link_at_node = self.grid.links_at_node[:, 3].copy()
        self.west_node_at_horizontal_link = self.grid.nodes_at_link[:, 0].copy(
        )
        self.east_node_at_horizontal_link = self.grid.nodes_at_link[:, 1].copy(
        )
        self.south_node_at_vertical_link = self.grid.nodes_at_link[:, 0].copy()
        self.north_node_at_vertical_link = self.grid.nodes_at_link[:, 1].copy()

        # Process boundary nodes and links
        # Neumann boundary condition (gradient = 0) is assumed
        bound_node_north = np.where(self.grid.node_is_boundary(
            self.node_north))
        bound_node_south = np.where(self.grid.node_is_boundary(
            self.node_south))
        bound_node_east = np.where(self.grid.node_is_boundary(self.node_east))
        bound_node_west = np.where(self.grid.node_is_boundary(self.node_west))
        self.node_north[bound_node_north] = bound_node_north
        self.node_south[bound_node_south] = bound_node_south
        self.node_east[bound_node_east] = bound_node_east
        self.node_west[bound_node_west] = bound_node_west

        bound_link_north = np.where(self.link_north == -1)
        bound_link_south = np.where(self.link_south == -1)
        bound_link_east = np.where(self.link_east == -1)
        bound_link_west = np.where(self.link_west == -1)
        self.link_north[bound_link_north] = bound_link_north
        self.link_south[bound_link_south] = bound_link_south
        self.link_east[bound_link_east] = bound_link_east
        self.link_west[bound_link_west] = bound_link_west

        bound_node_north_at_link = np.where(
            self.grid.node_is_boundary(self.north_node_at_vertical_link))
        bound_node_south_at_link = np.where(
            self.grid.node_is_boundary(self.south_node_at_vertical_link))
        bound_node_east_at_link = np.where(
            self.grid.node_is_boundary(self.east_node_at_horizontal_link))
        bound_node_west_at_link = np.where(
            self.grid.node_is_boundary(self.west_node_at_horizontal_link))
        self.north_node_at_vertical_link[bound_node_north_at_link] = \
            self.south_node_at_vertical_link[bound_node_north_at_link]
        self.south_node_at_vertical_link[bound_node_south_at_link] = \
            self.north_node_at_vertical_link[bound_node_south_at_link]
        self.east_node_at_horizontal_link[bound_node_east_at_link] = \
            self.west_node_at_horizontal_link[bound_node_east_at_link]
        self.west_node_at_horizontal_link[bound_node_west_at_link] = \
            self.east_node_at_horizontal_link[bound_node_west_at_link]

        # Find obliquely neighbor vertical links from horizontal links
        # and obliquely neighbor horizontal links from vertical links
        self.vertical_link_NE = self.north_link_at_node[
            self.east_node_at_horizontal_link[self.horizontal_active_links]]
        self.vertical_link_SE = self.south_link_at_node[
            self.east_node_at_horizontal_link[self.horizontal_active_links]]
        self.vertical_link_NW = self.north_link_at_node[
            self.west_node_at_horizontal_link[self.horizontal_active_links]]
        self.vertical_link_SW = self.south_link_at_node[
            self.west_node_at_horizontal_link[self.horizontal_active_links]]
        self.horizontal_link_NE = self.east_link_at_node[
            self.north_node_at_vertical_link[self.vertical_active_links]]
        self.horizontal_link_SE = self.east_link_at_node[
            self.south_node_at_vertical_link[self.vertical_active_links]]
        self.horizontal_link_NW = self.west_link_at_node[
            self.north_node_at_vertical_link[self.vertical_active_links]]
        self.horizontal_link_SW = self.west_link_at_node[
            self.south_node_at_vertical_link[self.vertical_active_links]]

        # Once the neighbor arrays are set up, we change the flag to True!
        self.neighbor_flag = True

    def run_one_step(self, dt=None):
        """Generate debris flow across a grid.

        For one time step, this generates 'turbidity current' across
        a given grid by calculating flow height and concentration at each node
        and velocities at each link.

        Outputs flow depth, concentration, horizontal and vertical
        velocity values through time at every point in the input grid.

        Parameters
        ----------------
        dt : float, optional
            time to finish calculation of this step. Inside the model,
            local value of dt is used for stability of calculation.
            If dt=None, dt is set to be equal to local dt.
        """

        # DH adds a loop to enable an imposed tstep while maintaining stability
        local_elapsed_time = 0.
        if dt is None:
            dt = np.inf  # to allow the loop to begin
        self.dt = dt

        # First, we check and see if the neighbor arrays have been
        # initialized
        if self.neighbor_flag is False:
            self.set_up_neighbor_arrays()

        # copy class attributes to local variables
        dx = self.grid.dx

        # In case another component has added data to the fields, we just
        # reset our water depths, topographic elevations and water
        # velocity variables to the fields.
        self.h = self.grid['node']['flow__depth']
        self.eta = self.grid['node']['topographic__elevation']
        self.u = self.grid['link']['flow__horizontal_velocity']
        self.v = self.grid['link']['flow__vertical_velocity']
        self.C = self.grid['node']['flow__sediment_concentration']
        self.Ch = self.C * self.h

        # map node values to links, and link values to nodes.
        self.find_wet_grids(self.h)
        self.map_values(self.h, self.u, self.v, self.Ch, self.eta, self.h_link,
                        self.u_node, self.v_node, self.Ch_link, self.U,
                        self.U_node)
        self.copy_values_to_temp()
        self.update_up_down_links_and_nodes()

        # continue calculation until the prescribed time elapsed
        while local_elapsed_time < dt:
oooooooooooooooos
















































































































































































































            # set local time step
            dt_local = self.calc_time_step()
            # Can really get into trouble if nothing happens but we still run:
            if not dt_local < np.inf:
                break
            if local_elapsed_time + dt_local > dt:
                dt_local = dt - local_elapsed_time
            self.dt_local = dt_local



<<<<<<< HEAD
            # find wet nodes and links
            wet_nodes, partial_wet_nodes = self.find_wet_grids(
                self.h, self.core_nodes, self.node_north, self.node_south,
                self.node_east, self.node_west)
            wet_horizontal_links, partial_wet_horizontal_links \
                = self.find_wet_grids(
                    self.h_link, self.horizontal_active_links,
                    self.link_north,
                    self.link_south,
                    self.link_east,
                    self.link_west)
            wet_vertical_links, partial_wet_vertical_links \
                = self.find_wet_grids(
                    self.h_link, self.vertical_active_links,
                    self.link_north,
                    self.link_south,
                    self.link_east,
                    self.link_west)

            # calculation of advecton terms in continuum (h) and
            # momentum (u and v) equations by CIP method
            self.cip_2d_M_advection(self.h,
                                    self.dhdx,
                                    self.dhdy,
                                    self.u_node,
                                    self.v_node,
                                    wet_nodes,
                                    self.horizontal_up_nodes[wet_nodes],
                                    self.horizontal_down_nodes[wet_nodes],
                                    self.vertical_up_nodes[wet_nodes],
                                    self.vertical_down_nodes[wet_nodes],
                                    dx,
                                    self.dt_local,
                                    out_f=self.h_temp,
                                    out_dfdx=self.dhdx_temp,
                                    out_dfdy=self.dhdy_temp)


























=======
            # Find wet and partial wet grids
            self.find_wet_grids(self.h)

            # calculation of advecton terms of momentum (u and v) equations
            # by CIP method
>>>>>>> unstable

            rcip_2d_M_advection(
                self.u,
                self.dudx,
                self.dudy,
                self.u,
                self.v,
                self.wet_horizontal_links,
                self.horizontal_up_links[self.wet_horizontal_links],
                self.horizontal_down_links[self.wet_horizontal_links],
                self.vertical_up_links[self.wet_horizontal_links],
                self.vertical_down_links[self.wet_horizontal_links],
                dx,
                self.dt_local,
                out_f=self.u_temp,
                out_dfdx=self.dudx_temp,
                out_dfdy=self.dudy_temp)

            rcip_2d_M_advection(
                self.v,
                self.dvdx,
                self.dvdy,
                self.u,
                self.v,
                self.wet_vertical_links,
                self.horizontal_up_links[self.wet_vertical_links],
                self.horizontal_down_links[self.wet_vertical_links],
                self.vertical_up_links[self.wet_vertical_links],
                self.vertical_down_links[self.wet_vertical_links],
                dx,
                self.dt_local,
                out_f=self.v_temp,
                out_dfdx=self.dvdx_temp,
                out_dfdy=self.dvdy_temp)

            # process values at partial wet grids
            rcip_2d_M_advection(
                self.u,
                self.dudx,
                self.dudy,
                self.u,
                self.v,
                self.partial_wet_horizontal_links,
                self.horizontal_up_links[self.partial_wet_horizontal_links],
                self.horizontal_down_links[self.partial_wet_horizontal_links],
                self.vertical_up_links[self.partial_wet_horizontal_links],
                self.vertical_down_links[self.partial_wet_horizontal_links],
                dx,
                self.dt_local,
                out_f=self.u_temp,
                out_dfdx=self.dudx_temp,
                out_dfdy=self.dudy_temp)
>>>>>>> unstable

            rcip_2d_M_advection(
                self.v,
                self.dvdx,
                self.dvdy,
                self.u,
                self.v,
                self.partial_wet_vertical_links,
                self.horizontal_up_links[self.partial_wet_vertical_links],
                self.horizontal_down_links[self.partial_wet_vertical_links],
                self.vertical_up_links[self.partial_wet_vertical_links],
                self.vertical_down_links[self.partial_wet_vertical_links],
                dx,
                self.dt_local,
                out_f=self.v_temp,
                out_dfdx=self.dvdx_temp,
                out_dfdy=self.dvdy_temp)

            # update values after calculating advection terms
            # map node values to links, and link values to nodes.
            # self.h_temp[:] = self.h[:]
            # self.Ch_temp[:] = self.Ch[:]
            self.map_links_to_nodes(self.u_temp, self.v_temp, self.u_node_temp,
                                    self.v_node_temp, self.U_temp,
                                    self.U_node_temp)
            self.update_values()
            self.update_up_down_links_and_nodes()

            # calculate non-advection terms using implicit method
            self.h_prev[:] = self.h_temp[:]
            self.Ch_prev[:] = self.Ch_temp[:]
            converge = 10.0
            count = 0
            while ((converge > 1.0 * 10**-12) and (count < self.implicit_num)):

                self.calc_G_u(self.h_temp, self.h_link_temp, self.u_temp,
                              self.v_temp, self.Ch_temp, self.Ch_link_temp,
                              self.eta_temp, self.U_temp,
                              self.wet_horizontal_links)
                self.calc_G_v(self.h_temp, self.h_link_temp, self.u_temp,
                              self.v_temp, self.Ch_temp, self.Ch_link_temp,
                              self.eta_temp, self.U_temp,
                              self.wet_vertical_links)

                cip_2d_nonadvection(
                    self.u,
                    self.dudx,
                    self.dudy,
                    self.G_u,
                    self.u_temp,
                    self.v_temp,
                    self.wet_horizontal_links,
                    self.horizontal_up_links[self.wet_horizontal_links],
                    self.horizontal_down_links[self.wet_horizontal_links],
                    self.vertical_up_links[self.wet_horizontal_links],
                    self.vertical_down_links[self.wet_horizontal_links],
                    dx,
                    self.dt_local,
                    out_f=self.u_temp,
                    out_dfdx=self.dudx_temp,
                    out_dfdy=self.dudy_temp)

                cip_2d_nonadvection(
                    self.v,
                    self.dvdx,
                    self.dvdy,
                    self.G_v,
                    self.u_temp,
                    self.v_temp,
                    self.wet_vertical_links,
                    self.horizontal_up_links[self.wet_vertical_links],
                    self.horizontal_down_links[self.wet_vertical_links],
                    self.vertical_up_links[self.wet_vertical_links],
                    self.vertical_down_links[self.wet_vertical_links],
                    dx,
                    self.dt_local,
                    out_f=self.v_temp,
                    out_dfdx=self.dvdx_temp,
                    out_dfdy=self.dvdy_temp)

                self.map_links_to_nodes(
                    self.u_temp,
                    self.v_temp,
                    self.u_node_temp,
                    self.v_node_temp,
                    self.U_temp,
                    self.U_node_temp,
                )

                # Calculate non-advection terms of h and Ch
                self.calc_G_h(self.h_temp, self.h_link_temp, self.u_temp,
                              self.u_node_temp, self.v_temp, self.v_node_temp,
                              self.Ch_temp, self.U_node_temp, self.wet_nodes)
                self.calc_G_Ch(self.Ch_temp, self.Ch_link_temp, self.u_temp,
                               self.v_temp, self.wet_nodes)
                self.h_temp[self.wet_nodes] = self.h[self.wet_nodes] \
                    + self.dt_local * self.G_h[self.wet_nodes]
                self.Ch_temp[self.wet_nodes] = (
                    self.Ch[self.wet_nodes] +
                    self.dt_local * self.G_Ch[self.wet_nodes])

                # update values
                self.map_nodes_to_links(self.h_temp, self.Ch_temp,
                                        self.eta_temp, self.h_link_temp,
                                        self.Ch_link_temp)

                # check convergence of calculation
                converge = np.sum(
                    ((self.h_temp[self.wet_nodes]
                      - self.h_prev[self.wet_nodes])
                     / self.h_temp[self.wet_nodes])**2
                ) / self.wet_nodes.shape[0] \
                    + np.sum(
                    ((self.Ch_temp[self.wet_nodes]
                      - self.Ch_prev[self.wet_nodes])
                     / self.Ch_temp[self.wet_nodes])**2
                ) / self.wet_nodes[0]

                self.h_prev[:] = self.h_temp[:]
                self.Ch_prev[:] = self.Ch_temp[:]

                count += 1

            if count == self.implicit_num:
                print('Implicit calculation did not converge')

            # Find wet and partial wet grids
            self.find_wet_grids(self.h_temp)

            # update values
            self.update_values()
            self.map_values(self.h, self.u, self.v, self.Ch, self.eta,
                            self.h_link, self.u_node, self.v_node,
                            self.Ch_link, self.U, self.U_node)

            # calculate deposition/erosion
            self.calc_deposition(self.h,
                                 self.Ch,
                                 self.u_node,
                                 self.v_node,
                                 self.eta,
                                 self.U_node,
                                 out_Ch=self.Ch_temp,
                                 out_eta=self.eta_temp)

            # Calculate diffusion term of momentum
            self.calc_nu_t(self.u, self.v, self.h_link, out=self.nu_t)
            cip_2d_diffusion(self.u,
                             self.v,
                             self.nu_t,
                             self.horizontal_active_links,
                             self.vertical_active_links,
                             self.link_north,
                             self.link_south,
                             self.link_east,
                             self.link_west,
                             dx,
                             self.dt_local,
                             out_u=self.u_temp,
                             out_v=self.v_temp)

            # update values
            self.update_values()
            self.map_values(self.h, self.u, self.v, self.Ch, self.eta,
                            self.h_link, self.u_node, self.v_node,
                            self.Ch_link, self.U, self.U_node)

            # Process partial wet grids
            self.process_partial_wet_grids(self.h_temp, self.u_temp,
                                           self.v_temp, self.Ch_temp)

            # update values
            self.update_values()
            self.map_values(self.h, self.u, self.v, self.Ch, self.eta,
                            self.h_link, self.u_node, self.v_node,
                            self.Ch_link, self.U, self.U_node)
            self.update_up_down_links_and_nodes()

            # apply the shock dissipation scheme
            shock_dissipation(self.Ch,
                              self.h,
                              self.wet_nodes,
                              self.node_north,
                              self.node_south,
                              self.node_east,
                              self.node_west,
                              self.dt_local,
                              self.kappa,
                              out=self.Ch_temp)

            shock_dissipation(self.u,
                              self.h_link,
                              self.wet_horizontal_links,
                              self.link_north,
                              self.link_south,
                              self.link_east,
                              self.link_west,
                              self.dt_local,
                              self.kappa,
                              out=self.u_temp)

            shock_dissipation(self.v,
                              self.h_link,
                              self.wet_vertical_links,
                              self.link_north,
                              self.link_south,
                              self.link_east,
                              self.link_west,
                              self.dt_local,
                              self.kappa,
                              out=self.v_temp)

            shock_dissipation(self.h,
                              self.h,
                              self.wet_nodes,
                              self.node_north,
                              self.node_south,
                              self.node_east,
                              self.node_west,
                              self.dt_local,
                              self.kappa,
                              out=self.h_temp)

            # Reset our field values with the newest flow depth and
            # discharge.
            self.update_values()
            self.map_values(self.h, self.u, self.v, self.Ch, self.eta,
                            self.h_link, self.u_node, self.v_node,
                            self.Ch_link, self.U, self.U_node)
            self.update_up_down_links_and_nodes()
            self.first_step = False
            # if self.first_stage_count > 10:
            #     self.first_step = False
            # else:
            #     self.first_stage_count += 1

            # Calculation is terminated if global dt is not specified.
            if dt is np.inf:
                break
            local_elapsed_time += self.dt_local

        # Update bed thickness and record results in the grid
        self.elapsed_time += local_elapsed_time
        self.bed_thick = self.eta - self.eta_init
        self.copy_values_to_grid()

    def calc_deposition(self,
                        h,
                        Ch,
                        u_node,
                        v_node,
                        eta,
                        U_node,
                        out_Ch=None,
                        out_eta=None):
        """Calculate deposition/erosion processes

           Parameters
           ----------------
           h
           Ch
           u_node
           v_node
           eta
           out_Ch
           out_eta
        """
        if out_Ch is None:
            out_Ch = np.zeros(Ch.shape)
        if out_eta is None:
            out_eta = np.zeros(eta.shape)
        nodes = self.wet_nodes

        # Predictor
        self.calc_G_eta(h,
                        u_node,
                        v_node,
                        Ch,
                        U_node,
                        nodes,
                        out_geta=self.G_eta_p)
        self.Ch_p[nodes] = Ch[nodes] + self.dt_local * (-self.G_eta[nodes])

        # Corrector
        self.calc_G_eta(h,
                        u_node,
                        v_node,
                        self.Ch_p,
                        U_node,
                        nodes,
                        out_geta=self.G_eta_c)
        self.G_eta = 0.5 * (self.G_eta_c + self.G_eta_p)

        out_Ch[nodes] = Ch[nodes] \
            + self.dt_local * (
            - self.G_eta[nodes])
        out_eta[nodes] = eta[nodes] \
            + self.dt_local * (
            self.G_eta[nodes] / (
                1 - self.lambda_p))

    def copy_values_to_temp(self):
        self.h_temp[:] = self.h[:]
        self.h_link_temp[:] = self.h_link[:]
        self.u_temp[:] = self.u[:]
        self.u_node_temp[:] = self.u_node[:]
        self.v_temp[:] = self.v[:]
        self.v_node_temp[:] = self.v_node[:]
        self.Ch_temp[:] = self.Ch[:]
        self.Ch_link_temp[:] = self.Ch_link[:]
        self.eta_temp[:] = self.eta[:]
        self.U_temp[:] = self.U[:]
        self.U_node_temp[:] = self.U_node[:]

    def process_partial_wet_grids(
            self,
            h_temp,
            u_temp,
            v_temp,
            Ch_temp,
    ):
        """Process parameters of partial wet nodes and links

           Parameters
           ----------------------------
           h_temp,
           u_temp,
           v_temp,
           Ch_temp,
        """
        #######################################
        # get variables at the current moment #
        #######################################
        h = self.h
        Ch = self.Ch
        u = self.u
        v = self.v
        g = self.g
        R = self.R
        Cf = self.Cf
        dt = self.dt_local
        dx = self.grid.dx
        horizontally_partial_wet_nodes = self.horizontally_partial_wet_nodes
        vertically_partial_wet_nodes = self.vertically_partial_wet_nodes
        horizontally_wettest_nodes = self.horizontally_wettest_nodes
        vertically_wettest_nodes = self.vertically_wettest_nodes
        partial_wet_horizontal_links = self.partial_wet_horizontal_links
        partial_wet_vertical_links = self.partial_wet_vertical_links
        horizontal_direction_wettest = self.horizontal_direction_wettest
        vertical_direction_wettest = self.vertical_direction_wettest

        # empirical coefficient (Homma)
        gamma = 0.35

        ######################################################
        # horizontal and vertical flow discharge between wet #
        # and partial wet nodes                              #
        ######################################################
        overspill_velocity_x = gamma * np.sqrt(
            2.0 * R * g * Ch[horizontally_wettest_nodes]) / dx * dt
        overspill_velocity_y = gamma * np.sqrt(
            2.0 * R * g * Ch[vertically_wettest_nodes]) / dx * dt

        M_horiz = h[horizontally_wettest_nodes] * overspill_velocity_x
        M_vert = h[vertically_wettest_nodes] * overspill_velocity_y
        CM_horiz = Ch[horizontally_wettest_nodes] * overspill_velocity_x
        CM_vert = Ch[vertically_wettest_nodes] * overspill_velocity_y

        ################################################################
        # Calculate time development of variables at partial wet nodes #
        ################################################################

        # overspilling horizontally
        half_dry = h_temp[horizontally_wettest_nodes] < 8.0 * M_horiz
        M_horiz[half_dry] = h_temp[horizontally_wettest_nodes][half_dry] / 8.0
        h_temp[horizontally_partial_wet_nodes] += M_horiz
        h_temp[horizontally_wettest_nodes] -= M_horiz
        c_half_dry = Ch_temp[horizontally_wettest_nodes] < 8.0 * CM_horiz
        CM_horiz[
            c_half_dry] = Ch_temp[horizontally_wettest_nodes][c_half_dry] / 8.0
        Ch_temp[horizontally_partial_wet_nodes] += CM_horiz
        Ch_temp[horizontally_wettest_nodes] -= CM_horiz

        # overspilling vertically
        half_dry = h_temp[vertically_wettest_nodes] < 8.0 * M_vert
        M_vert[half_dry] = h_temp[vertically_wettest_nodes][half_dry] / 8.0
        h_temp[vertically_partial_wet_nodes] += M_vert
        h_temp[vertically_wettest_nodes] -= M_vert
        c_half_dry = Ch_temp[vertically_wettest_nodes] < 8.0 * CM_vert
        CM_vert[
            c_half_dry] = Ch_temp[vertically_wettest_nodes][c_half_dry] / 8.0
        Ch_temp[vertically_partial_wet_nodes] += CM_vert
        Ch_temp[vertically_wettest_nodes] -= CM_vert

        ################################################################
        # Calculate time development of variables at partial wet links #
        ################################################################
        CfuU = Cf * u[partial_wet_horizontal_links] \
            * np.sqrt(u[partial_wet_horizontal_links]**2
                      + v[partial_wet_horizontal_links]**2)
        CfvU = Cf * v[partial_wet_vertical_links] \
            * np.sqrt(u[partial_wet_vertical_links]**2
                      + v[partial_wet_vertical_links]**2)
        hdw = horizontal_direction_wettest
        vdw = vertical_direction_wettest
        u_temp[partial_wet_horizontal_links] = u[
            partial_wet_horizontal_links] \
            + hdw * gamma * np.sqrt(2.0 * R * g
                                    * Ch[horizontally_wettest_nodes]) - CfuU
        v_temp[partial_wet_vertical_links] = v[
            partial_wet_vertical_links] \
            + vdw * gamma * np.sqrt(2.0 * R * g
                                    * Ch[vertically_wettest_nodes]) - CfvU

    def copy_values_to_grid(self):
        """Copy flow parameters to grid
        """
        self.grid.at_node['flow__depth'] = self.h
        self.grid.at_link['flow__horizontal_velocity'] = self.u
        self.grid.at_link['flow__vertical_velocity'] = self.v
        self.C[:] = self.C_init
        self.C[
            self.wet_nodes] = self.Ch[self.wet_nodes] / self.h[self.wet_nodes]
        self.grid.at_node['flow__sediment_concentration'] = self.C
        self.grid.at_node['topographic__elevation'] = self.eta
        self.grid.at_node['bed__thickness'] = self.bed_thick
        self.grid.at_node['flow__surface_elevation'] = self.eta + self.h
        self.grid.at_link[
            'flow_horizontal_velocity__horizontal_gradient'] = self.dudx
        self.grid.at_link[
            'flow_horizontal_velocity__vertical_gradient'] = self.dudy
        self.grid.at_link[
            'flow_vertical_velocity__horizontal_gradient'] = self.dvdx
        self.grid.at_link[
            'flow_vertical_velocity__vertical_gradient'] = self.dvdy

    def update_values(self):
        """Update variables from temporally variables and
           apply boundary conditions
        """

        # adjust illeagal values
        self.h_temp[np.where(self.h_temp < self.h_init)] = self.h_init
        self.Ch_temp[np.where(self.Ch_temp <= 0)] = self.C_init * self.h_init

        # copy values from temp to grid values
        self.h[:] = self.h_temp[:]
        self.u[:] = self.u_temp[:]
        self.dudx[:] = self.dudx_temp[:]
        self.dudy[:] = self.dudy_temp[:]
        self.v[:] = self.v_temp[:]
        self.dvdx[:] = self.dvdx_temp[:]
        self.dvdy[:] = self.dvdy_temp[:]
        self.Ch[:] = self.Ch_temp[:]
        self.eta[:] = self.eta_temp[:]
        self.U[:] = self.U_temp[:]
        self.U_node[:] = self.U_node_temp[:]

    def update_up_down_links_and_nodes(self):
        """update location of upcurrent and downcurrent
           nodes and links
        """
        self.find_horizontal_up_down_nodes(self.u_node,
                                           out_up=self.horizontal_up_nodes,
                                           out_down=self.horizontal_down_nodes)

        self.find_vertical_up_down_nodes(self.v_node,
                                         out_up=self.vertical_up_nodes,
                                         out_down=self.vertical_down_nodes)

        self.find_horizontal_up_down_links(self.u,
                                           out_up=self.horizontal_up_links,
                                           out_down=self.horizontal_down_links)

        self.find_vertical_up_down_links(self.v,
                                         out_up=self.vertical_up_links,
                                         out_down=self.vertical_down_links)

    def find_wet_grids(self, h):
        """Find wet and partial wet nodes and links
           In this model, "dry" nodes are not subject to calculate.
           Only "wet nodes" are considered in the model
           calculation. "wet" is judged by the flow depth (> h_w).
           The "partial wet node" is a dry node but the upcurrent
           node is wet. Flow depth and velocity at partial wet
           nodes are calculated by the YANG's model (YANG et al.,2016)


           Parameters
           --------------------------
           h : ndarray
               flow height values for detecting wet and dry grids

           Returns
           -------------------------
           wet_nodes : ndarray, int
               ndarray indicating wet nodes. Nodes showing flow height h
               value larger than the threshold(h_w) value are judged as
               wet grid

           wet_horizontal_links : ndarray, int
               ndarray indicating wet horizontal links. Links connected
               with two wet nodes are judged as wet links.

           wet_vertical_links : ndarray, int
               ndarray indicating wet horizontal links. Links connected
               with two wet nodes are judged as wet links.

           dry_nodes : ndarray, int
               ndarray indicating dry nodes. Nodes showing flow height
               below the threshold (h_w) value

           dry_links : ndarray, int
               ndarray indicating links that are not wet or partial wet
               condition.

           horizontally_partial_wet_nodes : ndarray, int
               ndarray indicating horizontally partial wet nodes. Nodes
               showing h value lower than the threshold(h_w) value but an
               horizontally upcurrent node is wet

           vertically_partial_wet_nodes : ndarray, int
               ndarray indicating horizontally partial wet nodes. Nodes
               showing h value lower than the threshold(h_w) value but an
               horizontally upcurrent node is wet

           horizontally_wettest_nodes : ndarray, int
               ndarray indicating wet nodes horizontally adjacent to
               partially wet nodes.

           vertically_wettest_nodes : ndarray, int
               ndarray indicating wet nodes vertically adjacent to
               partially wet nodes.

           partial_wet_horizontal_links : ndarray, int
               ndarray indicating partially wet horizontal links

           partial_wet_vertical_links : ndarray, int
               ndarray indicating partially wet horizontal links

           horizontal_direction_wettest : ndarray, float
               ndarray indicating direction of gradient (1.0 or -1.0) at
               partial wet horizontal links

           vertical_direction_wettest : ndarray, float
               ndarray indicating direction of gradient (1.0 or -1.0) at
               partial wet vertical links


        """
        #############################
        # Copy parameters from self #
        #############################

        core = self.core_nodes
        horiz_links = self.horizontal_active_links
        vert_links = self.vertical_active_links

        east_nodes_at_node = self.node_east[core]
        west_nodes_at_node = self.node_west[core]
        north_nodes_at_node = self.node_north[core]
        south_nodes_at_node = self.node_south[core]

        east_link_at_node = self.east_link_at_node[core]
        west_link_at_node = self.west_link_at_node[core]
        north_link_at_node = self.north_link_at_node[core]
        south_link_at_node = self.south_link_at_node[core]

        east_nodes_at_link = self.east_node_at_horizontal_link[horiz_links]
        west_nodes_at_link = self.west_node_at_horizontal_link[horiz_links]
        north_nodes_at_link = self.north_node_at_vertical_link[vert_links]
        south_nodes_at_link = self.south_node_at_vertical_link[vert_links]

        h_w = self.h_w

        ############################
        # find wet nodes and links #
        ############################
        self.wet_nodes = core[np.where(h[core] > h_w)]
        self.wet_horizontal_links = horiz_links[np.where(
            (h[west_nodes_at_link] > h_w) & (h[east_nodes_at_link] > h_w))]
        self.wet_vertical_links = vert_links[np.where(
            (h[north_nodes_at_link] > h_w) & (h[south_nodes_at_link] > h_w))]

        ######################################################
        #find partial wet nodes and links in horizontal axis #
        ######################################################
        wet_at_east = np.where((h[core] < h_w) & (h[east_nodes_at_node] > h_w))
        horizontally_partial_wet_nodes_E = core[wet_at_east]
        horizontally_wettest_nodes_E = east_nodes_at_node[wet_at_east]
        partial_wet_horizontal_links_E = east_link_at_node[wet_at_east]
        horizontal_direction_wettest_E = -1.0 * np.ones(wet_at_east[0].shape)
        wet_at_west = np.where((h[core] < h_w) & (h[west_nodes_at_node] > h_w))
        horizontally_partial_wet_nodes_W = core[wet_at_west]
        horizontally_wettest_nodes_W = west_nodes_at_node[wet_at_west]
        partial_wet_horizontal_links_W = west_link_at_node[wet_at_west]
        horizontal_direction_wettest_W = 1.0 * np.ones(wet_at_west[0].shape)

        self.horizontally_partial_wet_nodes = np.concatenate([
            horizontally_partial_wet_nodes_E, horizontally_partial_wet_nodes_W
        ])
        self.horizontally_wettest_nodes = np.concatenate(
            [horizontally_wettest_nodes_E, horizontally_wettest_nodes_W])
        self.partial_wet_horizontal_links = np.concatenate(
            [partial_wet_horizontal_links_E, partial_wet_horizontal_links_W])
        self.horizontal_direction_wettest = np.concatenate(
            [horizontal_direction_wettest_E, horizontal_direction_wettest_W])

        ######################################################
        #find partial wet nodes and links in vertical axis #
        ######################################################
        # vertical partial wet check
        wet_at_north = np.where((h[core] < h_w)
                                & (h[north_nodes_at_node] > h_w))
        vertically_partial_wet_nodes_N = core[wet_at_north]
        vertically_wettest_nodes_N = north_nodes_at_node[wet_at_north]
        partial_wet_vertical_links_N = north_link_at_node[wet_at_north]
        vertical_direction_wettest_N = -1.0 * np.ones(wet_at_north[0].shape)

        wet_at_south = np.where((h[core] < h_w)
                                & (h[south_nodes_at_node] > h_w))
        vertically_partial_wet_nodes_S = core[wet_at_south]
        vertically_wettest_nodes_S = south_nodes_at_node[wet_at_south]
        partial_wet_vertical_links_S = south_link_at_node[wet_at_south]
        vertical_direction_wettest_S = 1.0 * np.ones(wet_at_south[0].shape)

        self.vertically_partial_wet_nodes = np.concatenate(
            [vertically_partial_wet_nodes_N, vertically_partial_wet_nodes_S])
        self.vertically_wettest_nodes = np.concatenate(
            [vertically_wettest_nodes_N, vertically_wettest_nodes_S])
        self.partial_wet_vertical_links = np.concatenate(
            [partial_wet_vertical_links_N, partial_wet_vertical_links_S])
        self.vertical_direction_wettest = np.concatenate(
            [vertical_direction_wettest_N, vertical_direction_wettest_S])

        #######################################
        # wet and partial wet nodes and links #
        #######################################
        self.wet_pwet_nodes = np.unique(
            np.concatenate([
                self.wet_nodes, self.horizontally_partial_wet_nodes,
                self.vertically_partial_wet_nodes
            ]))
        self.wet_pwet_links = np.unique(
            np.concatenate([
                self.wet_horizontal_links, self.wet_horizontal_links,
                self.wet_vertical_links, self.partial_wet_horizontal_links,
                self.partial_wet_vertical_links
            ]))

        ############################
        # find dry nodes and links #
        ############################
        self.dry_nodes = np.setdiff1d(core, self.wet_pwet_nodes)
        self.dry_links = np.setdiff1d(self.active_links, self.wet_pwet_links)

        return (self.wet_nodes, self.wet_horizontal_links,
                self.wet_vertical_links, self.horizontally_partial_wet_nodes,
                self.vertically_partial_wet_nodes,
                self.horizontally_wettest_nodes, self.vertically_wettest_nodes,
                self.partial_wet_horizontal_links,
                self.partial_wet_vertical_links,
                self.horizontal_direction_wettest,
                self.vertical_direction_wettest, self.wet_pwet_nodes,
                self.wet_pwet_links, self.dry_nodes, self.dry_links)

    def calc_nu_t(self, u, v, h_link, out=None):
        """Calculate eddy viscosity for horizontal diffusion of momentum

           Parameters
           -----------------------

           u : ndarray, float
               horizontal velocity
           v : ndarray, float
               vertical velocity

           Return
           -----------------------
           out : ndarray, float
               eddy viscosity for horizontal diffusion of momentum

        """
        if out is None:
            out = np.zeros(u.shape)

        karman = 0.4

        out = 1 / 6. * karman * np.sqrt(u**2 + v**2) * h_link

    def calc_G_h(self, h, h_link, u, u_node, v, v_node, Ch, U_node,
                 core_nodes):
        """Calculate non-advection term for h
        """

        # core_nodes = self.core_nodes
        link_north = self.north_link_at_node[core_nodes]
        link_south = self.south_link_at_node[core_nodes]
        link_east = self.east_link_at_node[core_nodes]
        link_west = self.west_link_at_node[core_nodes]
        dx = self.grid.dx

        # ew_node = get_ew(U_node, Ch, self.R, self.g, 0.1)
        ew_node = self.ew_node

        self.G_h[core_nodes] = ew_node[core_nodes] * U_node[core_nodes] \
            - (v[link_north] * h_link[link_north]
               - v[link_south] * h_link[link_south]) \
            / dx \
            - (u[link_east] * h_link[link_east]
               - u[link_west] * h_link[link_west]) \
            / dx

    def calc_G_Ch(self, Ch, Ch_link, u, v, core_nodes):
        """Calculate non-advection term for Ch
        """

        # core_nodes = self.core_nodes
        link_north = self.north_link_at_node[core_nodes]
        link_south = self.south_link_at_node[core_nodes]
        link_east = self.east_link_at_node[core_nodes]
        link_west = self.west_link_at_node[core_nodes]
        dx = self.grid.dx

        phi_x = u * Ch_link
        phi_y = v * Ch_link

        self.G_Ch[core_nodes] = - (phi_x[link_east] - phi_x[link_west]) \
            / dx \
                                - (phi_y[link_north] - phi_y[link_south]) \
            / dx

    def calc_G_u(self, h, h_link, u, v, Ch, Ch_link, eta, U, link_horiz):
        """Calculate non-advection term for u
        """

        # link_horiz = self.horizontal_active_links
        node_east = self.east_node_at_horizontal_link[link_horiz]
        node_west = self.west_node_at_horizontal_link[link_horiz]

        dx = self.grid.dx

        Rg = self.R * self.g
        eta_grad_at_link = self.grid.calc_grad_at_link(eta)
        eta_grad_x = eta_grad_at_link[link_horiz]
        U_horiz_link = U[link_horiz]
        # ew_link = get_ew(U_horiz_link, Ch_link[link_horiz], self.R, self.g,
        # 0.1)
        ew_link = self.ew_link[link_horiz]
        u_star_2 = self.Cf * u[link_horiz] * U_horiz_link

        self.G_u[link_horiz] = -Rg * Ch_link[link_horiz] * eta_grad_x \
            / h_link[link_horiz] \
            - 0.5 * Rg * (
            (Ch[node_east] * h[node_east]
             - Ch[node_west] * h[node_west])
            / dx) / h_link[link_horiz] \
            - u_star_2 \
            - ew_link * U_horiz_link * u[link_horiz] / h_link[link_horiz]

    def calc_G_v(self, h, h_link, u, v, Ch, Ch_link, eta, U, link_vert):
        """Calculate non-advection term for v
        """

        # link_vert = self.vertical_active_links
        node_north = self.north_node_at_vertical_link[link_vert]
        node_south = self.south_node_at_vertical_link[link_vert]

        dx = self.grid.dx

        Rg = self.R * self.g
        eta_grad_at_link = self.grid.calc_grad_at_link(eta)
        eta_grad_y = eta_grad_at_link[link_vert]
        U_vert_link = U[link_vert]
        # ew_link = get_ew(U_vert_link, Ch_link[link_vert], self.R, self.g, 0.1)
        ew_link = self.ew_link[link_vert]
        v_star_2 = self.Cf * v[link_vert] * U_vert_link

        self.G_v[link_vert] = -Rg * Ch_link[link_vert] * eta_grad_y \
            / h_link[link_vert] \
            - 0.5 * Rg * (
            Ch[node_north] * h[node_north]
            - Ch[node_south] * h[node_south]) \
            / dx / h_link[link_vert]\
            - v_star_2 \
            - ew_link * U_vert_link * v[link_vert] / h_link[link_vert]

    def calc_G_eta(
            self,
            h,
            u_node,
            v_node,
            Ch,
            U_node,
            core,
            out_geta=None,
    ):
        """Calculate non-advection term for eta
        """
        if out_geta is None:
            out_geta = np.zeros(h.shape)

        ws = self.ws
        r0 = self.r0
        u_star_at_node = np.sqrt(self.Cf * U_node[core]**2)
        self.es = get_es(self.R, self.g, self.Ds, self.nu, u_star_at_node)

        out_geta[core] = ws * (r0 * Ch[core] / h[core] - self.es)

        # remove too large gradients
        maxC = 0.05
        illeagal_val = np.where(out_geta[core] * self.dt_local > Ch[core])
        out_geta[core][illeagal_val] = Ch[core[illeagal_val]] / self.dt_local
        illeagal_val2 = np.where(
            Ch[core] - out_geta[core] * self.dt_local > maxC * h[core])
        out_geta[core][illeagal_val2] = (
            maxC * h[core][illeagal_val2] -
            Ch[core][illeagal_val2]) / self.dt_local

    def map_values(self, h, u, v, Ch, eta, h_link, u_node, v_node, Ch_link, U,
                   U_node):
        """map parameters at nodes to links, and those at links to nodes
        """
        self.map_links_to_nodes(u, v, u_node, v_node, U, U_node)
        self.map_nodes_to_links(h, Ch, eta, h_link, Ch_link)

    def map_links_to_nodes(self, u, v, u_node, v_node, U, U_node):
        """map parameters at links to nodes
        """

        # set velocity zero at dry links and nodes
        u[self.dry_links] = 0
        v[self.dry_links] = 0
        u_node[self.dry_nodes] = 0
        v_node[self.dry_nodes] = 0

        # Map values of horizontal links to vertical links, and
        # values of vertical links to horizontal links.
        # Horizontal velocity is only updated at horizontal links,
        # and vertical velocity is updated at vertical links during
        # CIP procedures. Therefore we need to map those values each other.
        u[self.vertical_active_links] = (
            u[self.horizontal_link_NE] + u[self.horizontal_link_NW] +
            u[self.horizontal_link_SE] + u[self.horizontal_link_SW]) / 4.0
        v[self.horizontal_active_links] = (
            v[self.vertical_link_NE] + v[self.vertical_link_NW] +
            v[self.vertical_link_SE] + v[self.vertical_link_SW]) / 4.0

        # Calculate composite velocity at links
        U[self.active_links] = np.sqrt(u[self.active_links]**2 +
                                       v[self.active_links]**2)

        # map link values (u, v) to nodes
        self.map_mean_of_links_to_node(u, self.core_nodes, out=u_node)
        self.map_mean_of_links_to_node(v, self.core_nodes, out=v_node)
        self.map_mean_of_links_to_node(U, self.core_nodes, out=U_node)
        # self.grid.map_mean_of_links_to_node(u, out=u_node)
        # self.grid.map_mean_of_links_to_node(v, out=v_node)
        # self.grid.map_mean_of_links_to_node(U, out=U_node)

    def map_mean_of_links_to_node(self, f, core, out=None):

        if out is None:
            out = np.zeros(self.number_of_nodes)
        out[core] = (f[self.north_link_at_node[core]] +
                     f[self.south_link_at_node[core]] +
                     f[self.east_link_at_node[core]] +
                     f[self.west_link_at_node[core]]) / 4.0

        return out

    def map_nodes_to_links(self, h, Ch, eta, h_link, Ch_link):
        """map parameters at nodes to links
        """
        grid = self.grid

        # remove illeagal values
        # h[h < self.h_init] = self.h_init
        # Ch[Ch < self.C_init * self.h_init] = self.C_init * self.h_init

        # map node values (h, C, eta) to links
        grid.map_mean_of_link_nodes_to_link(h, out=h_link)
        grid.map_mean_of_link_nodes_to_link(Ch, out=Ch_link)

    def find_horizontal_up_down_nodes(self, u, out_up=None, out_down=None):
        """Find indeces of nodes that locate
           at horizontally upcurrent and downcurrent directions
        """
        if out_up is None:
            out_up = np.empty(u.shape, dtype=np.int64)
        if out_down is None:
            out_down = np.empty(u.shape, dtype=np.int64)

        out_up[:] = self.node_west[:]
        out_down[:] = self.node_east[:]
        negative_u_index = np.where(u < 0)[0]
        out_up[negative_u_index] = self.node_east[negative_u_index]
        out_down[negative_u_index] = self.node_west[negative_u_index]

        return out_up, out_down

    def find_vertical_up_down_nodes(self, u, out_up=None, out_down=None):
        """Find indeces of nodes that locate
           at vertically upcurrent and downcurrent directions
        """

        if out_up is None:
            out_up = np.empty(u.shape, dtype=np.int64)
        if out_down is None:
            out_down = np.empty(u.shape, dtype=np.int64)

        out_up[:] = self.node_south[:]
        out_down[:] = self.node_north[:]
        negative_u_index = np.where(u < 0)[0]
        out_up[negative_u_index] = self.node_north[negative_u_index]
        out_down[negative_u_index] = self.node_south[negative_u_index]

        return out_up, out_down

    def find_horizontal_up_down_links(self, u, out_up=None, out_down=None):
        """Find indeces of nodes that locate
           at horizontally upcurrent and downcurrent directions
        """
        if out_up is None:
            out_up = np.zeros(u.shape, dtype=np.int64)
        if out_down is None:
            out_down = np.zeros(u.shape, dtype=np.int64)

        out_up[:] = self.link_west[:]
        out_down[:] = self.link_east[:]
        negative_u_index = np.where(u < 0)[0]
        out_up[negative_u_index] = self.link_east[negative_u_index]
        out_down[negative_u_index] = self.link_west[negative_u_index]
        return out_up, out_down

    def find_vertical_up_down_links(self, u, out_up=None, out_down=None):
        """Find indeces of nodes that locate
           at vertically upcurrent and downcurrent directions
        """

        if out_up is None:
            out_up = np.zeros(u.shape, dtype=np.int64)
        if out_down is None:
            out_down = np.zeros(u.shape, dtype=np.int64)

        out_up[:] = self.link_south[:]
        out_down[:] = self.link_north[:]
        negative_u_index = np.where(u < 0)[0]
        out_up[negative_u_index] = self.link_north[negative_u_index]
        out_down[negative_u_index] = self.link_south[negative_u_index]

        return out_up, out_down


def create_topography(
        length=8000,
        width=2000,
        spacing=20,
        slope_outside=0.1,
        slope_inside=0.05,
        slope_basin_break=2000,
        canyon_basin_break=2200,
        canyon_center=1000,
        canyon_half_width=100,
):
    # making grid
    # size of calculation domain is 4 x 8 km with dx = 20 m
    lgrids = int(length / spacing)
    wgrids = int(width / spacing)
    grid = RasterModelGrid((lgrids, wgrids), spacing=spacing)
    grid.add_zeros('flow__depth', at='node')
    grid.add_zeros('topographic__elevation', at='node')
    grid.add_zeros('flow__horizontal_velocity', at='link')
    grid.add_zeros('flow__vertical_velocity', at='link')
    grid.add_zeros('flow__sediment_concentration', at='node')
    grid.add_zeros('bed__thickness', at='node')

    # making topography
    # set the slope
    grid.at_node['topographic__elevation'] = (
        grid.node_y - slope_basin_break) * slope_outside

    # set canyon
    d0 = slope_inside * (canyon_basin_break - slope_basin_break)
    d = slope_inside * (grid.node_y - canyon_basin_break) - d0
    a = d0 / canyon_half_width**2
    canyon_elev = a * (grid.node_x - canyon_center)**2 + d
    inside = np.where(canyon_elev < grid.at_node['topographic__elevation'])
    grid.at_node['topographic__elevation'][inside] = canyon_elev[inside]

    # set basin
    basin_height = 0
    basin_region = grid.at_node['topographic__elevation'] < basin_height
    grid.at_node['topographic__elevation'][basin_region] = basin_height
    grid.set_closed_boundaries_at_grid_edges(False, False, False, False)

    return grid


def create_init_flow_region(
        grid,
        initial_flow_concentration=0.02,
        initial_flow_thickness=200,
        initial_region_radius=200,
        initial_region_center=[1000, 7000],
):
    # making initial flow region
    initial_flow_region = (
        (grid.node_x - initial_region_center[0])**2 +
        (grid.node_y - initial_region_center[1])**2) < initial_region_radius**2
    grid.at_node['flow__depth'][initial_flow_region] = initial_flow_thickness
    grid.at_node['flow__sediment_concentration'][
        initial_flow_region] = initial_flow_concentration


def create_topography_from_geotiff(geotiff_filename,
                                   xlim=None,
                                   ylim=None,
                                   spacing=500):

    # read a geotiff file into ndarray
    topo_file = gdal.Open(geotiff_filename, gdalconst.GA_ReadOnly)
    topo_data = topo_file.GetRasterBand(1).ReadAsArray()
    if (xlim is not None) and (ylim is not None):
        topo_data = topo_data[xlim[0]:xlim[1], ylim[0]:ylim[1]]

    grid = RasterModelGrid(topo_data.shape, spacing=spacing)
    grid.add_zeros('flow__depth', at='node')
    grid.add_zeros('topographic__elevation', at='node')
    grid.add_zeros('flow__horizontal_velocity', at='link')
    grid.add_zeros('flow__vertical_velocity', at='link')
    grid.add_zeros('flow__sediment_concentration', at='node')
    grid.add_zeros('bed__thickness', at='node')

    grid.at_node['topographic__elevation'][grid.nodes] = topo_data

    return grid


if __name__ == '__main__':
    # import ipdb
    # ipdb.set_trace()
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

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
        initial_flow_thickness=50,
        initial_region_radius=50,
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
                            Ds=100 * 10**-6,
                            h_init=0.00001,
                            h_w=0.01,
                            C_init=0.00001,
                            implicit_num=20,
                            r0=1.5)

    # start calculation
    t = time.time()
    save_grid(grid, 'tc{:04d}.grid'.format(0), clobber=True)
    Ch_init = np.sum(tc.Ch)
    last = 2

    for i in range(1, last + 1):
        tc.run_one_step(dt=1.0)
        save_grid(grid, 'tc{:04d}.grid'.format(i), clobber=True)
        print("", end="\r")
        print("{:.1f}% finished".format(i / last * 100), end='\r')
        if np.sum(tc.Ch) / Ch_init < 0.01:
            break

    print('elapsed time: {} sec.'.format(time.time() - t))
