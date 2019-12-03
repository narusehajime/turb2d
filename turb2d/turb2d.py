import numpy as np
from landlab import Component, FieldError, RasterModelGrid
from landlab.utils.decorators import use_file_name_or_kwds
from landlab.grid.structured_quad import links
from landlab.io.native_landlab import save_grid
from cip import rcip_2d_M_advection, cip_2d_nonadvection, cip_2d_diffusion
from sediment_func import get_es, get_ew, get_ws
import time
import ipdb
"""A component of landlab that simulates a turbidity current on 2D grids

This component simulates turbidity currents using the 2-D numerical model of
shallow-water equations over topography on the basis of 3 equation model of
Parker et al. (1986). This component is based on the landlab component
 overland_flow that was written by Jordan Adams.

.. codeauthor:: Hajime Naruse

Examples
---------
    # making grid
    # size of calculation domain is 4 x 8 km with dx = 20 m
    grid = RasterModelGrid((400, 100), spacing=10.0)
    grid.add_zeros('flow__depth', at='node')
    grid.add_zeros('topographic__elevation', at='node')
    grid.add_zeros('flow__horizontal_velocity', at='link')
    grid.add_zeros('flow__vertical_velocity', at='link')
    grid.add_zeros('flow__sediment_concentration', at='node')
    grid.add_zeros('bed__thickness', at='node')

    # making topography
    # set the slope
    slope = 0.1
    slope_basin_break = 1000
    grid.at_node['topographic__elevation'] = (
        grid.node_y - slope_basin_break) * slope

    # set canyon
    canyon_center = 500
    canyon_half_width = 400
    canyon_depth = 50
    canyon = ((grid.node_x >= canyon_center - canyon_half_width) &
              (grid.node_x <= canyon_center + canyon_half_width))
    grid.at_node['topographic__elevation'][canyon] -= canyon_depth - np.abs(
        (grid.node_x[canyon] -
         canyon_center)) * canyon_depth / canyon_half_width

    # set basin
    basin_region = grid.at_node['topographic__elevation'] < 0
    grid.at_node['topographic__elevation'][basin_region] = 0
    grid.set_closed_boundaries_at_grid_edges(False, False, False, False)

    # making initial flow region
    initial_flow_concentration = 0.02
    initial_flow_thickness = 100
    initial_region_radius = 100
    initial_region_center = [500, 3500]
    initial_flow_region = (
        (grid.node_x - initial_region_center[0])**2 +
        (grid.node_y - initial_region_center[1])**2) < initial_region_radius**2
    grid.at_node['flow__depth'][initial_flow_region] = initial_flow_thickness
    grid.at_node['flow__sediment_concentration'][
        initial_flow_region] = initial_flow_concentration

    # making turbidity current object
    tc = TurbidityCurrent2D(
        grid,
        Cf=0.004,
        alpha=0.2,
        kappa=0.001,
        Ds=80 * 10**-6,
    )

    # start calculation
    t = time.time()
    save_grid(grid, 'tc{:04d}.grid'.format(0), clobber=True)
    last = 100
    for i in range(1, last + 1):
        tc.run_one_step(dt=100.0)
        save_grid(grid, 'tc{:04d}.grid'.format(i), clobber=True)
        print("", end="\r")
        print("{:.1f}% finished".format(i / last * 100), end='\r')
    print('elapsed time: {} sec.'.format(time.time() - t))

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
    )

    _output_var_names = (
        'flow__depth',
        'flow__horizontal_velocity',
        'flow__vertical_velocity',
        'flow__sediment_concentration',
        'topographic__elevation',
        'bed__thickness',
    )

    _var_units = {
        'flow__depth': 'm',
        'flow__horizontal_velocity': 'm/s',
        'flow__vertical_velocity': 'm/s',
        'flow__sediment_concentration': '1',
        'topographic__elevation': 'm',
        'bed__thickness': 'm',
    }

    _var_mapping = {
        'flow__depth': 'node',
        'flow__horizontal_velocity': 'link',
        'flow__vertical_velocity': 'link',
        'flow__sediment_concentration': 'node',
        'topographic__elevation': 'node',
        'bed__thickness': 'node',
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
    }

    @use_file_name_or_kwds
    def __init__(self,
                 grid,
                 default_fixed_links=False,
                 h_init=0.0001,
                 h_w=0.001,
                 alpha=0.1,
                 Cf=0.004,
                 g=9.81,
                 R=1.65,
                 Ds=100 * 10**-6,
                 lambda_p=0.4,
                 r0 = 1.5,
                 nu=1.010 * 10**-6,
                 kappa=0.001,
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
            self.bed_thick = grid.add_zeros(
                'bed__thickness',
                at='node',
                units=self._var_units['bed__thickness'])
            self.h = grid.add_zeros('flow__depth',
                                    at='node',
                                    units=self._var_units['flow__depth'])
            self.u = grid.add_zeros(
                'flow__horizontal_velocity',
                at='link',
                units=self._var_units['flow__horizontal_velocity'])
            self.v = grid.add_zeros(
                'flow__vertical_velocity',
                at='link',
                units=self._var_units['flow__vertical_velocity'])
            self.C = grid.add_zeros(
                'flow__sediment_concentration',
                at='link',
                units=self._var_units['flow__sediment_concentration'])

        except FieldError:
            # Field was already set
            self.u = grid.at_link['flow__horizontal_velocity']
            self.v = grid.at_link['flow__vertical_velocity']
            self.h = grid.at_node['flow__depth']
            self.C = grid.at_node['flow__sediment_concentration']
            self.eta = self._grid.at_node['topographic__elevation']
            self.bed_thick = self._grid.at_node['bed__thickness']

        self.h += self.h_init
        self.C += self.C_init

        # For gradient of parameters at nodes and links
        try:
            self.dxidx = grid.add_zeros('flow_surface__horizontal_gradient',
                                        at='node')
            self.dxidy = grid.add_zeros('flow_surface__vertical_gradient',
                                        at='node')
            self.dhdx = grid.add_zeros('flow_depth__horizontal_gradient',
                                       at='node')
            self.dhdy = grid.add_zeros('flow_depth__vertical_gradient',
                                       at='node')
            self.dudx = grid.add_zeros(
                'flow_horizontal_velocity__horizontal_gradient', at='link')
            self.dudy = grid.add_zeros(
                'flow_horizontal_velocity__vertical_gradient', at='link')
            self.dvdx = grid.add_zeros(
                'flow_vertical_velocity__horizontal_gradient', at='link')
            self.dvdy = grid.add_zeros(
                'flow_vertical_velocity__vertical_gradient', at='link')
            self.dCdx = grid.add_zeros(
                'flow_sediment_concentration__horizontal_gradient', at='node')
            self.dCdy = grid.add_zeros(
                'flow_sediment_concentration__vertical_gradient', at='node')

            self.eta_grad = grid.add_zeros('topographic_elevation__gradient',
                                           at='link')

        except FieldError:
            self.dxidx = grid.at_node['flow_surface__horizontal_gradient']
            self.dxidy = grid.at_node['flow_surface__vertical_gradient']
            self.dhdx = grid.at_node['flow_depth__horizontal_gradient']
            self.dhdy = grid.at_node['flow_depth__vertical_gradient']
            self.dudx = grid.at_link[
                'flow_horizontal_velocity__horizontal_gradient']
            self.dudy = grid.at_link[
                'flow_horizontal_velocity__vertical_gradient']
            self.dvdx = grid.at_link[
                'flow_vertical_velocity__horizontal_gradient']
            self.dvdy = grid.at_link[
                'flow_vertical_velocity__vertical_gradient']
            self.dCdx = grid.at_node[
                'flow_sediment_concentration__horizontal_gradient']
            self.dCdy = grid.at_node[
                'flow_sediment_concentration__vertical_gradient']

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
        self.u_node = np.zeros(grid.number_of_nodes)
        self.v_node = np.zeros(grid.number_of_nodes)
        self.h_link = np.zeros(grid.number_of_links)
        self.C_link = np.zeros(grid.number_of_links)

        self.ew_node = np.zeros(grid.number_of_nodes)
        self.ew_link = np.zeros(grid.number_of_links)
        self.es = np.zeros(grid.number_of_nodes)
        self.nu_t = np.zeros(grid.number_of_links)

        self.G_h = np.zeros(grid.number_of_nodes)
        self.G_u = np.zeros(grid.number_of_links)
        self.G_v = np.zeros(grid.number_of_links)
        self.G_Ch = np.zeros(grid.number_of_nodes)
        self.G_eta = np.zeros(grid.number_of_nodes)

        self.h_temp = np.zeros(grid.number_of_nodes)
        self.h_link_temp = np.zeros(grid.number_of_links)
        self.dhdx_temp = np.zeros(grid.number_of_nodes)
        self.dhdy_temp = np.zeros(grid.number_of_nodes)
        self.u_temp = np.zeros(grid.number_of_links)
        self.u_node_temp = np.zeros(grid.number_of_nodes)
        self.dudx_temp = np.zeros(grid.number_of_links)
        self.dudy_temp = np.zeros(grid.number_of_links)
        self.v_temp = np.zeros(grid.number_of_links)
        self.v_node_temp = np.zeros(grid.number_of_nodes)
        self.dvdx_temp = np.zeros(grid.number_of_links)
        self.dvdy_temp = np.zeros(grid.number_of_links)
        self.C_temp = np.zeros(grid.number_of_nodes)
        self.C_link_temp = np.zeros(grid.number_of_links)
        self.dCdx_temp = np.zeros(grid.number_of_nodes)
        self.dCdy_temp = np.zeros(grid.number_of_nodes)
        self.eta_temp = self.eta.copy()
        self.eta_init = self.eta.copy()

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
      
        self.east_link_at_node = np.zeros(grid.number_of_nodes,
                                            dtype=np.int64)
        self.north_link_at_node = np.zeros(grid.number_of_nodes,
                                            dtype=np.int64)
        self.west_link_at_node = np.zeros(grid.number_of_nodes,
                                            dtype=np.int64)
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

        self.neighbor_flag = False
        self.default_fixed_links = default_fixed_links

    def calc_time_step(self):
        """Calculate time step
        """
        sqrt_RCgh = np.sqrt(self.R * self.C * self.g * self.h)
        # sqrt_gh = np.sqrt(self.g * self.h)

        dt_local = self.alpha * self._grid.dx \
            / np.amax(np.array([np.amax(np.abs(self.u_node) + sqrt_RCgh),
                                np.amax(np.abs(self.v_node) + sqrt_RCgh), 1.0]))

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
        self.west_node_at_horizontal_link = self.grid.nodes_at_link[:, 0].copy()
        self.east_node_at_horizontal_link = self.grid.nodes_at_link[:, 1].copy()
        self.south_node_at_vertical_link = self.grid.nodes_at_link[:, 0].copy()
        self.north_node_at_vertical_link = self.grid.nodes_at_link[:, 1].copy()

        # Process boundary nodes and links
        # Neumann boundary condition (gradient = 0) is assumed
        bound_node_north = np.where(self.grid.node_is_boundary(self.node_north))
        bound_node_south = np.where(self.grid.node_is_boundary(self.node_south))
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

        # map node values to links, and link values to nodes.
        self.map_values(self.h, self.u, self.v, self.C, self.eta, self.h_link,
                        self.u_node, self.v_node, self.C_link)
        self.update_up_down_links_and_nodes()

        # continue calculation until the prescribed time elapsed
        while local_elapsed_time < dt:
            # set local time step
            dt_local = self.calc_time_step()
            # Can really get into trouble if nothing happens but we still run:
            if not dt_local < np.inf:
                break
            if local_elapsed_time + dt_local > dt:
                dt_local = dt - local_elapsed_time
            self.dt_local = dt_local

            # Find wet and partial wet grids
            (wet_nodes,
                wet_horizontal_links,
                wet_vertical_links,
                horizontally_partial_wet_nodes,
                vertically_partial_wet_nodes,
                horizontally_wettest_nodes,
                vertically_wettest_nodes,
                partial_wet_horizontal_links,
                partial_wet_vertical_links,
                horizontal_direction_wettest,
                vertical_direction_wettest
            ) = self.find_wet_grids(self.h)

            # calculation of advecton terms of momentum (u and v) equations
            # by CIP method

            rcip_2d_M_advection(
                self.u,
                self.dudx,
                self.dudy,
                self.u,
                self.v,
                wet_horizontal_links,
                self.horizontal_up_links[wet_horizontal_links],
                self.horizontal_down_links[wet_horizontal_links],
                self.vertical_up_links[wet_horizontal_links],
                self.vertical_down_links[wet_horizontal_links],
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
                wet_vertical_links,
                self.horizontal_up_links[wet_vertical_links],
                self.horizontal_down_links[wet_vertical_links],
                self.vertical_up_links[wet_vertical_links],
                self.vertical_down_links[wet_vertical_links],
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
                partial_wet_horizontal_links,
                self.horizontal_up_links[partial_wet_horizontal_links],
                self.horizontal_down_links[partial_wet_horizontal_links],
                self.vertical_up_links[partial_wet_horizontal_links],
                self.vertical_down_links[partial_wet_horizontal_links],
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
                partial_wet_vertical_links,
                self.horizontal_up_links[partial_wet_vertical_links],
                self.horizontal_down_links[partial_wet_vertical_links],
                self.vertical_up_links[partial_wet_vertical_links],
                self.vertical_down_links[partial_wet_vertical_links],
                dx,
                self.dt_local,
                out_f=self.v_temp,
                out_dfdx=self.dvdx_temp,
                out_dfdy=self.dvdy_temp)


            # update values after calculating advection terms
            # map node values to links, and link values to nodes.
            self.h_temp[:] = self.h[:]
            self.C_temp[:] = self.C[:]
            self.update_values()
            self.map_values(self.h, self.u, self.v, self.C, self.eta,
                            self.h_link, self.u_node, self.v_node, self.C_link)
            self.update_up_down_links_and_nodes()

            # calculate non-advection terms using implicit method
            self.copy_values_to_temp()
            h_prev = self.h_temp.copy()
            C_prev = self.C_temp.copy()
            converge = 10.0
            count = 0
            while ((converge > 1.0 * 10**-15) and (count < self.implicit_num)):
                # for i in range(1):
                # calculate non-advection terms on wet grids

                self.calc_G_u(self.h_temp, self.h_link, self.u_temp,
                              self.v_temp, self.C_temp, self.C_link,
                              self.eta_temp,wet_horizontal_links)
                self.calc_G_v(self.h_temp, self.h_link, self.u_temp,
                              self.v_temp, self.C_temp, self.C_link_temp,
                              self.eta_temp, wet_vertical_links)

                cip_2d_nonadvection(
                    self.u,
                    self.dudx,
                    self.dudy,
                    self.G_u,
                    self.u,
                    self.v,
                    wet_horizontal_links,
                    self.horizontal_up_links[wet_horizontal_links],
                    self.horizontal_down_links[wet_horizontal_links],
                    self.vertical_up_links[wet_horizontal_links],
                    self.vertical_down_links[wet_horizontal_links],
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
                    self.u,
                    self.v,
                    wet_vertical_links,
                    self.horizontal_up_links[wet_vertical_links],
                    self.horizontal_down_links[wet_vertical_links],
                    self.vertical_up_links[wet_vertical_links],
                    self.vertical_down_links[wet_vertical_links],
                    dx,
                    self.dt_local,
                    out_f=self.v_temp,
                    out_dfdx=self.dvdx_temp,
                    out_dfdy=self.dvdy_temp)

                self.map_values(self.h_temp, self.u_temp, self.v_temp,
                                self.C_temp, self.eta_temp, self.h_link_temp,
                                self.u_node_temp, self.v_node_temp,
                                self.C_link_temp)

                # Calculate non-advection terms of h and Ch
                self.calc_G_h(self.h_temp, self.h_link_temp, self.u_temp,
                              self.u_node_temp, self.v_temp, self.v_node_temp,
                              self.C_temp, wet_nodes)
                self.calc_G_Ch(self.h_temp, self.h_link_temp, self.u_temp,
                              self.u_node_temp, self.v_temp, self.v_node_temp,
                              self.C_temp, self.C_link_temp, wet_nodes)
                self.calc_G_eta(self.h_temp, self.u_node_temp,
                                self.v_node_temp, self.C_temp, wet_nodes)

                # Update C, h and deposition/erosion
                self.h_temp[wet_nodes] = self.h[wet_nodes] \
                                         + self.dt_local * self.G_h[wet_nodes]
                self.C_temp[wet_nodes] = (self.C[wet_nodes] * self.h[
                    wet_nodes] + self.dt_local * self.G_Ch[wet_nodes]) \
                    / self.h_temp[wet_nodes]
                self.eta_temp[wet_nodes] = self.eta[wet_nodes] \
                                       + self.dt_local * self.G_eta[wet_nodes]

                # Process partial wet grids
                self.process_partial_wet_grids(
                                       self.h_temp,
                                       self.u_temp,
                                       self.v_temp,
                                       self.C_temp,
                                       horizontally_partial_wet_nodes,
                                       vertically_partial_wet_nodes,
                                       horizontally_wettest_nodes,
                                       vertically_wettest_nodes,
                                       partial_wet_horizontal_links,
                                       partial_wet_vertical_links,
                                       horizontal_direction_wettest,
                                       vertical_direction_wettest,
                )

                # judge convergence of implicit scheme
                # converge = np.sum(
                #     ((self.h_temp[wet_nodes]
                #      - h_prev[wet_nodes])
                #     / self.h_temp[wet_nodes])**2 \
                #     + ((self.C_temp[wet_nodes] * self.h_temp[wet_nodes]
                #         - C_prev[wet_nodes] * h_prev[wet_nodes]) \
                #        / (self.C_temp[wet_nodes] * self.h_temp[wet_nodes]))**2\
                #     )  / self.grid.number_of_core_nodes
                converge = np.sum(
                    ((self.h_temp[wet_nodes]
                     - h_prev[wet_nodes])
                    / self.h_temp[wet_nodes])**2 \
                    )  / self.grid.number_of_core_nodes

                h_prev[:] = self.h_temp[:]
                C_prev[:] = self.C_temp[:]

                self.map_values(self.h_temp, self.u_temp, self.v_temp,
                                self.C_temp, self.eta_temp, self.h_link_temp,
                                self.u_node_temp, self.v_node_temp,
                                self.C_link_temp)
                count += 1

            if count == self.implicit_num:
                print('Implicit calculation did not converge')

            # update values
            self.update_values()

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
            self.map_values(self.h, self.u, self.v, self.C, self.eta,
                            self.h_link, self.u_node, self.v_node, self.C_link)

            # apply the shock dissipation scheme
            self.shock_dissipation(self.C,
                                   self.h,
                                   wet_nodes,
                                   self.node_north,
                                   self.node_south,
                                   self.node_east,
                                   self.node_west,
                                   self.dt_local,
                                   out=self.C_temp)

            self.shock_dissipation(self.u,
                                   self.h_link,
                                   wet_horizontal_links,
                                   self.link_north,
                                   self.link_south,
                                   self.link_east,
                                   self.link_west,
                                   self.dt_local,
                                   out=self.u_temp)

            self.shock_dissipation(self.v,
                                   self.h_link,
                                   wet_vertical_links,
                                   self.link_north,
                                   self.link_south,
                                   self.link_east,
                                   self.link_west,
                                   self.dt_local,
                                   out=self.v_temp)

            self.shock_dissipation(self.h,
                                   self.h,
                                   wet_nodes,
                                   self.node_north,
                                   self.node_south,
                                   self.node_east,
                                   self.node_west,
                                   self.dt_local,
                                   out=self.h_temp)

            # Reset our field values with the newest flow depth and
            # discharge.
            self.update_values()
            self.map_values(self.h, self.u, self.v, self.C, self.eta,
                            self.h_link, self.u_node, self.v_node, self.C_link)
            self.update_up_down_links_and_nodes()

            # Calculation is terminated if global dt is not specified.
            if dt is np.inf:
                break
            local_elapsed_time += self.dt_local

        # Update bed thickness and record results in the grid
        self.elapsed_time += local_elapsed_time
        self.bed_thick = self.eta - self.eta_init
        self.copy_values_to_grid()

    def copy_values_to_temp(self):
        self.h_temp = np.copy(self.h)
        self.u_temp = np.copy(self.u)
        self.v_temp = np.copy(self.v)
        self.C_temp = np.copy(self.C)
        self.eta_temp = np.copy(self.eta)

    def process_partial_wet_grids(self,
                          h_temp,
                          u_temp,
                          v_temp,
                          C_temp,
                          horizontally_partial_wet_nodes,
                          vertically_partial_wet_nodes,
                          horizontally_wettest_nodes,
                          vertically_wettest_nodes,
                          partial_wet_horizontal_links,
                          partial_wet_vertical_links,
                          horizontal_direction_wettest,
                          vertical_direction_wettest,
    ):
        """Process parameters of partial wet nodes and links
           
           Parameters
           ----------------------------
           h_temp,
           u_temp,
           v_temp,
           C_temp,
           horizontally_partial_wet_nodes,
           vertically_partial_wet_nodes,
           horizontally_wettest_nodes,
           vertically_wettest_nodes,
           partial_wet_horizontal_links,
           partial_wet_vertical_links

        """
        #######################################
        # get variables at the current moment #
        #######################################
        h = self.h
        C = self.C
        u = self.u
        v = self.v
        g = self.g
        R = self.R
        Cf = self.Cf
        dt = self.dt_local

        # empirical coefficient (Homma) 
        gamma = 0.35

        ######################################################
        # horizontal and vertical flow discharge between wet #
        # and partial wet nodes                              #
        ######################################################
        M_horiz = gamma * h[horizontally_wettest_nodes] \
            * np.sqrt(2.0 * R * C[horizontally_wettest_nodes]
                      * g * h[horizontally_wettest_nodes])
        M_vert = gamma * h[vertically_wettest_nodes] \
            * np.sqrt(2.0 * R * C[vertically_wettest_nodes]
                      * g * h[vertically_wettest_nodes])

        ################################################################
        # Calculate time development of variables at partial wet nodes #
        ################################################################

        # overspilling horizontally
        overspill = np.zeros(h.shape) # to record variation of flow depth
        overspill[horizontally_partial_wet_nodes] = M_horiz * dt
        overspill[horizontally_wettest_nodes] = - M_horiz * dt
        C_temp[horizontally_partial_wet_nodes] = C[
            horizontally_wettest_nodes]

        # overspilling vertically
        overspill[vertically_partial_wet_nodes] = overspill[
            vertically_partial_wet_nodes] + M_vert * dt
        overspill[vertically_wettest_nodes] = overspill[
            vertically_wettest_nodes] - M_vert * dt
        C_temp[vertically_partial_wet_nodes] = C_temp[vertically_wettest_nodes]

        # sum of variation by overspilling
        partial_wet_nodes = np.unique(np.concatenate([
            horizontally_partial_wet_nodes, vertically_partial_wet_nodes]))
        wettest_nodes = np.unique(np.concatenate([
            horizontally_wettest_nodes, vertically_wettest_nodes]))
        pwet_wettest_nodes = np.concatenate([partial_wet_nodes, wettest_nodes])
        h_temp[pwet_wettest_nodes] = h[pwet_wettest_nodes] \
                                    + overspill[pwet_wettest_nodes]

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
            + hdw * gamma * np.sqrt(2.0 * R * C[horizontally_wettest_nodes]
                      * g * h[horizontally_wettest_nodes]) - CfuU
        v_temp[partial_wet_vertical_links] = v[
            partial_wet_vertical_links] \
            + vdw * gamma * np.sqrt(2.0 * R * C[vertically_wettest_nodes]
                      * g * h[vertically_wettest_nodes]) - CfvU

    def copy_values_to_grid(self):
        """Copy flow parameters to grid
        """
        self.grid.at_node['flow__depth'] = self.h
        self.grid.at_link['flow__horizontal_velocity'] = self.u
        self.grid.at_link['flow__vertical_velocity'] = self.v
        self.grid.at_node['flow__sediment_concentration'] = self.C
        self.grid.at_node['topographic__elevation'] = self.eta
        self.grid.at_node['bed__thickness'] = self.bed_thick

    def update_values(self):
        """Update variables from temporally variables and
           apply boundary conditions
        """

        self.h[:] = self.h_temp[:]
        self.dhdx[:] = self.dhdx_temp[:]
        self.dhdy[:] = self.dhdy_temp[:]
        self.u[:] = self.u_temp[:]
        self.dudx[:] = self.dudx_temp[:]
        self.dudy[:] = self.dudy_temp[:]
        self.v[:] = self.v_temp[:]
        self.dvdx[:] = self.dvdx_temp[:]
        self.dvdy[:] = self.dvdy_temp[:]
        self.C[:] = self.C_temp[:]
        self.dCdx[:] = self.dCdx_temp[:]
        self.dCdy[:] = self.dCdy_temp[:]
        self.eta[:] = self.eta_temp[:]

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

    def find_wet_grids(
            self,
            h
    ):
        """Find wet and partial wet nodes and links
           In this model, "dry" nodes are not subject to calculate.
           Only "wet nodes" are considered in the model 
           calculation. "wet" is judged by the flow depth (> h_w).
           The "partial wet node" is a dry node but the upcurrent
           node is wet. Flow depth and velocity at partial wet
           nodes are calculated by the YANG's model (YANG et al.,
           2016)

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
        wet_nodes = core[np.where(h[core] > h_w)]
        wet_horizontal_links = horiz_links[np.where(
            (h[west_nodes_at_link] > h_w) & (h[east_nodes_at_link] > h_w))]
        wet_vertical_links = vert_links[np.where(
            (h[north_nodes_at_link] > h_w) & (h[south_nodes_at_link] > h_w))]

        ######################################################
        #find partial wet nodes and links in horizontal axis #
        ######################################################
        wet_at_east = np.where((h[core] < h_w)
                           & (h[east_nodes_at_node] > h_w))
        horizontally_partial_wet_nodes_E = core[wet_at_east]
        horizontally_wettest_nodes_E = east_nodes_at_node[wet_at_east]
        partial_wet_horizontal_links_E = east_link_at_node[wet_at_east]
        horizontal_direction_wettest_E = - 1.0 * np.ones(wet_at_east[0].shape)
        wet_at_west = np.where((h[core] < h_w)
                           & (h[west_nodes_at_node] > h_w))
        horizontally_partial_wet_nodes_W = core[wet_at_west]
        horizontally_wettest_nodes_W = west_nodes_at_node[wet_at_west]
        partial_wet_horizontal_links_W = west_link_at_node[wet_at_west]
        horizontal_direction_wettest_W = 1.0 * np.ones(wet_at_west[0].shape)

        horizontally_partial_wet_nodes = np.concatenate(
            [horizontally_partial_wet_nodes_E,
             horizontally_partial_wet_nodes_W])
        horizontally_wettest_nodes = np.concatenate(
            [horizontally_wettest_nodes_E,
             horizontally_wettest_nodes_W])
        partial_wet_horizontal_links = np.concatenate(
            [partial_wet_horizontal_links_E,
             partial_wet_horizontal_links_W])
        horizontal_direction_wettest = np.concatenate(
            [horizontal_direction_wettest_E,
             horizontal_direction_wettest_W])
                
        ######################################################
        #find partial wet nodes and links in vertical axis #
        ######################################################
        # vertical partial wet check
        wet_at_north = np.where((h[core] < h_w)
                           & (h[north_nodes_at_node] > h_w))
        vertically_partial_wet_nodes_N = core[wet_at_north]
        vertically_wettest_nodes_N = north_nodes_at_node[wet_at_north]
        partial_wet_vertical_links_N = north_link_at_node[wet_at_north]
        vertical_direction_wettest_N = - 1.0 * np.ones(wet_at_north[0].shape)
        
        wet_at_south = np.where((h[core] < h_w)
                           & (h[south_nodes_at_node] > h_w))
        vertically_partial_wet_nodes_S = core[wet_at_south]
        vertically_wettest_nodes_S = south_nodes_at_node[wet_at_south]
        partial_wet_vertical_links_S = south_link_at_node[wet_at_south]
        vertical_direction_wettest_S = 1.0 * np.ones(wet_at_south[0].shape)

        vertically_partial_wet_nodes = np.concatenate(
            [vertically_partial_wet_nodes_N,
             vertically_partial_wet_nodes_S])
        vertically_wettest_nodes = np.concatenate(
            [vertically_wettest_nodes_N,
             vertically_wettest_nodes_S])
        partial_wet_vertical_links = np.concatenate(
            [partial_wet_vertical_links_N,
             partial_wet_vertical_links_S])
        vertical_direction_wettest = np.concatenate(
            [vertical_direction_wettest_N,
             vertical_direction_wettest_S])


        
        return (wet_nodes,
                wet_horizontal_links,
                wet_vertical_links,
                horizontally_partial_wet_nodes,
                vertically_partial_wet_nodes,
                horizontally_wettest_nodes,
                vertically_wettest_nodes,
                partial_wet_horizontal_links,
                partial_wet_vertical_links,
                horizontal_direction_wettest,
                vertical_direction_wettest
        )

    def shock_dissipation(
            self,
            f,
            h,
            core,
            north_id,
            south_id,
            east_id,
            west_id,
            dt,
            out=None,
    ):
        """ adding artificial viscosity for numerical stability

            Parameters            ------------------
            f : ndarray, float
                parameter for which the artificial viscosity is applied
            h : ndarray, float
                flow height
            core : ndarray, int
                indeces of core nodes or links
            north_id : ndarray, int
                indeces of nodes or links that locate north of core
            south_id : ndarray, int
                indeces of nodes or links that locate south of core
            east_id : ndarray, int
                indeces of nodes or links that locate east of core
            west_id : ndarray, int
                indeces of nodes or links that locate west of core
        """
        if out is None:
            out = np.zeros(f.shape)

        kappa = self.kappa  # artificial viscosity
        dx = self.grid.dx
        eps_i = np.zeros(h.shape)
        eps_i_half = np.zeros(h.shape)
        north = north_id[core]
        south = south_id[core]
        east = east_id[core]
        west = west_id[core]

        # First, artificial diffusion is applied to east-west direction
        eps_i[core] = kappa * dx * np.abs(h[east] - 2 * h[core] + h[west]) / \
            (h[east] + 2 * h[core] + h[west]) / dt
        eps_i_half[core] = np.max([eps_i[east], eps_i[core]], axis=0)
        out[core] = f[core] + eps_i_half[core] * (f[east] - f[core]) - \
            eps_i_half[west] * (f[core] - f[west])

        # Next, artificial diffusion is applied to north-south direction
        eps_i[core] = kappa * dx * np.abs(h[north] - 2 * h[core] +
                                          h[south]) / (h[north] + 2 * h[core] +
                                                       h[south]) / dt
        eps_i_half[core] = np.max([eps_i[north], eps_i[core]], axis=0)
        out[core] = out[core] + eps_i_half[core] * (out[north] - out[core]) \
            - eps_i_half[south] * (out[core] - out[south])

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
        
        kappa = 0.4
       
        out = 1 / 6. * kappa * np.sqrt(u**2 + v**2) * h_link
        

    def calc_G_h(self, h, h_link, u, u_node, v, v_node, C, core_nodes):
        """Calculate non-advection term for h
        """

        # core_nodes = self.core_nodes
        link_north = self.north_link_at_node[core_nodes]
        link_south = self.south_link_at_node[core_nodes]
        link_east = self.east_link_at_node[core_nodes]
        link_west = self.west_link_at_node[core_nodes]
        dx = self.grid.dx
        
        U_node = np.sqrt(u_node**2 + v_node**2)
        ew_node = get_ew(U_node, h, C, self.R,
                         self.g, self.h_w)
        # ew_node = np.zeros(h.shape)

        self.G_h[core_nodes] = ew_node[core_nodes] * U_node[core_nodes] \
                               - (v[link_north] * h_link[link_north]
                                  - v[link_south] * h_link[link_south]) \
                                  / dx \
                                - (u[link_east] * h_link[link_east]
                                   - u[link_west] * h_link[link_west]) \
                                   / dx

    def calc_G_Ch(self,
                  h,
                  h_link,
                  u,
                  u_node,
                  v,
                  v_node,
                  C,
                  C_link,
                  core_nodes):
        """Calculate non-advection term for Ch
        """

        # core_nodes = self.core_nodes
        link_north = self.north_link_at_node[core_nodes]
        link_south = self.south_link_at_node[core_nodes]
        link_east = self.east_link_at_node[core_nodes]
        link_west = self.west_link_at_node[core_nodes]
        dx = self.grid.dx
        
        ws = self.ws
        U_node = np.sqrt(u_node**2 + v_node**2)
        u_star_node = np.sqrt(self.Cf) * U_node
        es = get_es(self.R, self.g, self.Ds, self.nu, u_star_node)
        # es = np.zeros(h.shape)
        r0 = self.r0

        phi_x = u * C_link * h_link
        phi_y = v * C_link * h_link

        self.G_Ch[core_nodes] = ws * (es[core_nodes] - r0 * C[core_nodes]) \
                                - (phi_x[link_east] - phi_x[link_west]) \
                                / dx \
                                - (phi_y[link_north] - phi_y[link_south]) \
                                / dx

        # self.G_Ch[np.where((h < (self.h_init*2)) & (self.G_Ch > 0))] = 0


    def calc_G_u(self, h, h_link, u, v, C, C_link, eta, link_horiz):
        """Calculate non-advection term for u
        """

        # link_horiz = self.horizontal_active_links
        node_east = self.east_node_at_horizontal_link[link_horiz]
        node_west = self.west_node_at_horizontal_link[link_horiz]
        
        dx = self.grid.dx
        v_on_horiz = v[link_horiz]

        Rg = self.R * self.g
        eta_grad_at_link = self.grid.calc_grad_at_link(eta)
        eta_grad_x = eta_grad_at_link[link_horiz]
        U_horiz_link = np.sqrt(u[link_horiz]**2 + v_on_horiz**2)
        ew_link = get_ew(U_horiz_link, h_link[link_horiz], C_link[link_horiz],
                         self.R, self.g, self.h_w)
        # ew_link = np.zeros(ew_link.shape)
        u_star_2 = self.Cf * u[link_horiz] * U_horiz_link

        self.G_u[link_horiz] = -Rg * C_link[link_horiz] * eta_grad_x \
                   - 0.5 * Rg * (
                  (C[node_east] * h[node_east] - C[node_west] * h[node_west])\
                       / dx + 
                  C_link[link_horiz] * \
                  (h[node_east] - h[node_west]) / dx
                   ) \
                  - u_star_2 \
                  - ew_link * U_horiz_link * u[link_horiz] / h_link[link_horiz]

    def calc_G_v(self, h, h_link, u, v, C, C_link, eta, link_vert):
        """Calculate non-advection term for v
        """

        # link_vert = self.vertical_active_links
        node_north = self.north_node_at_vertical_link[link_vert]
        node_south = self.south_node_at_vertical_link[link_vert]
        
        dx = self.grid.dx
        u_on_vert = u[link_vert]

        Rg = self.R * self.g
        eta_grad_at_link = self.grid.calc_grad_at_link(eta)
        eta_grad_y = eta_grad_at_link[link_vert]
        U_vert_link = np.sqrt(v[link_vert]**2 + u_on_vert**2)
        ew_link = get_ew(U_vert_link, h_link[link_vert], C_link[link_vert],
                         self.R, self.g, self.h_w)
        # ew_link = np.zeros(ew_link.shape)
        v_star_2 = self.Cf * v[link_vert] * U_vert_link

        self.G_v[link_vert] = -Rg * C_link[link_vert] * eta_grad_y \
                   - 0.5 * Rg * (
                  (C[node_north] * h[node_north]
                   - C[node_south] * h[node_south])\
                       / dx + 
                  C_link[link_vert] * \
                  (h[node_north] - h[node_south]) / dx
                   ) \
                  - v_star_2 \
                  - ew_link * U_vert_link * v[link_vert] / h_link[link_vert]

    def calc_G_eta(self, h, u_node, v_node, C, core):
        """Calculate non-advection term for eta
        """

        ws = self.ws
        r0 = self.r0
        u_star_at_node = self.Cf * (u_node[core]**2 + v_node[core]**2)
        es = get_es(self.R, self.g, self.Ds, self.nu, u_star_at_node)
        # es = np.zeros(es.shape)

        self.G_eta[core] = ws * (r0 * C[core] -
                                       es) / (1 - self.lambda_p)

    def map_values(self, h, u, v, C, eta, h_link, u_node, v_node, C_link):
        """map parameters at nodes to links, and those at links to nodes
        """

        grid = self.grid

        # Map values of horizontal links to vertical links, and
        # values of vertical links to horizontal links.
        # Horizontal velocity is only updated at horizontal links,
        # and vertical velocity is updated at vertical links during
        # CIP procedures. Therefore we need to map those values each other.
        # self.v[self.
        #        horizontal_active_links] = grid.map_mean_of_link_nodes_to_link(
        #            self.v_node)[self.horizontal_active_links]
        # self.u[self.
        #        vertical_active_links] = grid.map_mean_of_link_nodes_to_link(
        #            self.u_node)[self.vertical_active_links]
        self.u[self.vertical_active_links] = (
            self.u[self.horizontal_link_NE] + self.u[self.horizontal_link_NW] +
            self.u[self.horizontal_link_SE] +
            self.u[self.horizontal_link_SW]) / 4.0
        self.v[self.horizontal_active_links] = (
            self.v[self.vertical_link_NE] + self.v[self.vertical_link_NW] +
            self.v[self.vertical_link_SE] +
            self.v[self.vertical_link_SW]) / 4.0

        # self.v[self.
        #        horizontal_active_links] = grid.map_mean_of_link_nodes_to_link(
        #            self.v_node)[self.horizontal_active_links]
        # self.u[
        #     self.vertical_active_links] = grid.map_mean_of_link_nodes_to_link(
        #         self.u_node)[self.vertical_active_links]

        # adjust illeagal values
        h[np.where(h < self.h_init)] = self.h_init
        C[np.where(C <= 0)] = self.C_init

        # map link values (u, v) to nodes
        grid.map_mean_of_links_to_node(u, out=u_node)
        grid.map_mean_of_links_to_node(v, out=v_node)

        # map node values (h, C, eta) to links
        grid.map_mean_of_link_nodes_to_link(h, out=h_link)
        grid.map_mean_of_link_nodes_to_link(C, out=C_link)

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


if __name__ == '__main__':
    # making grid
    # size of calculation domain is 4 x 8 km with dx = 20 m
    grid = RasterModelGrid((400, 100), spacing=10.0)
    grid.add_zeros('flow__depth', at='node')
    grid.add_zeros('topographic__elevation', at='node')
    grid.add_zeros('flow__horizontal_velocity', at='link')
    grid.add_zeros('flow__vertical_velocity', at='link')
    grid.add_zeros('flow__sediment_concentration', at='node')
    grid.add_zeros('bed__thickness', at='node')

    # making topography
    # set the slope
    slope = 0.1
    slope_basin_break = 1000
    grid.at_node['topographic__elevation'] = (grid.node_y -
                                              slope_basin_break) * slope

    # set canyon
    canyon_center = 500
    canyon_half_width = 400
    canyon_depth = 50
    a = canyon_depth / canyon_half_width ** 2
    w = canyon_half_width
    c = canyon_center
    canyon = ((grid.node_x >= canyon_center - canyon_half_width) &
              (grid.node_x <= canyon_center + canyon_half_width))
    # grid.at_node['topographic__elevation'][canyon] -= canyon_depth - np.abs(
    #     (grid.node_x[canyon] -
    #      canyon_center)) * canyon_depth / canyon_half_width
    grid.at_node['topographic__elevation'][canyon] += a * (
        grid.node_x[canyon] - canyon_center)**2 - canyon_depth

    # set basin
    basin_region = grid.at_node['topographic__elevation'] < 0
    grid.at_node['topographic__elevation'][basin_region] = 0
    grid.set_closed_boundaries_at_grid_edges(False, False, False, False)

    # making initial flow region
    initial_flow_concentration = 0.02
    initial_flow_thickness = 100
    initial_region_radius = 100
    initial_region_center = [500, 3500]
    initial_flow_region = (
        (grid.node_x - initial_region_center[0])**2 +
        (grid.node_y - initial_region_center[1])**2) < initial_region_radius**2
    grid.at_node['flow__depth'][initial_flow_region] = initial_flow_thickness
    grid.at_node['flow__sediment_concentration'][
        initial_flow_region] = initial_flow_concentration

    # making turbidity current object
    tc = TurbidityCurrent2D(
        grid,
        Cf=0.004,
        alpha=0.1,
        kappa=0.001,
        Ds=100 * 10**-6,
        h_init=0.0001,
        h_w = 0.001,
        C_init=0.00001,
        implicit_num=20,
        r0=1.5
    )

    # start calculation
    t = time.time()
    save_grid(grid, 'tc{:04d}.grid'.format(0), clobber=True)
    last = 100

    ipdb.set_trace()
    for i in range(1, last + 1):
        tc.run_one_step(dt=20.0)
        save_grid(grid, 'tc{:04d}.grid'.format(i), clobber=True)
        print("", end="\r")
        print("{:.1f}% finished".format(i / last * 100), end='\r')
    print('elapsed time: {} sec.'.format(time.time() - t))
