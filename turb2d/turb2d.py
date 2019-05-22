import numpy as np
from landlab import Component, FieldError, RasterModelGrid
from landlab.utils.decorators import use_file_name_or_kwds
from landlab.grid.structured_quad import links
from landlab.io.native_landlab import save_grid
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
        'flow__depth':
        'The depth of flow at each node.',
        'flow__horizontal_velocity':
        'Horizontal component of flow velocity at each link',
        'flow__vertical_velocity':
        'Vertical component of flow velocity at each link',
        'flow__sediment_concentration':
        'Sediment concentration in flow',
        'topographic__elevation':
        'The land surface elevation.',
        'bed__thickness':
        'The bed thickness',
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
                 nu=1.010 * 10**-6,
                 nu_t=0.01,
                 kappa=0.001,
                 flow_type='3eq',
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
        self.nu_t = nu_t
        self.R = R
        self.Ds = Ds
        self.flow_type = flow_type
        self.h_w = h_w
        self.nu = nu
        self.kappa = kappa
        self.lambda_p = lambda_p

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
            self.h = grid.add_zeros(
                'flow__depth', at='node', units=self._var_units['flow__depth'])
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
            # Field was already set; still, fill it with zeros
            self.u = grid.at_link['flow__horizontal_velocity']
            self.v = grid.at_link['flow__vertical_velocity']
            self.h = grid.at_node['flow__depth']
            self.C = grid.at_node['flow__sediment_concentration']
            self.eta = self._grid.at_node['topographic__elevation']
            self.bed_thick = self._grid.at_node['bed__thickness']

        self.h += self.h_init

        # For gradient of parameters at nodes and links
        try:
            self.dxidx = grid.add_zeros(
                'flow_surface__horizontal_gradient', at='node')
            self.dxidy = grid.add_zeros(
                'flow_surface__vertical_gradient', at='node')
            self.dhdx = grid.add_zeros(
                'flow_depth__horizontal_gradient', at='node')
            self.dhdy = grid.add_zeros(
                'flow_depth__vertical_gradient', at='node')
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

            self.eta_grad = grid.add_zeros(
                'topographic_elevation__gradient', at='link')

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
        self.r0 = np.zeros(grid.number_of_nodes)

        self.G_h = np.zeros(grid.number_of_nodes)
        self.G_u = np.zeros(grid.number_of_links)
        self.G_v = np.zeros(grid.number_of_links)
        self.G_C = np.zeros(grid.number_of_nodes)
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

        self.horizontal_up_nodes = np.zeros(
            grid.number_of_nodes, dtype=np.int64)
        self.vertical_up_nodes = np.zeros(grid.number_of_nodes, dtype=np.int64)
        self.horizontal_down_nodes = np.zeros(
            grid.number_of_nodes, dtype=np.int64)
        self.vertical_down_nodes = np.zeros(
            grid.number_of_nodes, dtype=np.int64)

        self.horizontal_up_links = np.zeros(
            grid.number_of_links, dtype=np.int64)
        self.vertical_up_links = np.zeros(grid.number_of_links, dtype=np.int64)
        self.horizontal_down_links = np.zeros(
            grid.number_of_links, dtype=np.int64)
        self.vertical_down_links = np.zeros(
            grid.number_of_links, dtype=np.int64)

        # Calculate subordinate parameters
        self.ws = self.get_ws()

        # Record initial topography
        self.eta_init = self.eta

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
        # sqrt_RCgh = np.sqrt(self.R * self.C * self.g * self.h)
        sqrt_gh = np.sqrt(self.g * self.h)

        dt_local = self.alpha * self._grid.dx \
            / np.amax(np.array([np.amax(self.u_node + sqrt_gh),
                                np.amax(self.v_node + sqrt_gh), 1.0]))

        return dt_local

    def set_up_neighbor_arrays(self):
        """Create and initialize link neighbor arrays.

        Set up arrays of neighboring horizontal and vertical nodes that are
        needed for CIP solution.
        """

        # Find the neighbor nodes
        self.core_nodes = self.grid.core_nodes
        neighbor_nodes = self.grid.adjacent_nodes_at_node
        self.node_east = neighbor_nodes[:, 0]
        self.node_north = neighbor_nodes[:, 1]
        self.node_west = neighbor_nodes[:, 2]
        self.node_south = neighbor_nodes[:, 3]

        # Find the neighbor links
        self.active_links = self.grid.active_links
        neighbor_links = links.neighbors_at_link(
            self.grid.shape, np.arange(self.grid.number_of_links))
        self.link_east = neighbor_links[:, 0]
        self.link_north = neighbor_links[:, 1]
        # This is to fix a bug in links.neighbors_at_link
        self.link_north[self.link_north == self.grid.number_of_links] = -1
        self.link_west = neighbor_links[:, 2]
        self.link_south = neighbor_links[:, 3]

        # Find obliquely neighbor vertical links from horizontal links
        # and obliquely neighbor horizontal links from vertical links
        east_link_at_node = self.grid.links_at_node[:, 0]
        north_link_at_node = self.grid.links_at_node[:, 1]
        west_link_at_node = self.grid.links_at_node[:, 2]
        south_link_at_node = self.grid.links_at_node[:, 3]
        west_node_at_horizontal_link = self.grid.nodes_at_link[:, 0]
        east_node_at_horizontal_link = self.grid.nodes_at_link[:, 1]
        south_node_at_vertical_link = self.grid.nodes_at_link[:, 0]
        north_node_at_vertical_link = self.grid.nodes_at_link[:, 1]
        self.vertical_link_NE = north_link_at_node[
            east_node_at_horizontal_link[self.horizontal_active_links]]
        self.vertical_link_SE = south_link_at_node[
            east_node_at_horizontal_link[self.horizontal_active_links]]
        self.vertical_link_NW = north_link_at_node[
            west_node_at_horizontal_link[self.horizontal_active_links]]
        self.vertical_link_SW = south_link_at_node[
            west_node_at_horizontal_link[self.horizontal_active_links]]
        self.horizontal_link_NE = east_link_at_node[
            north_node_at_vertical_link[self.vertical_active_links]]
        self.horizontal_link_SE = east_link_at_node[
            south_node_at_vertical_link[self.vertical_active_links]]
        self.horizontal_link_NW = west_link_at_node[
            north_node_at_vertical_link[self.vertical_active_links]]
        self.horizontal_link_SW = west_link_at_node[
            south_node_at_vertical_link[self.vertical_active_links]]

        # Process boundary links
        bound_north = np.where(self.link_north == -1)
        bound_south = np.where(self.link_south == -1)
        bound_east = np.where(self.link_east == -1)
        bound_west = np.where(self.link_west == -1)
        self.link_north[bound_north] = bound_north
        self.link_south[bound_south] = bound_south
        self.link_east[bound_east] = bound_east
        self.link_west[bound_west] = bound_west

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
            self.cip_2d_M_advection(
                self.h,
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

            self.cip_2d_M_advection(
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

            self.cip_2d_M_advection(
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

            self.cip_2d_M_advection(
                self.C,
                self.dCdx,
                self.dCdy,
                self.u_node,
                self.v_node,
                wet_nodes,
                self.horizontal_up_nodes[wet_nodes],
                self.horizontal_down_nodes[wet_nodes],
                self.vertical_up_nodes[wet_nodes],
                self.vertical_down_nodes[wet_nodes],
                dx,
                self.dt_local,
                out_f=self.C_temp,
                out_dfdx=self.dCdx_temp,
                out_dfdy=self.dCdy_temp)

            # applying slipping condition to velocities at partial wet links
            self.update_values()
            self.map_values(self.h, self.u, self.v, self.C, self.eta,
                            self.h_link, self.u_node, self.v_node, self.C_link)
            self.slip_condition_at_head(
                self.u_node,
                self.h,
                partial_wet_nodes,
                self.node_north[partial_wet_nodes],
                self.node_south[partial_wet_nodes],
                self.node_east[partial_wet_nodes],
                self.node_west[partial_wet_nodes],
                out=self.u_node_temp)
            self.slip_condition_at_head(
                self.v_node,
                self.h,
                partial_wet_nodes,
                self.node_north[partial_wet_nodes],
                self.node_south[partial_wet_nodes],
                self.node_east[partial_wet_nodes],
                self.node_west[partial_wet_nodes],
                out=self.v_node_temp)

            self.update_up_down_links_and_nodes()

            self.cip_2d_M_advection(
                self.h,
                self.dhdx,
                self.dhdy,
                self.u_node_temp,
                self.v_node_temp,
                partial_wet_nodes,
                self.horizontal_up_nodes[partial_wet_nodes],
                self.horizontal_down_nodes[partial_wet_nodes],
                self.vertical_up_nodes[partial_wet_nodes],
                self.vertical_down_nodes[partial_wet_nodes],
                dx,
                self.dt_local,
                out_f=self.h_temp,
                out_dfdx=self.dhdx_temp,
                out_dfdy=self.dhdy_temp)

            self.cip_2d_M_advection(
                self.C,
                self.dCdx,
                self.dCdy,
                self.u_node_temp,
                self.v_node_temp,
                partial_wet_nodes,
                self.horizontal_up_nodes[partial_wet_nodes],
                self.horizontal_down_nodes[partial_wet_nodes],
                self.vertical_up_nodes[partial_wet_nodes],
                self.vertical_down_nodes[partial_wet_nodes],
                dx,
                self.dt_local,
                out_f=self.C_temp,
                out_dfdx=self.dCdx_temp,
                out_dfdy=self.dCdy_temp)

            # update values after calculating advection terms
            # map node values to links, and link values to nodes.
            self.update_values()
            self.map_values(self.h, self.u, self.v, self.C, self.eta,
                            self.h_link, self.u_node, self.v_node, self.C_link)
            self.update_up_down_links_and_nodes()

            # calculate non-advection terms
            self.copy_values_to_temp()
            for i in range(5):
                self.map_values(self.h_temp, self.u_temp, self.v_temp,
                                self.C_temp, self.eta_temp, self.h_link_temp,
                                self.u_node_temp, self.v_node_temp,
                                self.C_link_temp)

                # calculate non-advection terms on wet grids
                self.calc_nonadvection_terms(
                    self.h_temp, self.u_temp, self.v_temp, self.C_temp,
                    self.eta_temp, self.h_link_temp, self.u_node_temp,
                    self.v_node_temp, self.C_link_temp, wet_nodes,
                    wet_horizontal_links, wet_vertical_links)

                self.cip_2d_nonadvection(
                    self.h,
                    self.dhdx,
                    self.dhdy,
                    self.G_h,
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

                self.cip_2d_nonadvection(
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

                self.cip_2d_nonadvection(
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

                self.cip_2d_nonadvection(
                    self.C,
                    self.dCdx,
                    self.dCdy,
                    self.G_C,
                    self.u_node,
                    self.v_node,
                    wet_nodes,
                    self.horizontal_up_nodes[wet_nodes],
                    self.horizontal_down_nodes[wet_nodes],
                    self.vertical_up_nodes[wet_nodes],
                    self.vertical_down_nodes[wet_nodes],
                    dx,
                    self.dt_local,
                    out_f=self.C_temp,
                    out_dfdx=self.dCdx_temp,
                    out_dfdy=self.dCdy_temp)

                # only h and C are calculated on partial wet grids
                # calculate non-advection terms on wet grids
                self.calc_nonadvection_terms(
                    self.h_temp, self.u_temp, self.v_temp, self.C_temp,
                    self.eta_temp, self.h_link_temp, self.u_node_temp,
                    self.v_node_temp, self.C_link_temp, partial_wet_nodes,
                    partial_wet_horizontal_links, partial_wet_vertical_links)

                self.cip_2d_nonadvection(
                    self.h,
                    self.dhdx,
                    self.dhdy,
                    self.G_h,
                    self.u_node,
                    self.v_node,
                    partial_wet_nodes,
                    self.horizontal_up_nodes[partial_wet_nodes],
                    self.horizontal_down_nodes[partial_wet_nodes],
                    self.vertical_up_nodes[partial_wet_nodes],
                    self.vertical_down_nodes[partial_wet_nodes],
                    dx,
                    self.dt_local,
                    out_f=self.h_temp,
                    out_dfdx=self.dhdx_temp,
                    out_dfdy=self.dhdy_temp)

                self.cip_2d_nonadvection(
                    self.C,
                    self.dCdx,
                    self.dCdy,
                    self.G_C,
                    self.u_node,
                    self.v_node,
                    partial_wet_nodes,
                    self.horizontal_up_nodes[partial_wet_nodes],
                    self.horizontal_down_nodes[partial_wet_nodes],
                    self.vertical_up_nodes[partial_wet_nodes],
                    self.vertical_down_nodes[partial_wet_nodes],
                    dx,
                    self.dt_local,
                    out_f=self.C_temp,
                    out_dfdx=self.dCdx_temp,
                    out_dfdy=self.dCdy_temp)

                # Calculate deposition/erosion
                self.eta_temp = self.eta + self.dt_local * self.G_eta

            # update values
            self.update_values()

            # Calculate diffusion term of momentum
            self.cip_2d_diffusion(
                self.u,
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

            # apply the shock dissipation scheme
            self.shock_dissipation(
                self.C,
                self.h,
                wet_nodes,
                self.node_north,
                self.node_south,
                self.node_east,
                self.node_west,
                self.dt_local,
                out=self.C_temp)

            self.shock_dissipation(
                self.u,
                self.h_link,
                wet_horizontal_links,
                self.link_north,
                self.link_south,
                self.link_east,
                self.link_west,
                self.dt_local,
                out=self.u_temp)

            self.shock_dissipation(
                self.v,
                self.h_link,
                wet_vertical_links,
                self.link_north,
                self.link_south,
                self.link_east,
                self.link_west,
                self.dt_local,
                out=self.v_temp)

            self.shock_dissipation(
                self.h,
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

    def slip_condition_at_head(self,
                               U,
                               h,
                               partial_wet,
                               north,
                               south,
                               east,
                               west,
                               out=None):
        """Applying the slip condition at head. In this scheme, velocities
           at partial wet links are "copied" from the wettest (the largest h)
           links
           
           Parameters
           ---------------
           U : ndarray, float
             horizontal or vertical velocity
           h : ndarray, float
             flow depth
           partial_wet : ndarray, int
             indeces of partial wet grids at which flow depth is below the wet
             criterion but the adjacent grid(s) is wet.
           north : ndarray, int
             indeces of grids locating north of partial wet grids
           south : ndarray, int
             indeces of grids locating south of partial wet grids
           east : ndarray, int
             indeces of grids locating east of partial wet grids
           west : ndarray, int
             indeces of grids locating west of partial wet grids
           out : ndarray, float
             outputs (updated flow velocity U)

           Return
           ----------------------
           out : ndarray, float
            updated flow velocity where the slip conditon is applied
        """

        if out is None:
            out = np.zeros(U.shape)

        wettest = np.empty(partial_wet.shape, dtype=('int64'))
        wettest_at_north = np.where((h[north] > h[south]) & (
            h[north] > h[east]) & (h[north] > h[west]))
        wettest_at_east = np.where((h[east] > h[south]) &
                                   (h[east] > h[north]) & (h[east] > h[west]))
        wettest_at_west = np.where((h[west] > h[north]) & (h[west] > h[east]) &
                                   (h[west] > h[south]))
        wettest[:] = south[:]
        wettest[wettest_at_north] = north[wettest_at_north]
        wettest[wettest_at_east] = east[wettest_at_east]
        wettest[wettest_at_west] = west[wettest_at_west]

        out[partial_wet] = U[wettest]

        return out

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
        self.find_horizontal_up_down_nodes(
            self.u_node,
            out_up=self.horizontal_up_nodes,
            out_down=self.horizontal_down_nodes)

        self.find_vertical_up_down_nodes(
            self.v_node,
            out_up=self.vertical_up_nodes,
            out_down=self.vertical_down_nodes)

        self.find_horizontal_up_down_links(
            self.u,
            out_up=self.horizontal_up_links,
            out_down=self.horizontal_down_links)

        self.find_vertical_up_down_links(
            self.v,
            out_up=self.vertical_up_links,
            out_down=self.vertical_down_links)

    def find_wet_grids(
            self,
            h,
            core,
            north,
            south,
            east,
            west,
    ):
        """Find wet and partial wet nodes or links
           In this model, "dry" nodes or links are not subject to calculate
           wet grids (nodes and links) including partial wet grids are
           considered in the model calculation. "wet" is judged by the flow 
           depth (> h_w), and "partial wet" is a dry wet but the upcurrent
           grid is wet.

           Parameters
           --------------------------
           h : ndarray
               flow height values for detecting wet and dry grids

           core : ndarray
               ndarry indicating indeces of core nodes
               or links

           horizontal_up : ndarray
               ndarray indicating horizontally upcurrent nodes or links.

           vertical_up : ndarray
               ndarray indicating vertically upcurrent nodes or links.

           Returns
           -------------------------
           out_wet : ndarray, int
               ndarray indicating wet grids. Grids (node or link) showing h
               value larger than the threshold(h_w) value are judged as
               wet grid
           
           out_partial_wet : ndarray, int
               ndarray indicating partially wet grids. Grids (node or link)
               showing h value lower than the threshold(h_w) value but an 
               upcurrent node or a link is a wet grid

        """
        # wet_check = (h[core] > self.h_w) | (h[horizontal_up][core] >
        #                                     self.h_w) | (h[vertical_up][core] >
        #                                                  self.h_w)
        # wet_check = np.where((h[core] > self.h_w) | (h[north][core] > self.h_w)
        #                      | (h[south][core] > self.h_w)
        #                      | (h[east][core] > self.h_w)
        #                      | (h[west][core] > self.h_w))
        wet_check = np.where(h[core] > self.h_w)
        out_wet = core[wet_check]
        partial_wet_check = np.where((h[core] < self.h_w) & (
            (h[north][core] > self.h_w)
            | (h[south][core] > self.h_w) | (h[east][core] > self.h_w)
            | (h[west][core] > self.h_w)))
        out_partial_wet = core[partial_wet_check]

        return out_wet, out_partial_wet

    def calc_closure_functions(self, h, u, v, C, h_link, u_node, v_node,
                               C_link):
        """Calculate closure functions for non-advection terms
        """

        # Calculate entrainment rates of water and sediment
        U_node = np.sqrt(u_node**2 + v_node**2)
        U_link = np.sqrt(u**2 + v**2)
        u_star_node = np.sqrt(self.Cf) * U_node
        self.ew_node = self.get_ew(U_node, h, C)
        self.ew_link = self.get_ew(U_link, h_link, C_link)
        self.es = self.get_es(u_star_node)
        self.r0[:] = 1.5

    def cip_2d_diffusion(self,
                         u,
                         v,
                         nu_t,
                         h_active,
                         v_active,
                         north,
                         south,
                         east,
                         west,
                         dx,
                         dt,
                         out_u=None,
                         out_v=None):
        """Caclulate horizontal and vertical diffusion of velocities u and v
        """
        if out_u is None:
            out_u = np.zeros(u.shape)
        if out_v is None:
            out_v = np.zeros(v.shape)

        out_u[h_active] = u[h_active]
        +nu_t * dt * (
            (u[east][h_active] - 2 * u[h_active] + u[west][h_active]) +
            (u[north][h_active] - 2 * u[h_active] + u[south][h_active])
        ) / dx**2

        out_v[v_active] = v[v_active]
        +nu_t * dt * (
            (v[east][v_active] - 2 * v[v_active] + v[west][v_active]) +
            (v[north][v_active] - 2 * v[v_active] + v[south][v_active])
        ) / dx**2

        return out_u, out_v

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

    def get_ew(self, U, h, C, out=None):
        """ calculate entrainemnt coefficient of ambient water to a turbidity
            current layer

            Parameters
            ----------
            U : ndarray
               Flow velocities of ambient water and a turbidity current.
               Row 0 is for an ambient water, and Row 1 is for a turbidity
               current.
            h : ndarray
               Flow heights of ambient water and a turbidity current. Row 0
               is an ambient water, and Row 1 is a turbidity current.
            C : ndarray
               Sediment concentration
            out : ndarray
               Outputs

            Returns
            ---------
            e_w : ndarray
               Entrainment coefficient of ambient water

        """
        if out is None:
            out = np.zeros(U.shape)

        Ri = np.zeros(U.shape)
        flow_exist = np.where((h[:] > self.h_w) & (U > 0.01))
        Ri[flow_exist] = self.R * self.g * C[flow_exist] \
            * h[flow_exist] / U[flow_exist] ** 2
        out[flow_exist] = 0.075 / np.sqrt(
            1 + 718. + Ri[flow_exist]**2.4)  # Parker et al. (1987)

        return out

    def get_ws(self):
        """ Calculate settling velocity of sediment particles
            on the basis of Ferguson and Church (1982)

        Return
        ------------------
        ws : settling velocity of sediment particles [m/s]

        """

        # copy parameters to local variables
        R = self.R
        g = self.g
        Ds = self.Ds
        nu = self.nu

        # Coefficients for natural sands
        C_1 = 18.
        C_2 = 1.0

        ws = R * g * Ds**2 / (C_1 * nu + (0.75 * C_2 * R * g * Ds**3)**0.5)

        return ws

    def get_es(self, u_star, out=None):
        """ Calculate entrainment rate of basal sediment to suspension
            Based on Garcia and Parker (1991)

            Parameters
            --------------
            u_star : ndarray
                flow shear velocity
            out : ndarray
                Outputs (entrainment rate of basal sediment)

            Returns
            ---------------
            out : ndarray
                dimensionless entrainment rate of basal sediment into
                suspension
        """

        if out is None:
            out = np.zeros(u_star.shape)

        # basic parameters
        R = self.R
        g = self.g
        Ds = self.Ds
        nu = self.nu
        ws = self.ws

        # calculate subordinate parameters
        Rp = np.sqrt(R * g * Ds) * Ds / nu
        sus_index = u_star / ws

        # coefficients for calculation
        a = 7.8 * 10**-7
        alpha = 0.6
        p = 0.1

        # calculate entrainemnt rate
        Z = sus_index * Rp**alpha
        out[:] = p * a * Z**5 / (1 + (a / 0.3) * Z**5)

        return out

    def calc_nonadvection_terms(self, h, U, V, C, eta, h_link, u_node, v_node,
                                C_link, core_nodes, link_horiz, link_vert):
        """calculate non-advection terms
        """

        # copy class attributes to local variables
        dx = self.grid.dx
        u = U[link_horiz]
        u_on_vert = U[link_vert]
        v = V[link_vert]
        v_on_horiz = V[link_horiz]
        Cf = self.Cf
        Rg = self.R * self.g
        ws = self.ws
        r0 = np.zeros(h.shape)
        # core_nodes = self.core_nodes
        node_north = self.node_north[core_nodes]
        node_south = self.node_south[core_nodes]
        node_east = self.node_east[core_nodes]
        node_west = self.node_west[core_nodes]
        # link_horiz = self.horizontal_active_links
        # link_vert = self.vertical_active_links
        link_north = self.link_north
        link_south = self.link_south
        link_east = self.link_east
        link_west = self.link_west

        # calculate closure functions
        vel_at_node = np.sqrt(u_node**2 + v_node**2)
        vel_at_link = np.sqrt(U**2 + V**2)
        u_star_at_node = np.sqrt(self.Cf) * vel_at_node
        ew_node = self.get_ew(vel_at_node, h, C)
        ew_link = self.get_ew(vel_at_link, h_link, C_link)
        es = self.get_es(u_star_at_node)
        r0[:] = 1.5

        # Calculate topographic gradient
        eta_grad_at_link = self.grid.calc_grad_at_link(eta)
        eta_grad_x = eta_grad_at_link[link_horiz]
        eta_grad_y = eta_grad_at_link[link_vert]

        # Calculate shear stress
        if self.flow_type == '3eq':
            u_star_2 = Cf * u * np.sqrt(u**2 + v_on_horiz**2)
            v_star_2 = Cf * v * np.sqrt(u_on_vert**2 + v**2)

        # Calculate non-advection terms
        self.G_h[core_nodes] = ew_node[core_nodes] * np.sqrt(
            u_node[core_nodes]**2 + v_node[core_nodes]**2) - h[core_nodes] * (
                (v_node[node_north] - v_node[node_south]) / (2 * dx) +
                (u_node[node_east] - u_node[node_west]) / (2 * dx))

        self.G_u[link_horiz] = -Rg * C_link[
            link_horiz] * eta_grad_x - 0.5 * Rg * h_link[link_horiz] * (
                C_link[link_east][link_horiz] - C_link[link_west][link_horiz]
            ) / (2 * dx) - Rg * C_link[link_horiz] * (
                h_link[link_east][link_horiz] - h_link[link_west][link_horiz]
            ) / (2 * dx) - u_star_2 / h_link[link_horiz] - ew_link[
                link_horiz] * u * np.sqrt(u**2 +
                                          v_on_horiz**2) / h_link[link_horiz]

        self.G_v[link_vert] = -Rg * C_link[
            link_vert] * eta_grad_y - 0.5 * Rg * h_link[link_vert] * (
                C_link[link_north][link_vert] - C_link[link_south][link_vert]
            ) / (2 * dx) - Rg * C_link[link_vert] * (
                h_link[link_north][link_vert] - h_link[link_south][link_vert]
            ) / (2 * dx) - v_star_2 / h_link[link_vert] - ew_link[
                link_vert] * v * np.sqrt(u_on_vert**2 +
                                         v**2) / h_link[link_vert]

        sedimentation_rate = ws * (
            r0[core_nodes] * C[core_nodes] - es[core_nodes])

        self.G_C[core_nodes] = (
            -sedimentation_rate - ew_node[core_nodes] * C[core_nodes] *
            np.sqrt(u_node[core_nodes]**2 + v_node[core_nodes]**2)
        ) / h[core_nodes]

        self.G_eta[core_nodes] = sedimentation_rate / (1 - self.lambda_p)

    def map_values(self, h, u, v, C, eta, h_link, u_node, v_node, C_link):
        """map parameters at nodes to links, and those at links to nodes
        """

        grid = self.grid

        # Map values of horizontal links to vertical links, and
        # values of vertical links to horizontal links.
        # Horizontal velocity is only updated at horizontal links,
        # and vertical velocity is updated at vertical links during
        # CIP procedures. Therefore we need to map those values each other.
        u[self.vertical_active_links] = (
            u[self.horizontal_link_NE] + u[self.horizontal_link_SE] +
            u[self.horizontal_link_NW] + u[self.horizontal_link_SW]) / 4.0
        v[self.horizontal_active_links] = (
            v[self.vertical_link_NE] + v[self.vertical_link_SE] +
            v[self.vertical_link_NW] + v[self.vertical_link_SW]) / 4.0

        # adjust illeagal values
        h[np.where(h < self.h_init)] = self.h_init
        C[np.where(C < 0)] = 0.0

        # map link values (u, v) to nodes
        grid.map_mean_of_links_to_node(u, out=u_node)
        grid.map_mean_of_links_to_node(v, out=v_node)

        # map node values (h, C, eta) to links
        grid.map_mean_of_link_nodes_to_link(h, out=h_link)
        grid.map_mean_of_link_nodes_to_link(C, out=C_link)

    def cip_2d_M_advection(self,
                           f,
                           dfdx,
                           dfdy,
                           u,
                           v,
                           core,
                           h_up,
                           h_down,
                           v_up,
                           v_down,
                           dx,
                           dt,
                           out_f=None,
                           out_dfdx=None,
                           out_dfdy=None):
        """Calculate one time step using M-type 2D cip method
        """

        # First, the variables out and temp are allocated to
        # store the calculation results

        if out_f is None:
            out_f = np.empty(f.shape)
        if out_dfdx is None:
            out_dfdx = np.empty(dfdx.shape)
        if out_dfdy is None:
            out_dfdy = np.empty(dfdy.shape)

        # 1st step for horizontal advection
        D_x = -np.where(u > 0., 1.0, -1.0) * dx
        xi_x = -u * dt
        a = (dfdx[core] + dfdx[h_up]) / (D_x[core] ** 2)\
            + 2 * (f[core] - f[h_up]) / (D_x[core] ** 3)
        b = 3 * (f[h_up] - f[core]) / (D_x[core] ** 2)\
            - (2 * dfdx[core] + dfdx[h_up]) / D_x[core]
        out_f[core] = a * (xi_x[core] ** 3) + b * (xi_x[core] ** 2)\
            + dfdx[core] * xi_x[core] + f[core]
        out_dfdx[core] = 3 * a * (xi_x[core] ** 2) + 2 * b * xi_x[core]\
            + dfdx[core]
        out_dfdy[core] = dfdy[core] - xi_x[core] / \
            D_x[core] * (dfdy[core] - dfdy[h_up])

        # 2nd step for vertical advection
        D_y = -np.where(v > 0., 1.0, -1.0) * dx
        xi_y = -v * dt
        a = (out_dfdy[core] + out_dfdy[v_up]) / (D_y[core] ** 2)\
            + 2 * (out_f[core] - out_f[v_up]) / (D_y[core] ** 3)
        b = 3 * (out_f[v_up] - out_f[core]) / (D_y[core] ** 2)\
            - (2 * out_dfdy[core] + out_dfdy[v_up]) / D_y[core]
        out_f[core] = a * (xi_y[core] ** 3) + b * (xi_y[core] ** 2)\
            + out_dfdy[core] * xi_y[core] + out_f[core]
        out_dfdy[core] = 3 * a * (xi_y[core] ** 2) + 2 * b * xi_y[core]\
            + out_dfdy[core]
        out_dfdx[core] = out_dfdx[core] - xi_y[core] / \
            D_y[core] * (out_dfdx[core] - out_dfdx[v_up])

        return out_f, out_dfdx, out_dfdy

    def cip_2d_nonadvection(self,
                            f,
                            dfdx,
                            dfdy,
                            G,
                            u,
                            v,
                            core,
                            h_up,
                            h_down,
                            v_up,
                            v_down,
                            dx,
                            dt,
                            out_f=None,
                            out_dfdx=None,
                            out_dfdy=None):

        if out_f is None:
            out_f = np.zeros(f.shape)
        if out_dfdx is None:
            out_dfdx = np.zeros(dfdx.shape)
        if out_dfdy is None:
            out_dfdy = np.zeros(dfdy.shape)

        D_x = -np.where(u > 0., 1.0, -1.0) * dx
        xi_x = -u * dt
        D_y = -np.where(v > 0., 1.0, -1.0) * dx
        xi_y = -v * dt

        # non-advection term
        out_f[core] = f[core] + G[core] * dt
        out_dfdx[core] = dfdx[core] + ((out_f[h_down] - f[h_down]) \
                                       - (out_f[h_up] - f[h_up])) / \
            (-2 * D_x[core]) - dfdx[core] * \
            (xi_x[h_down] - xi_x[h_up]) / (2 * D_x[core])

        out_dfdy[core] = dfdy[core]
        +((out_f[v_down] - f[v_down]) -
          (out_f[v_up] - f[v_up])) / (-2 * D_y[core]) - dfdy[core] * (
              xi_y[v_down] - xi_y[v_up]) / (2 * D_y[core])

        return out_f, out_dfdx, out_dfdy

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
        kappa=0.0001,
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
