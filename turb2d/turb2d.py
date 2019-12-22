"""A component of landlab that simulates a turbidity current on 2D grids

This component simulates turbidity currents using the 2-D numerical model of
shallow-water equations over topography on the basis of 3 equation model of
Parker et al. (1986). This component is based on the landlab component
 overland_flow that was written by Jordan Adams.

.. codeauthor: : Hajime Naruse

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
        alpha=0.1,
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
from .gridutils import set_up_neighbor_arrays, update_up_down_links_and_nodes
from .gridutils import map_values, map_links_to_nodes, map_nodes_to_links
from .wetdry import find_wet_grids, process_partial_wet_grids
from .sediment_func import get_es, get_ew, get_ws
from .cip import cip_2d_diffusion, shock_dissipation, update_gradient
from .cip import rcip_2d_M_advection, cip_2d_nonadvection, cip_2d_M_advection
from landlab.io.native_landlab import save_grid, load_grid
from landlab.io.netcdf import write_netcdf
from landlab.grid.structured_quad import links
from landlab.utils.decorators import use_file_name_or_kwds
from landlab import Component, FieldError, RasterModelGrid
import numpy as np
import time
from osgeo import gdal, gdalconst


class TurbidityCurrent2D(Component):
    """Simulate a turbidity current using the CIP method.

    Landlab component that simulates turbidity current using the CIP method
    for solving the 2D shallow water equations.

    This component calculates flow depth, shear stress across any raster grid.
    Default input file is named "turbidity_current_input.txt" and is
    contained in the turb2d directory.

    The primary method of this class is: func: 'run_one_step'
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
                 h_init=0.0001,
                 h_w=0.01,
                 alpha=0.05,
                 Cf=0.004,
                 g=9.81,
                 R=1.65,
                 Ds=100 * 10**-6,
                 lambda_p=0.4,
                 r0=1.5,
                 nu=1.010 * 10**-6,
                 kappa=0.001,
                 implicit_num=20,
                 C_init=0.00001,
                 gamma=0.35,
                 **kwds):
        """Create a debris-flow component.

        Parameters
        ----------
        grid: RasterModelGrid
            A landlab grid.
        h_init: float, optional
            Thickness of initial thin layer of flow to prevent divide by zero
            errors(m).
        h_w: float, optional
            Thickness of flow to judge "wet" nodes and links(m).
        alpha: float, optional
            Time step coefficient
        Cf: float, optional
            Dimensionless Chezy friction coefficient.
        g: float, optional
            Acceleration due to gravity(m/s ^ 2)
        R: float, optional
            Submerged specific density(rho_s/rho_f - 1)(1)
        Ds: float, optional
            Sediment diameter(m)
        lambda_p: float, optional
            Bed sediment porosity(1)
        nu: float, optional
            Kinematic viscosity of water(at 293K)
        kappa: float, optional
            Artificial viscosity. This value, alpha and h_w affect to 
            calculation stability.
        implicit_num: float, optional
            Maximum number of loops for implicit calculation.
        r0: float, optional
            Ratio of near-bed concentration to layer-averaged concentration
        C_init: float, optional
            Minimum value of sediment concentration.
        gamma: float, optional
            Coefficient for calculating flow front between dry and wet grids.

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
        self.h_w = h_w
        self.nu = nu
        self.kappa = kappa
        self.r0 = r0
        self.lambda_p = lambda_p
        self.implicit_num = implicit_num
        self.C_init = C_init
        self.gamma = gamma

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

        try:
            self.u_node = grid.add_zeros(
                'flow__horizontal_velocity_at_node',
                at='node',
                units=self._var_units['flow__horizontal_velocity'])
            self.v_node = grid.add_zeros(
                'flow__vertical_velocity_at_node',
                at='node',
                units=self._var_units['flow__vertical_velocity'])
        except FieldError:
            self.u_node = grid.at_node['flow__horizontal_velocity_at_node']
            self.v_node = grid.at_node['flow__vertical_velocity_at_node']

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

        self.horizontal_up_nodes = np.zeros(grid.number_of_nodes, dtype=np.int)
        self.vertical_up_nodes = np.zeros(grid.number_of_nodes, dtype=np.int)
        self.horizontal_down_nodes = np.zeros(grid.number_of_nodes,
                                              dtype=np.int)
        self.vertical_down_nodes = np.zeros(grid.number_of_nodes, dtype=np.int)

        self.horizontal_up_links = np.zeros(grid.number_of_links, dtype=np.int)
        self.vertical_up_links = np.zeros(grid.number_of_links, dtype=np.int)
        self.horizontal_down_links = np.zeros(grid.number_of_links,
                                              dtype=np.int)
        self.vertical_down_links = np.zeros(grid.number_of_links, dtype=np.int)

        self.east_link_at_node = np.zeros(grid.number_of_nodes, dtype=np.int)
        self.north_link_at_node = np.zeros(grid.number_of_nodes, dtype=np.int)
        self.west_link_at_node = np.zeros(grid.number_of_nodes, dtype=np.int)
        self.south_link_at_node = np.zeros(grid.number_of_nodes, dtype=np.int)
        self.west_node_at_horizontal_link = np.zeros(grid.number_of_links,
                                                     dtype=np.int)
        self.east_node_at_horizontal_link = np.zeros(grid.number_of_links,
                                                     dtype=np.int)
        self.south_node_at_vertical_link = np.zeros(grid.number_of_links,
                                                    dtype=np.int)
        self.north_node_at_vertical_link = np.zeros(grid.number_of_links,
                                                    dtype=np.int)

        # set variables to be used for processing wet/dry grids
        self.wet_nodes = None
        self.wet_horizontal_links = None
        self.wet_vertical_links = None
        self.horizontally_partial_wet_nodes = None
        self.vertically_partial_wet_nodes = None
        self.horizontally_wettest_nodes = None
        self.vertically_wettest_nodes = None
        self.partial_wet_horizontal_links = None
        self.partial_wet_vertical_links = None
        self.horizontal_direction_wettest = None
        self.vertical_direction_wettest = None
        self.wet_pwet_nodes = None
        self.wet_pwet_links = None
        self.wet_pwet_horizontal_links = None
        self.wet_pwet_vertical_links = None
        self.dry_nodes = None
        self.dry_links = None

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

    def calc_time_step(self):
        """Calculate time step
        """

        sqrt_RCgh = np.sqrt(self.R * self.C * self.g * self.h)

        dt_local = self.alpha * self._grid.dx \
            / np.amax(np.array([np.amax(np.abs(self.u_node) + sqrt_RCgh),
                                np.amax(np.abs(self.v_node) + sqrt_RCgh),
                                1.0]))

        if self.first_step is True:
            dt_local *= 0.1

        return dt_local

    # @profile
    def run_one_step(self, dt=None):
        """Generate debris flow across a grid.

        For one time step, this generates 'turbidity current' across
        a given grid by calculating flow height and concentration at each node
        and velocities at each link.

        Outputs flow depth, concentration, horizontal and vertical
        velocity values through time at every point in the input grid.

        Parameters
        ----------------
        dt: float, optional
            time to finish calculation of this step. Inside the model,
            local value of dt is used for stability of calculation.
            If dt = None, dt is set to be equal to local dt.
        """

        # DH adds a loop to enable an imposed tstep while maintaining stability
        local_elapsed_time = 0.
        if dt is None:
            dt = np.inf  # to allow the loop to begin
        self.dt = dt

        # First, we check and see if the neighbor arrays have been
        # initialized
        if self.neighbor_flag is False:
            set_up_neighbor_arrays(self)

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
        find_wet_grids(self, self.h)
        map_values(self, self.h, self.u, self.v, self.Ch, self.eta,
                   self.h_link, self.u_node, self.v_node, self.Ch_link, self.U,
                   self.U_node)
        self.copy_values_to_temp()
        update_up_down_links_and_nodes(self)

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
            find_wet_grids(self, self.h)

            # calculation of advecton terms of momentum (u and v) equations
            # by CIP method
            self._advection_phase()

            # calculate non-advection terms using implicit method
            self._nonadvection_phase()

            # calculate deposition/erosion
            self._deposition_phase()

            # Calculate diffusion term of momentum
            self._diffusion_phase()

            # Process partial wet grids
            self._process_wet_dry_boundary()

            # apply the shock dissipation scheme
            self._shock_dissipation_phase()

            #the end of the loop of one local time step
            self.first_step = False

            # Calculation is terminated if global dt is not specified.
            if dt is np.inf:
                break
            local_elapsed_time += self.dt_local

        # This is the end of the calculation
        # Update bed thickness and record results in the grid
        self.elapsed_time += local_elapsed_time
        self.bed_thick = self.eta - self.eta_init
        self.copy_values_to_grid()

    def _advection_phase(self):
        """Calculate advection phase of the model
           Advection of flow velocities is calculated by CIP
        """
        rcip_2d_M_advection(
            self.u,
            self.dudx,
            self.dudy,
            self.u,
            self.v,
            self.wet_pwet_horizontal_links,
            self.horizontal_up_links[self.wet_pwet_horizontal_links],
            self.horizontal_down_links[self.wet_pwet_horizontal_links],
            self.vertical_up_links[self.wet_pwet_horizontal_links],
            self.vertical_down_links[self.wet_pwet_horizontal_links],
            self.grid.dx,
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
            self.wet_pwet_vertical_links,
            self.horizontal_up_links[self.wet_pwet_vertical_links],
            self.horizontal_down_links[self.wet_pwet_vertical_links],
            self.vertical_up_links[self.wet_pwet_vertical_links],
            self.vertical_down_links[self.wet_pwet_vertical_links],
            self.grid.dx,
            self.dt_local,
            out_f=self.v_temp,
            out_dfdx=self.dvdx_temp,
            out_dfdy=self.dvdy_temp)

        # update values after calculating advection terms
        # map node values to links, and link values to nodes.
        # self.h_temp[:] = self.h[:]
        # self.Ch_temp[:] = self.Ch[:]
        map_values(self, self.h_temp, self.u_temp, self.v_temp, self.Ch_temp,
                   self.eta_temp, self.h_link_temp, self.u_node_temp,
                   self.v_node_temp, self.Ch_link_temp, self.U_temp,
                   self.U_node_temp)
        self.update_values()
        update_up_down_links_and_nodes(self)

    def _nonadvection_phase(self):
        """Calculate non-advection phase of the model
           Pressure terms for velocities and mass conservation equations
           are solved implicitly
        """
        self.h_prev[:] = self.h_temp[:]
        self.Ch_prev[:] = self.Ch_temp[:]
        converge = 10.0
        count = 0
        while ((converge > 1.0 * 10**-15) and (count < self.implicit_num)):

            self.calc_G_u(self.h_temp, self.h_link_temp, self.u_temp,
                          self.v_temp, self.Ch_temp, self.Ch_link_temp,
                          self.eta_temp, self.U_temp,
                          self.wet_horizontal_links)
            self.calc_G_v(self.h_temp, self.h_link_temp, self.u_temp,
                          self.v_temp, self.Ch_temp, self.Ch_link_temp,
                          self.eta_temp, self.U_temp, self.wet_vertical_links)
            self.u_temp[self.wet_horizontal_links] = self.u[
                self.wet_horizontal_links] + self.G_u[
                    self.wet_horizontal_links] * self.dt_local
            self.v_temp[self.wet_vertical_links] = self.v[
                self.wet_vertical_links] + self.G_v[
                    self.wet_vertical_links] * self.dt_local

            # cip_2d_nonadvection(
            #     self.u,
            #     self.dudx,
            #     self.dudy,
            #     self.G_u,
            #     self.u_temp,
            #     self.v_temp,
            #     self.wet_horizontal_links,
            #     self.horizontal_up_links[self.wet_horizontal_links],
            #     self.horizontal_down_links[self.wet_horizontal_links],
            #     self.vertical_up_links[self.wet_horizontal_links],
            #     self.vertical_down_links[self.wet_horizontal_links],
            #     self.grid.dx,
            #     self.dt_local,
            #     out_f=self.u_temp,
            #     out_dfdx=self.dudx_temp,
            #     out_dfdy=self.dudy_temp)

            # cip_2d_nonadvection(
            #     self.v,
            #     self.dvdx,
            #     self.dvdy,
            #     self.G_v,
            #     self.u_temp,
            #     self.v_temp,
            #     self.wet_vertical_links,
            #     self.horizontal_up_links[self.wet_vertical_links],
            #     self.horizontal_down_links[self.wet_vertical_links],
            #     self.vertical_up_links[self.wet_vertical_links],
            #     self.vertical_down_links[self.wet_vertical_links],
            #     self.grid.dx,
            #     self.dt_local,
            #     out_f=self.v_temp,
            #     out_dfdx=self.dvdx_temp,
            #     out_dfdy=self.dvdy_temp)

            map_links_to_nodes(
                self,
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
            map_nodes_to_links(self, self.h_temp, self.Ch_temp, self.eta_temp,
                               self.h_link_temp, self.Ch_link_temp)

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
            ) / self.wet_nodes.shape[0]

            self.h_prev[:] = self.h_temp[:]
            self.Ch_prev[:] = self.Ch_temp[:]

            count += 1

        if count == self.implicit_num:
            print('Implicit calculation did not converge')

        # update gradient terms
        update_gradient(self.u, self.u_temp, self.dudx, self.dudy,
                        self.wet_pwet_horizontal_links,
                        self.link_north[self.wet_pwet_horizontal_links],
                        self.link_south[self.wet_pwet_horizontal_links],
                        self.link_east[self.wet_pwet_horizontal_links],
                        self.link_south[self.wet_pwet_horizontal_links],
                        self.grid.dx, self.dudx_temp, self.dudy_temp)
        update_gradient(self.v, self.v_temp, self.dvdx, self.dvdy,
                        self.wet_pwet_vertical_links,
                        self.link_north[self.wet_pwet_vertical_links],
                        self.link_south[self.wet_pwet_vertical_links],
                        self.link_east[self.wet_pwet_vertical_links],
                        self.link_south[self.wet_pwet_vertical_links],
                        self.grid.dx, self.dvdx_temp, self.dvdy_temp)

        # Find wet and partial wet grids
        find_wet_grids(self, self.h_temp)

        # update values
        self.update_values()
        map_values(self, self.h, self.u, self.v, self.Ch, self.eta,
                   self.h_link, self.u_node, self.v_node, self.Ch_link, self.U,
                   self.U_node)

    def _deposition_phase(self):
        """Calculate depositional and erosional processes
        """
        self.calc_deposition(self.h,
                             self.Ch,
                             self.u_node,
                             self.v_node,
                             self.eta,
                             self.U_node,
                             out_Ch=self.Ch_temp,
                             out_eta=self.eta_temp)
        self.update_values()
        map_nodes_to_links(self, self.h, self.Ch, self.eta, self.h_link,
                           self.Ch_link)

    def _diffusion_phase(self):
        """Calculate diffusion processes of momentum
        """
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
                         self.grid.dx,
                         self.dt_local,
                         out_u=self.u_temp,
                         out_v=self.v_temp)

        # update values
        self.update_values()
        map_links_to_nodes(self, self.u, self.v, self.u_node, self.v_node,
                           self.U, self.U_node)

    def _process_wet_dry_boundary(self):
        """Calculate processes at wet and dry boundary
        """
        process_partial_wet_grids(self, self.h, self.u, self.v, self.Ch,
                                  self.h_temp, self.u_temp, self.v_temp,
                                  self.Ch_temp)
        update_up_down_links_and_nodes(self)

        # update values
        self.update_values()
        map_values(self, self.h, self.u, self.v, self.Ch, self.eta,
                   self.h_link, self.u_node, self.v_node, self.Ch_link, self.U,
                   self.U_node)

    def _shock_dissipation_phase(self):
        """Calculate shock dissipation phase of the model
        """
        shock_dissipation(
            self.Ch,
            self.Ch * self.h,
            # self.wet_pwet_nodes,
            self.core_nodes,
            self.node_north,
            self.node_south,
            self.node_east,
            self.node_west,
            self.dt_local,
            self.kappa,
            out=self.Ch_temp)

        shock_dissipation(
            self.u,
            self.Ch_link * self.h_link,
            # self.wet_pwet_horizontal_links,
            self.horizontal_active_links,
            self.link_north,
            self.link_south,
            self.link_east,
            self.link_west,
            self.dt_local,
            self.kappa,
            out=self.u_temp)

        shock_dissipation(
            self.v,
            self.Ch_link * self.h_link,
            # self.wet_pwet_vertical_links,
            self.vertical_active_links,
            self.link_north,
            self.link_south,
            self.link_east,
            self.link_west,
            self.dt_local,
            self.kappa,
            out=self.v_temp)

        shock_dissipation(
            self.h,
            self.Ch * self.h,
            # self.wet_pwet_nodes,
            self.core_nodes,
            self.node_north,
            self.node_south,
            self.node_east,
            self.node_west,
            self.dt_local,
            self.kappa,
            out=self.h_temp)

        update_gradient(self.u,
                        self.u_temp,
                        self.dudx,
                        self.dudy,
                        self.wet_pwet_horizontal_links,
                        self.link_north[self.wet_pwet_horizontal_links],
                        self.link_south[self.wet_pwet_horizontal_links],
                        self.link_east[self.wet_pwet_horizontal_links],
                        self.link_west[self.wet_pwet_horizontal_links],
                        self.grid.dx,
                        out_dfdx=self.dudx_temp,
                        out_dfdy=self.dudy_temp)

        update_gradient(self.v,
                        self.v_temp,
                        self.dvdx,
                        self.dvdy,
                        self.wet_pwet_vertical_links,
                        self.link_north[self.wet_pwet_vertical_links],
                        self.link_south[self.wet_pwet_vertical_links],
                        self.link_east[self.wet_pwet_vertical_links],
                        self.link_west[self.wet_pwet_vertical_links],
                        self.grid.dx,
                        out_dfdx=self.dvdx_temp,
                        out_dfdy=self.dvdy_temp)

        # Reset our field values with the newest flow depth and
        # discharge.
        self.update_values()
        map_values(self, self.h, self.u, self.v, self.Ch, self.eta,
                   self.h_link, self.u_node, self.v_node, self.Ch_link, self.U,
                   self.U_node)
        update_up_down_links_and_nodes(self)

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
        self.grid.at_node['flow__horizontal_velocity_at_node'] = self.u_node
        self.grid.at_node['flow__vertical_velocity_at_node'] = self.v_node
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

    def calc_nu_t(self, u, v, h_link, out=None):
        """Calculate eddy viscosity for horizontal diffusion of momentum

           Parameters
           -----------------------

           u: ndarray, float
               horizontal velocity
           v: ndarray, float
               vertical velocity

           Return
           -----------------------
           out: ndarray, float
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

        # flow discharge
        q_x = u * h_link
        q_y = v * h_link

        # adjust flow edge
        q_x[self.partial_wet_horizontal_links] = 0
        q_y[self.partial_wet_vertical_links] = 0

        self.G_h[core_nodes] = ew_node[core_nodes] * U_node[core_nodes] \
            - (q_x[link_east] - q_x[link_west]) / dx \
            - (q_y[link_north] - q_y[link_south]) / dx

    def calc_G_Ch(self, Ch, Ch_link, u, v, core_nodes):
        """Calculate non-advection term for Ch
        """

        # core_nodes = self.core_nodes
        link_north = self.north_link_at_node[core_nodes]
        link_south = self.south_link_at_node[core_nodes]
        link_east = self.east_link_at_node[core_nodes]
        link_west = self.west_link_at_node[core_nodes]
        dx = self.grid.dx

        # flow sediment discharge
        phi_x = u * Ch_link
        phi_y = v * Ch_link

        # adjust flow edge
        phi_x[self.partial_wet_horizontal_links] = 0
        phi_y[self.partial_wet_vertical_links] = 0

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
        #                  0.1)
        ew_link = self.ew_link[link_horiz]
        u_star_2 = self.Cf * u[link_horiz] * U_horiz_link

        self.G_u[link_horiz] = (
            -Rg * Ch_link[link_horiz] * eta_grad_x \
            - 0.5 * Rg *
            (Ch[node_east] * h[node_east] - Ch[node_west] * h[node_west])
             / dx - u_star_2
            - ew_link * U_horiz_link * u[link_horiz]) \
            / h_link[link_horiz]

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

        self.G_v[link_vert] = (
            -Rg * Ch_link[link_vert] * eta_grad_y
            - 0.5 * Rg *
            (Ch[node_north] * h[node_north] - Ch[node_south] * h[node_south])
            /dx - v_star_2
            - ew_link * U_vert_link * v[link_vert]) \
            / h_link[link_vert]

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

    def save_grid(self, filename, clobber=True):
        """save a grid file

           This function saves grid as a pickled file. Although a grid file
           contain all variables, its file size is large.

           Parameters
           ------------------------------
           filename : string
               A file name to store a grid.

           clobber : boolean
               Overwrite an existing file
        """

        save_grid(self.grid, filename, clobber=clobber)

    def save_nc(self, filename):
        """save a grid in netCDF format

           This function saves grid as a netCDF file that can be loaded by
           paraview or VizIt.

           Parameters
           ------------------------------
           filename : string
               A file name to store a grid.

           clobber : boolean
               Overwrite an existing file
        """
        write_netcdf(filename, self.grid)


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
    grid = create_topography(
        length=5000,
        width=2000,
        spacing=10,
        slope_outside=0.2,
        slope_inside=0.1,
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
                            alpha=0.05,
                            kappa=0.001,
                            Ds=100 * 10**-6,
                            h_w=0.001,
                            implicit_num=20,
                            r0=1.5)

    # start calculation
    t = time.time()
    tc.save_nc('test{:04d}.nc'.format(0))
    Ch_init = np.sum(tc.Ch)
    last = 2

    for i in range(1, last + 1):
        tc.run_one_step(dt=1.0)
        tc.save_nc('test{:04d}.nc'.format(i))
        print("", end="\r")
        print("{:.1f}% finished".format(i / last * 100), end='\r')
        if np.sum(tc.Ch) / Ch_init < 0.01:
            break

    print('elapsed time: {} sec.'.format(time.time() - t))
    tc.save_grid('test{:04d}.grid'.format(i), clobber=True)
