"""A component of landlab that simulates a turbidity current on 2D grids

This component simulates turbidity currents using the 2-D numerical model of
shallow-water equations over topography on the basis of 3 equation model of
Parker et al. (1986). 

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
from .gridutils import find_boundary_links_nodes, adjust_negative_values
from .wetdry import find_wet_grids, process_partial_wet_grids
from .sediment_func import get_es, get_ew, get_ws
from .cip import cip_2d_diffusion, shock_dissipation
from .cip import update_gradient, update_gradient2
from .cip import rcip_2d_M_advection, cip_2d_nonadvection, cip_2d_M_advection
from .cip import cip_2d_advection, forester_filter, rcip_2d_advection
from landlab.io.native_landlab import save_grid, load_grid
from landlab.io.netcdf import write_netcdf
from landlab.grid.structured_quad import links
from landlab import Component, FieldError, RasterModelGrid
import numpy as np
import time
from osgeo import gdal, gdalconst
from scipy.ndimage import median_filter
from scipy.integrate import odeint

# import ipdb
# ipdb.set_trace()


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

    def __init__(self,
                 grid,
                 h_init=0.0,
                 p_w=10**(-5),
                 h_w=0.01,
                 alpha=0.1,
                 Cf=0.004,
                 g=9.81,
                 R=1.65,
                 Ds=100 * 10**-6,
                 lambda_p=0.4,
                 r0=1.5,
                 nu=1.010 * 10**-6,
                 kappa2=0.25,
                 kappa4=0.004,
                 nu_a=0.8,
                 implicit_num=50,
                 implicit_threshold=1.0 * 10**-15,
                 C_init=0.0,
                 gamma=0.35,
                 water_entrainment=True,
                 suspension=True,
                 bedload=True,
                 sed_entrainment_func='GP1991field',
                 scheme='CCUP',
                 **kwds):
        """Create a component of turbidity current 

        Parameters
        ----------
        grid: RasterModelGrid
            A landlab grid.
        h_init: float, optional
            Thickness of initial thin layer of flow to prevent divide by zero
            errors(m).
        p_w: float, optional
            Water pressure (Ch^2) to judge "wet" nodes and links(m^2).
        h_w: float, optional
            Minimum flow depth (h) to judge "wet" nodes and links(m).
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
        kappa2: float, optional
            Second order artificial viscosity. This value, alpha and
            h_w affect to calculation stability.
        kappa4: float, optional
            Forth order artificial viscosity. This value, alpha
             and h_w affect to calculation stability.
        nu_a: float, optional
            Artificial viscosity coefficient (0.6-1.0). Default is 0.8.
        implicit_num: float, optional
            Maximum number of loops for implicit calculation.
        implicit_threshold: float, optional
            Threshold value to finish the implicit calculation
        r0: float, optional
            Ratio of near-bed concentration to layer-averaged concentration
        C_init: float, optional
            Minimum value of sediment concentration.
        gamma: float, optional
            Coefficient for calculating velocity of flow front 
        suspension: boolean, optional
            turn on the function for entrainment/settling of suspension
        bedload: boolean, optional
            turn on the function for bedload
        water_entrainment: boolean, optional
            turn on the function for ambient water entrainment
        sed_entrainment_func: string, optional
            Choose the function to be used for sediment entrainment. Default
            is 'GP1991field', and other options are: 'GP1991exp', 'vanRijn1984'.
        scheme: string, optional
            Choose the numerical scheme for solution of non-advection terms.
            Currently "CCUP" and "CentralDifference" are available.
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
        self.p_w = p_w
        self.h_w = h_w
        self.nu = nu
        self.kappa2 = kappa2
        self.kappa4 = kappa4
        self.nu_a = nu_a
        self.r0 = r0
        self.lambda_p = lambda_p
        self.implicit_num = implicit_num
        self.implicit_threshold = implicit_threshold
        self.C_init = C_init
        self.gamma = gamma
        self.water_entrainment = water_entrainment
        self.suspension = suspension
        self.bedload = bedload
        self.sed_entrainment_func = sed_entrainment_func
        self.scheme = scheme

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
            self.Ch = self.C * self.h

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
        self.Ch += self.h_init * self.C_init

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

        try:
            self.dhdx = grid.add_zeros('flow_depth__horizontal_gradient',
                                       at='node')
            self.dhdy = grid.add_zeros('flow_depth__vertical_gradient',
                                       at='node')
            self.dChdx = grid.add_zeros(
                'flow_sediment_volume__horizontal_gradient', at='node')
            self.dChdy = grid.add_zeros(
                'flow_sediment_volume__vertical_gradient', at='node')

        except FieldError:
            self.dhdx = grid.at_node['flow_depth__horizontal_gradient']
            self.dhdy = grid.at_node['flow_depth__vertical_gradient']
            self.dChdx = grid.at_node[
                'flow_sediment_volume__horizontal_gradient']
            self.dChdy = grid.at_node[
                'flow_sediment_volume__vertical_gradient']

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
        # self.Ch = self.C * self.h
        self.u_node = np.zeros(grid.number_of_nodes)
        self.v_node = np.zeros(grid.number_of_nodes)
        self.h_link = np.zeros(grid.number_of_links)
        self.Ch_link = np.zeros(grid.number_of_links)

        # composite velocity
        self.U = np.zeros(grid.number_of_links)
        self.U_node = np.zeros(grid.number_of_nodes)

        # water and sediment entrainment rates
        self.ew_node = np.zeros(grid.number_of_nodes)
        self.ew_link = np.zeros(grid.number_of_links)
        self.es = np.zeros(grid.number_of_nodes)

        # diffusion coefficient of velocity
        self.nu_t = np.zeros(grid.number_of_links)

        # temporary values for calculation of non-advection terms
        self.G_h = np.zeros(grid.number_of_nodes)
        self.G_u = np.zeros(grid.number_of_links)
        self.G_v = np.zeros(grid.number_of_links)
        self.G_Ch = np.zeros(grid.number_of_nodes)
        self.G_eta = np.zeros(grid.number_of_nodes)
        self.G_eta_k1 = np.zeros(grid.number_of_nodes)
        self.G_eta_k2 = np.zeros(grid.number_of_nodes)
        self.G_eta_k3 = np.zeros(grid.number_of_nodes)
        self.G_eta_k4 = np.zeros(grid.number_of_nodes)

        # temporary values for flow depth
        self.h_temp = np.zeros(grid.number_of_nodes)
        self.dhdx_temp = np.zeros(grid.number_of_nodes)
        self.dhdy_temp = np.zeros(grid.number_of_nodes)
        self.h_prev = np.zeros(grid.number_of_nodes)
        self.h_link_temp = np.zeros(grid.number_of_links)

        # temporary values for horizontal velocity
        self.u_temp = np.zeros(grid.number_of_links)
        self.u_node_temp = np.zeros(grid.number_of_nodes)
        self.dudx_temp = np.zeros(grid.number_of_links)
        self.dudy_temp = np.zeros(grid.number_of_links)

        # temporary values for vertical velocity
        self.v_temp = np.zeros(grid.number_of_links)
        self.v_node_temp = np.zeros(grid.number_of_nodes)
        self.dvdx_temp = np.zeros(grid.number_of_links)
        self.dvdy_temp = np.zeros(grid.number_of_links)

        # temporary values for wave celerity
        self.Cs = np.zeros(grid.number_of_nodes)

        # volume of suspended sediment
        self.Ch_temp = np.zeros(grid.number_of_nodes)
        self.dChdx_temp = np.zeros(grid.number_of_nodes)
        self.dChdy_temp = np.zeros(grid.number_of_nodes)
        self.Ch_prev = np.zeros(grid.number_of_nodes)
        self.Ch_p = np.zeros(grid.number_of_nodes)
        self.Ch_link_temp = np.zeros(grid.number_of_links)

        # topographic elevation
        self.eta_temp = self.eta.copy()
        self.eta_p = np.zeros(grid.number_of_nodes)
        self.eta_init = self.eta.copy()

        # length of flow velocity vector
        self.U_temp = self.U.copy()
        self.U_node_temp = self.U_node.copy()

        # aritificial viscosity
        self.q_x = np.zeros(grid.number_of_nodes)
        self.q_y = np.zeros(grid.number_of_nodes)

        # arrays to record upcurrent and downcurrent nodes
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

        # arrays to refer adjacent links and nodes
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

        # ids to process boundary conditions
        self.bound_links = None
        self.edge_links = None
        self.fixed_grad_nodes = None
        self.fixed_grad_anchor_nodes = None
        self.fixed_grad_links = None
        self.fixed_grad_anchor_links = None
        self.fixed_value_nodes = None
        self.fixed_value_anchor_nodes = None
        self.fixed_value_links = None
        self.fixed_value_anchor_links = None
        self.fixed_value_edge_links = None

        # set variables to be used for processing wet/dry grids
        self.wet_nodes = None
        self.partial_wet_nodes = None
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
        self.wettest_nodes = None
        self.wet_pwet_nodes = None
        self.wet_pwet_links = None
        self.wet_pwet_horizontal_links = None
        self.wet_pwet_vertical_links = None
        self.dry_nodes = None
        self.dry_links = None
        self.horizontal_overspill_velocity = None
        self.vertical_overspill_velocity = None

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

        Ch_is_positive = self.Ch > 0
        self.Cs[Ch_is_positive] = np.sqrt(self.R * self.g *
                                          self.Ch[Ch_is_positive])
        self.Cs[~Ch_is_positive] = 0

        dt_local = self.alpha * self.grid.dx / np.amax(
            np.array([
                np.amax(np.abs(self.u_node) + self.Cs),
                np.amax(np.abs(self.v_node) + self.Cs), 1.0
            ]))

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

        # if self.first_step is True:
        #     self.initialize_gradients()

        # In case another component has added data to the fields, we just
        # reset our water depths, topographic elevations and water
        # velocity variables to the fields.
        self.h = self.grid['node']['flow__depth']
        self.eta = self.grid['node']['topographic__elevation']
        self.u = self.grid['link']['flow__horizontal_velocity']
        self.v = self.grid['link']['flow__vertical_velocity']
        self.u_node = self.grid['node']['flow__horizontal_velocity_at_node']
        self.v_node = self.grid['node']['flow__vertical_velocity_at_node']
        self.C = self.grid['node']['flow__sediment_concentration']
        self.Ch = self.C * self.h

        # Initialize boundary conditions and wet/dry grids
        find_boundary_links_nodes(self)
        find_wet_grids(self)
        map_values(self, self.h, self.dhdx, self.dhdy, self.u, self.dudx,
                   self.v, self.dvdy, self.Ch, self.dChdx, self.dChdy,
                   self.eta, self.h_link, self.u_node, self.v_node,
                   self.Ch_link, self.U, self.U_node)
        self.update_boundary_conditions(h=self.h,
                                        u=self.u,
                                        v=self.v,
                                        Ch=self.Ch,
                                        h_link=self.h_link,
                                        Ch_link=self.Ch_link,
                                        u_node=self.u_node,
                                        v_node=self.v_node,
                                        eta=self.eta)
        update_up_down_links_and_nodes(self)
        self.copy_values_to_temp()

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
            find_wet_grids(self)
            map_values(self, self.h, self.dhdx, self.dhdy, self.u, self.dudx,
                       self.v, self.dvdy, self.Ch, self.dChdx, self.dChdy,
                       self.eta, self.h_link, self.u_node, self.v_node,
                       self.Ch_link, self.U, self.U_node)

            # Process partial wet grids
            self._process_wet_dry_boundary()

            # calculate non-advection terms using implicit method
            self._nonadvection_phase()

            # Calculate diffusion term of momentum
            # self._diffusion_phase()

            # calculate advection terms using cip method
            self._advection_phase()

            # # re-check wet grids
            find_wet_grids(self)

            # # calculate deposition/erosion
            # self._deposition_phase()

            # # apply the shock dissipation scheme
            # if self.scheme == 'CentralDifference':
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
        cip_2d_advection(self.u,
                         self.dudx,
                         self.dudy,
                         self.u,
                         self.v,
                         self.wet_pwet_horizontal_links,
                         self.horizontal_up_links,
                         self.vertical_up_links,
                         self.grid.dx,
                         self.dt_local,
                         out_f=self.u_temp,
                         out_dfdx=self.dudx_temp,
                         out_dfdy=self.dudy_temp)

        cip_2d_advection(self.v,
                         self.dvdx,
                         self.dvdy,
                         self.u,
                         self.v,
                         self.wet_pwet_vertical_links,
                         self.horizontal_up_links,
                         self.vertical_up_links,
                         self.grid.dx,
                         self.dt_local,
                         out_f=self.v_temp,
                         out_dfdx=self.dvdx_temp,
                         out_dfdy=self.dvdy_temp)

        if self.scheme != 'CentralDifference':
            cip_2d_advection(self.h,
                             self.dhdx,
                             self.dhdy,
                             self.u_node,
                             self.v_node,
                             self.wet_pwet_nodes,
                             self.horizontal_up_nodes,
                             self.vertical_up_nodes,
                             self.grid.dx,
                             self.dt_local,
                             out_f=self.h_temp,
                             out_dfdx=self.dhdx_temp,
                             out_dfdy=self.dhdy_temp)

            cip_2d_advection(self.Ch,
                             self.dChdx,
                             self.dChdy,
                             self.u_node,
                             self.v_node,
                             self.wet_pwet_nodes,
                             self.horizontal_up_nodes,
                             self.vertical_up_nodes,
                             self.grid.dx,
                             self.dt_local,
                             out_f=self.Ch_temp,
                             out_dfdx=self.dChdx_temp,
                             out_dfdy=self.dChdy_temp)

        adjust_negative_values(self.h_temp,
                               self.Ch_temp,
                               self.wet_pwet_nodes,
                               self.node_east,
                               self.node_west,
                               self.node_north,
                               self.node_south,
                               out_h=self.h_temp,
                               out_Ch=self.Ch_temp)

        # update gradient terms
        self.update_gradients2()

        # update values after calculating advection terms
        # map node values to links, and link values to nodes.
        self.update_values()
        map_values(self, self.h, self.dhdx, self.dhdy, self.u, self.dudx,
                   self.v, self.dvdy, self.Ch, self.dChdx, self.dChdy,
                   self.eta, self.h_link, self.u_node, self.v_node,
                   self.Ch_link, self.U, self.U_node)
        update_up_down_links_and_nodes(self)

    def _nonadvection_phase(self):
        """Calculate non-advection phase of the model
           Pressure terms for velocities and mass conservation equations
           are solved implicitly by CCUP method
        """

        if self.scheme == 'CCUP':
            self._CCUP()
        elif self.scheme == 'CentralDifference':
            self._central_difference()

        # update gradient terms
        self.update_gradients()

        # update values
        self.update_values()
        map_values(self, self.h, self.dhdx, self.dhdy, self.u, self.dudx,
                   self.v, self.dvdy, self.Ch, self.dChdx, self.dChdy,
                   self.eta, self.h_link, self.u_node, self.v_node,
                   self.Ch_link, self.U, self.U_node)
        update_up_down_links_and_nodes(self)

        # # re-check wet and partial wet grids
        # find_wet_grids(self)

    def _central_difference(self):
        """solve non-advection terms by central difference scheme
        """
        north_link = self.north_link_at_node[self.wet_pwet_nodes]
        south_link = self.south_link_at_node[self.wet_pwet_nodes]
        east_link = self.east_link_at_node[self.wet_pwet_nodes]
        west_link = self.west_link_at_node[self.wet_pwet_nodes]
        dx = self.grid.dx
        dy = self.grid.dx

        # copy and map values
        self.copy_values_to_temp()
        self.h_prev[self.wet_pwet_nodes] = self.h_temp[self.wet_pwet_nodes]
        map_nodes_to_links(self, self.h_temp, self.dhdx_temp, self.dhdy_temp,
                           self.Ch_temp, self.dChdx_temp, self.dChdy_temp,
                           self.eta_temp, self.h_link_temp, self.Ch_link_temp)
        err = 10.0
        count = 0

        while err > 1.0 * 10**-10:
            # calculate gravity force
            self.calc_G_u(self.h_temp,
                          self.h_link_temp,
                          self.u_temp,
                          self.v_temp,
                          self.Ch_temp,
                          self.Ch_link_temp,
                          self.eta_temp,
                          self.U_temp,
                          self.wet_horizontal_links,
                          scheme='CentralDifference')
            self.calc_G_v(self.h_temp,
                          self.h_link_temp,
                          self.u_temp,
                          self.v_temp,
                          self.Ch_temp,
                          self.Ch_link_temp,
                          self.eta_temp,
                          self.U_temp,
                          self.wet_vertical_links,
                          scheme='CentralDifference')
            self.u_temp[self.wet_horizontal_links] = self.u[
                self.wet_horizontal_links] + self.G_u[
                    self.wet_horizontal_links] * self.dt_local
            self.v_temp[self.wet_vertical_links] = self.v[
                self.wet_vertical_links] + self.G_v[
                    self.wet_vertical_links] * self.dt_local

            # water entrainment
            if self.water_entrainment is True:
                self.ew_link[self.wet_horizontal_links] = get_ew(
                    self.U_temp[self.wet_horizontal_links],
                    self.Ch_link_temp[self.wet_horizontal_links], self.R,
                    self.g)
                self.ew_link[self.wet_vertical_links] = get_ew(
                    self.U_temp[self.wet_vertical_links],
                    self.Ch_link_temp[self.wet_vertical_links], self.R, self.g)
            else:
                self.ew_link[self.wet_horizontal_links] = 0
                self.ew_link[self.wet_vertical_links] = 0
                self.ew_node[self.wet_nodes] = 0

            # calculate friction terms using semi-implicit scheme
            self.u_temp[self.wet_horizontal_links] *= (
                1 / (1 + (self.Cf + self.ew_link[self.wet_horizontal_links]) *
                     self.U_temp[self.wet_horizontal_links] * self.dt_local /
                     self.h_link_temp[self.wet_horizontal_links]))
            self.v_temp[self.wet_vertical_links] *= (
                1 / (1 + (self.Cf + self.ew_link[self.wet_vertical_links]) *
                     self.U_temp[self.wet_vertical_links] * self.dt_local /
                     self.h_link_temp[self.wet_vertical_links]))

            # map values
            map_links_to_nodes(self, self.u_temp, self.dudx_temp, self.v_temp,
                               self.dvdy_temp, self.u_node_temp,
                               self.v_node_temp, self.U_temp, self.U_node_temp)

            # calculate mass conservation of h and Ch
            self.h_temp[self.wet_pwet_nodes] = self.h[self.wet_pwet_nodes] - (
                (self.h_link_temp[east_link] * self.u_temp[east_link] -
                 self.h_link_temp[west_link] * self.u_temp[west_link]) / dx +
                (self.h_link_temp[north_link] * self.v_temp[north_link] -
                 self.h_link_temp[south_link] * self.v_temp[south_link]) /
                dy) * self.dt_local
            self.Ch_temp[self.wet_pwet_nodes] = self.Ch[self.wet_pwet_nodes] - (
                (self.Ch_link_temp[east_link] * self.u_temp[east_link] -
                 self.Ch_link_temp[west_link] * self.u_temp[west_link]) / dx +
                (self.Ch_link_temp[north_link] * self.v_temp[north_link] -
                 self.Ch_link_temp[south_link] * self.v_temp[south_link]) /
                dy) * self.dt_local

            # calculate flow expansion by water entrainment
            if self.water_entrainment == True:
                self.ew_node[self.wet_nodes] = get_ew(
                    self.U_node_temp[self.wet_nodes], self.Ch[self.wet_nodes],
                    self.R, self.g)
                self.h_temp[self.wet_nodes] += self.ew_node[
                    self.wet_nodes] * self.U_node[
                        self.wet_nodes] * self.dt_local

            # map values
            map_nodes_to_links(self, self.h_temp, self.dhdx_temp,
                               self.dhdy_temp, self.Ch_temp, self.dChdx_temp,
                               self.dChdy_temp, self.eta_temp,
                               self.h_link_temp, self.Ch_link_temp)

            # check convergence of calculation
            err = np.linalg.norm(self.h_temp[self.wet_pwet_nodes] -
                                 self.h_prev[self.wet_pwet_nodes]) / (
                                     self.wet_pwet_nodes.shape[0] +
                                     1.0 * 10**-10)
            self.h_prev[self.wet_pwet_nodes] = self.h_temp[self.wet_pwet_nodes]
            count += 1
            if count == self.implicit_num:
                print('Implicit calculation did not converge')
                break

    def _CCUP(self):
        """solve non-advection terms by CCUP method
        """

        self.copy_values_to_temp()
        map_nodes_to_links(self, self.h_temp, self.dhdx_temp, self.dhdy_temp,
                           self.Ch_temp, self.dChdx_temp, self.dChdy_temp,
                           self.eta_temp, self.h_link_temp, self.Ch_link_temp)

        # copy grid ids
        wet_pwet_nodes = self.wet_pwet_nodes
        wet_pwet_h_links = self.wet_pwet_horizontal_links
        wet_pwet_v_links = self.wet_pwet_vertical_links
        # east_link_at_node = self.east_link_at_node[wet_pwet_nodes]
        # west_link_at_node = self.west_link_at_node[wet_pwet_nodes]
        # north_link_at_node = self.north_link_at_node[wet_pwet_nodes]
        # south_link_at_node = self.south_link_at_node[wet_pwet_nodes]
        # east_node_at_link = self.east_node_at_horizontal_link[wet_pwet_h_links]
        # west_node_at_link = self.west_node_at_horizontal_link[wet_pwet_h_links]
        # north_node_at_link = self.north_node_at_vertical_link[wet_pwet_v_links]
        # south_node_at_link = self.south_node_at_vertical_link[wet_pwet_v_links]
        east_link_at_h_link = self.link_east[wet_pwet_h_links]
        west_link_at_h_link = self.link_west[wet_pwet_h_links]
        north_link_at_h_link = self.link_north[wet_pwet_h_links]
        south_link_at_h_link = self.link_south[wet_pwet_h_links]
        east_link_at_v_link = self.link_east[wet_pwet_v_links]
        west_link_at_v_link = self.link_west[wet_pwet_v_links]
        north_link_at_v_link = self.link_north[wet_pwet_v_links]
        south_link_at_v_link = self.link_south[wet_pwet_v_links]

        # solve the pressure term by SOR method
        wet_nodes_num = self.wet_nodes.shape[0]
        err = 10.0
        count = 0
        alpha = 1.2
        dx = self.grid.dx
        dy = self.grid.dy
        dt = self.dt_local
        Rg = self.R * self.g
        p = self.Ch_temp * self.h_temp
        p_new = p.copy()
        p_north = p[self.node_north[self.wet_nodes]]
        p_south = p[self.node_south[self.wet_nodes]]
        p_east = p[self.node_east[self.wet_nodes]]
        p_west = p[self.node_west[self.wet_nodes]]
        u_east = self.u_temp[self.east_link_at_node[self.wet_nodes]]
        u_west = self.u_temp[self.west_link_at_node[self.wet_nodes]]
        v_north = self.v_temp[self.north_link_at_node[self.wet_nodes]]
        v_south = self.v_temp[self.south_link_at_node[self.wet_nodes]]
        h_link_north = self.h_link_temp[self.north_link_at_node[
            self.wet_nodes]]
        h_link_south = self.h_link_temp[self.south_link_at_node[
            self.wet_nodes]]
        h_link_east = self.h_link_temp[self.east_link_at_node[self.wet_nodes]]
        h_link_west = self.h_link_temp[self.west_link_at_node[self.wet_nodes]]
        Ch_link_north = self.Ch_link_temp[self.north_link_at_node[
            self.wet_nodes]]
        Ch_link_south = self.Ch_link_temp[self.south_link_at_node[
            self.wet_nodes]]
        Ch_link_east = self.Ch_link_temp[self.east_link_at_node[
            self.wet_nodes]]
        Ch_link_west = self.Ch_link_temp[self.west_link_at_node[
            self.wet_nodes]]
        eta = self.eta_temp[self.wet_nodes]
        eta_north = self.eta_temp[self.node_north[self.wet_nodes]]
        eta_south = self.eta_temp[self.node_south[self.wet_nodes]]
        eta_east = self.eta_temp[self.node_east[self.wet_nodes]]
        eta_west = self.eta_temp[self.node_west[self.wet_nodes]]
        C_link_north = Ch_link_north / h_link_north
        C_link_south = Ch_link_south / h_link_south
        C_link_east = Ch_link_east / h_link_east
        C_link_west = Ch_link_west / h_link_west
        a = np.empty(self.wet_nodes.shape)
        b = np.empty(self.wet_nodes.shape)
        c = np.empty(self.wet_nodes.shape)
        d = np.empty(self.wet_nodes.shape)
        e = np.empty(self.wet_nodes.shape)
        g = np.empty(self.wet_nodes.shape)
        w = np.empty(self.wet_nodes.shape)
        dx2 = dx * dx
        dy2 = dy * dy
        dt2 = dt * dt

        a[:] = -Rg * (
            1 / (2.0 * h_link_east) + 1 / (2.0 * h_link_west)) / dx2 - Rg * (
                1 / (2.0 * h_link_north) + 1 /
                (2.0 * h_link_south)) / dy2 - 1 / (2 * p[self.wet_nodes] * dt2)
        b[:] = Rg / (dx2 * 2.0 * h_link_east)
        c[:] = Rg / (dx2 * 2.0 * h_link_west)
        d[:] = Rg / (dx2 * 2.0 * h_link_north)
        e[:] = Rg / (dx2 * 2.0 * h_link_south)
        g[:] = -1 / (2 * dt2) + ((u_east - u_west) / dx +
                                 (v_north - v_south) / dy) / dt - Rg * (
                                     (C_link_east *
                                      (eta_east - eta) - C_link_west *
                                      (eta - eta_west)) / dx2 +
                                     (C_link_north *
                                      (eta_north - eta) - C_link_south *
                                      (eta - eta_south)) / dy2)

        while err > self.implicit_threshold:
            w[:] = (g - b * p_east - c * p_west - d * p_north -
                    e * p_south) / a
            err = np.linalg.norm(w - p_new[self.wet_nodes]) / (wet_nodes_num +
                                                               1.0 * 10**-10)
            p_new[self.
                  wet_nodes] = p_new[self.wet_nodes] * (1 - alpha) + alpha * w
            self.update_boundary_conditions(p=p_new)

            p_north[:] = p_new[self.node_north[self.wet_nodes]]
            p_south[:] = p_new[self.node_south[self.wet_nodes]]
            p_east[:] = p_new[self.node_east[self.wet_nodes]]
            p_west[:] = p_new[self.node_west[self.wet_nodes]]

            count += 1

            if count == self.implicit_num:
                print('Implicit calculation did not converge')
                break

        # calculate u, v based on pressure
        self.u_temp[self.wet_horizontal_links] -= Rg / (
            2 * self.h_link[self.wet_horizontal_links]
        ) * (
            p_new[self.east_node_at_horizontal_link[self.wet_horizontal_links]]
            -
            p_new[self.west_node_at_horizontal_link[self.wet_horizontal_links]]
        ) / dx * dt
        self.v_temp[self.wet_vertical_links] -= Rg / (
            2 * self.h_link[self.wet_vertical_links]
        ) * (p_new[self.north_node_at_vertical_link[self.wet_vertical_links]] -
             p_new[self.south_node_at_vertical_link[self.wet_vertical_links]]
             ) / dy * dt

        # map values
        map_values(self, self.h_temp, self.dhdx_temp, self.dhdy_temp,
                   self.u_temp, self.dudx_temp, self.v_temp, self.dvdy_temp,
                   self.Ch_temp, self.dChdx_temp, self.dChdy_temp,
                   self.eta_temp, self.h_link_temp, self.u_node_temp,
                   self.v_node_temp, self.Ch_link_temp, self.U_temp,
                   self.U_node_temp)

        # calculate gravity force
        self.calc_G_u(self.h, self.h_link, self.u, self.v, self.Ch,
                      self.Ch_link, self.eta, self.U,
                      self.wet_horizontal_links)
        self.calc_G_v(self.h, self.h_link, self.u, self.v, self.Ch,
                      self.Ch_link, self.eta, self.U, self.wet_vertical_links)
        self.u_temp[self.wet_horizontal_links] += self.G_u[
            self.wet_horizontal_links] * self.dt_local
        self.v_temp[self.wet_vertical_links] += self.G_v[
            self.wet_vertical_links] * self.dt_local

        # map values
        map_links_to_nodes(self, self.u_temp, self.dudx_temp, self.v_temp,
                           self.dvdy_temp, self.u_node_temp, self.v_node_temp,
                           self.U_temp, self.U_node_temp)

        # water entrainment
        if self.water_entrainment is True:
            self.ew_link[self.wet_horizontal_links] = get_ew(
                self.U[self.wet_horizontal_links],
                self.Ch_link[self.wet_horizontal_links], self.R, self.g)
            self.ew_link[self.wet_vertical_links] = get_ew(
                self.U[self.wet_vertical_links],
                self.Ch_link[self.wet_vertical_links], self.R, self.g)
            self.ew_node[self.wet_nodes] = get_ew(self.U_node[self.wet_nodes],
                                                  self.Ch[self.wet_nodes],
                                                  self.R, self.g)
        else:
            self.ew_link[self.wet_horizontal_links] = 0
            self.ew_link[self.wet_vertical_links] = 0
            self.ew_node[self.wet_nodes] = 0

        # calculate friction terms using semi-implicit scheme
        self.u_temp[self.wet_horizontal_links] *= 1 / (
            1 + (self.Cf + self.ew_link[self.wet_horizontal_links]) *
            self.U[self.wet_horizontal_links] * self.dt_local /
            self.h_link[self.wet_horizontal_links])
        self.v_temp[self.wet_vertical_links] *= 1 / (
            1 + (self.Cf + self.ew_link[self.wet_vertical_links]) *
            self.U[self.wet_vertical_links] * self.dt_local /
            self.h_link[self.wet_vertical_links])

        # diffusion of momentum
        self.calc_nu_t(self.u, self.v, self.h_link, out=self.nu_t)
        self.u_temp[wet_pwet_h_links] += self.nu_t[wet_pwet_h_links] * dt * (
               (self.u[east_link_at_h_link]
                - 2 * self.u[wet_pwet_h_links]
                + self.u[west_link_at_h_link])
                + (self.u[north_link_at_h_link]
                   - 2 * self.u[wet_pwet_h_links]
                   + self.u[south_link_at_h_link]))\
               / dx**2
        self.v_temp[wet_pwet_v_links] += self.nu_t[wet_pwet_v_links] * dt * (
               (self.v[east_link_at_v_link]
                - 2 * self.v[wet_pwet_v_links]
                + self.v[west_link_at_v_link])
                + (self.v[north_link_at_v_link]
                   - 2 * self.v[wet_pwet_v_links]
                   + self.v[south_link_at_v_link]))\
                / dx**2

        # calculate flow expansion by water entrainment
        self.h_temp[self.wet_nodes] += self.ew_node[
            self.wet_nodes] * self.U_node[self.wet_nodes] * self.dt_local

        # calculate sediment deposition
        if self.suspension is True:
            self.calc_deposition(self.h,
                                 self.Ch,
                                 self.u_node,
                                 self.v_node,
                                 self.eta,
                                 self.U_node,
                                 out_Ch=self.Ch_temp,
                                 out_eta=self.eta_temp)

        # apply artificial viscosity
        self._artificial_viscosity(self.h_temp, self.h_link_temp, self.u_temp,
                                   self.v_temp, self.Ch_temp,
                                   self.Ch_link_temp)
        self.update_boundary_conditions(h=self.h_temp,
                                        u=self.u_temp,
                                        v=self.v_temp,
                                        Ch=self.Ch_temp,
                                        h_link=self.h_link_temp,
                                        Ch_link=self.Ch_link_temp,
                                        u_node=self.u_node_temp,
                                        v_node=self.v_node_temp,
                                        eta=self.eta_temp)

        # calculate divergence of velocity
        div = (self.u_temp[self.east_link_at_node[self.wet_pwet_nodes]] - self.
               u_temp[self.west_link_at_node[self.wet_pwet_nodes]]) / (dx) + (
                   self.v_temp[self.north_link_at_node[self.wet_pwet_nodes]] -
                   self.v_temp[self.south_link_at_node[self.wet_pwet_nodes]]
               ) / (dy)

        # mass conservation
        self.h_temp[self.wet_pwet_nodes] = self.h_temp[self.wet_pwet_nodes] / (
            1 + div * dt)
        self.Ch_temp[self.wet_pwet_nodes] = self.Ch_temp[
            self.wet_pwet_nodes] / (1 + div * dt)

        adjust_negative_values(self.h_temp,
                               self.Ch_temp,
                               self.wet_pwet_nodes,
                               self.node_east,
                               self.node_west,
                               self.node_north,
                               self.node_south,
                               out_h=self.h_temp,
                               out_Ch=self.Ch_temp)

    def _artificial_viscosity(self, h, h_link, u, v, Ch, Ch_link):
        """Apply artificial viscosity to flow velocity 
           on the basis of the scheme proposed by Ogata and Yabe (1999)
        
           Parameters
           -------------------
           h : ndarray, float
               flow depth at nodes

           h_link : ndarray, float
               flow depth at links

           u : ndarray, float
               horizontal flow velocity at links

           v : ndarray, float
               vertical flow velocity at links

           Ch : ndarray, float
               Volume of suspended sediment at nodes

           Ch_link : ndarray, float
               Volume of suspended sediment at links

        """
        # copy values from self
        dx = self.grid.dx
        dy = self.grid.dx
        dt = self.dt_local
        wet_nodes = self.wet_pwet_nodes
        wet_horiz = self.wet_pwet_horizontal_links
        wet_vert = self.wet_pwet_vertical_links
        east_link = self.east_link_at_node[wet_nodes]
        west_link = self.west_link_at_node[wet_nodes]
        north_link = self.north_link_at_node[wet_nodes]
        south_link = self.south_link_at_node[wet_nodes]
        east_node = self.east_node_at_horizontal_link[wet_horiz]
        west_node = self.west_node_at_horizontal_link[wet_horiz]
        north_node = self.north_node_at_vertical_link[wet_vert]
        south_node = self.south_node_at_vertical_link[wet_vert]
        nu_a = self.nu_a  # artificial viscosity coefficient
        Rg = self.R * self.g
        Cs = self.Cs
        # p = Ch[wet_nodes] * h[wet_nodes]

        # basic parameters for artificial viscosity
        rho = (2.0 * h[wet_nodes]) / Rg
        rho_link_horiz = 2.0 * h_link[wet_horiz] / Rg
        rho_link_vert = 2.0 * h_link[wet_vert] / Rg
        Ch_is_positive = Ch[wet_nodes] > 0
        Cs[wet_nodes[Ch_is_positive]] = np.sqrt(Rg *
                                                Ch[wet_nodes[Ch_is_positive]])
        Cs[wet_nodes[~Ch_is_positive]] = 0

        # calculate aritificial viscosity only at grids where flow is
        # compressive (div < 0)
        div_horiz = (u[east_link] - u[west_link]) / dx
        div_vert = (v[north_link] - v[south_link]) / dy
        div = div_horiz + div_vert

        # horizontal aritificial visocosity
        # compress_horiz = div_horiz < 0
        # self.q_x[wet_nodes[~compress_horiz]] = 0
        # self.q_x[
        #     wet_nodes[compress_horiz]] = rho[compress_horiz] * nu_a * dx * (
        #         -Cs[compress_horiz] * div_horiz[compress_horiz] +
        #         (1.0 + gamma) / 2.0 * div_horiz[compress_horiz]**2 * dx)

        # vertical aritificial visocosity
        # compress_vert = div_vert < 0
        # self.q_y[wet_nodes[~compress_vert]] = 0
        # self.q_y[wet_nodes[compress_vert]] = rho[compress_vert] * nu_a * dy * (
        #     -Cs[compress_vert] * div_vert[compress_vert] +
        #     (1.0 + gamma) / 2.0 * div_vert[compress_vert]**2 * dy)

        # no directional dependent expression of artificial viscosity
        compress = div < 0
        self.q_x[wet_nodes[~compress]] = 0
        self.q_x[wet_nodes[compress]] = rho[compress] * nu_a * dx * (
            -Cs[wet_nodes][compress] * div[compress] +
            1.5 * div[compress]**2 * dx)

        # modify flow velocity based on artificial viscosity
        u[wet_horiz] -= (self.q_x[east_node] -
                         self.q_x[west_node]) / dx / rho_link_horiz * dt
        # v[wet_vert] -= (self.q_y[north_node] -
        #                 self.q_y[south_node]) / dy / rho_link_vert * dt
        v[wet_vert] -= (self.q_x[north_node] -
                        self.q_x[south_node]) / dy / rho_link_vert * dt

        # # # modify pressure
        # p -= (gamma - 1.0) * self.q[wet_nodes] * div * dt

        # # # modify h and Ch
        # C = Ch[wet_nodes] / h[wet_nodes]
        # h[wet_nodes] = np.sqrt(p / C)
        # Ch[wet_nodes] = np.sqrt(p * C)
        # # # h[wet_nodes] -= h[wet_nodes] * self.q[wet_nodes] * div
        # # # Ch[wet_nodes] -= Ch[wet_nodes] * self.q[wet_nodes] * div

    def _deposition_phase(self):
        """Calculate depositional and erosional processes
        """
        if self.suspension is True:
            self.calc_deposition(self.h,
                                 self.Ch,
                                 self.u_node,
                                 self.v_node,
                                 self.eta,
                                 self.U_node,
                                 out_Ch=self.Ch_temp,
                                 out_eta=self.eta_temp)
        self.update_values()
        map_nodes_to_links(self, self.h, self.dhdx, self.dhdy, self.Ch,
                           self.dChdx, self.dChdy, self.eta, self.h_link,
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

        # update gradient terms
        self.update_gradients()

        # update values
        self.update_values()
        map_links_to_nodes(self, self.u, self.dudx, self.v, self.dvdy,
                           self.u_node, self.v_node, self.U, self.U_node)

    def _process_wet_dry_boundary(self):
        """Calculate processes at wet and dry boundary
        """
        process_partial_wet_grids(self,
                                  self.h,
                                  self.u,
                                  self.v,
                                  self.Ch,
                                  h_out=self.h_temp,
                                  u_out=self.u_temp,
                                  v_out=self.v_temp,
                                  Ch_out=self.Ch_temp)
        update_up_down_links_and_nodes(self)

        # update gradient terms
        self.update_gradients()

        # update values
        self.update_values()
        map_values(self, self.h, self.dhdx, self.dhdy, self.u, self.dudx,
                   self.v, self.dvdy, self.Ch, self.dChdx, self.dChdy,
                   self.eta, self.h_link, self.u_node, self.v_node,
                   self.Ch_link, self.U, self.U_node)

    def _shock_dissipation_phase(self):
        """Calculate shock dissipation phase of the model
        """
        shock_dissipation(self.Ch,
                          self.R * self.g * self.Ch * self.h,
                          self.core_nodes,
                          self.node_north,
                          self.node_south,
                          self.node_east,
                          self.node_west,
                          self.dt_local,
                          self.kappa2,
                          self.kappa4,
                          out=self.Ch_temp)

        shock_dissipation(self.h,
                          self.R * self.g * self.Ch * self.h,
                          self.core_nodes,
                          self.node_north,
                          self.node_south,
                          self.node_east,
                          self.node_west,
                          self.dt_local,
                          self.kappa2,
                          self.kappa4,
                          out=self.h_temp)

        # shock_dissipation(self.u,
        #                   self.R * self.g * self.Ch_link * self.h_link,
        #                   self.horizontal_active_links,
        #                   self.link_north,
        #                   self.link_south,
        #                   self.link_east,
        #                   self.link_west,
        #                   self.dt_local,
        #                   self.kappa2,
        #                   self.kappa4,
        #                   out=self.u_temp)

        # shock_dissipation(self.v,
        #                   self.R * self.g * self.Ch_link * self.h_link,
        #                   self.vertical_active_links,
        #                   self.link_north,
        #                   self.link_south,
        #                   self.link_east,
        #                   self.link_west,
        #                   self.dt_local,
        #                   self.kappa2,
        #                   self.kappa4,
        #                   out=self.v_temp)

        # update gradient terms
        self.update_gradients()

        # Reset our field values with the newest flow depth and
        # discharge.
        self.update_values()
        map_values(self, self.h, self.dhdx, self.dhdy, self.u, self.dudx,
                   self.v, self.dvdy, self.Ch, self.dChdx, self.dChdy,
                   self.eta, self.h_link, self.u_node, self.v_node,
                   self.Ch_link, self.U, self.U_node)
        update_up_down_links_and_nodes(self)

    def initialize_gradients(self):
        """Initialize gradient terms for the first step
        """
        dx = self.grid.dx
        dy = self.grid.dy

        core_nodes = self.core_nodes
        node_east = self.node_east[self.core_nodes]
        node_west = self.node_west[self.core_nodes]
        node_north = self.node_north[self.core_nodes]
        node_south = self.node_south[self.core_nodes]

        horiz_active_links = self.horizontal_active_links
        vert_active_links = self.vertical_active_links
        horiz_link_east = self.link_east[horiz_active_links]
        horiz_link_west = self.link_west[horiz_active_links]
        horiz_link_north = self.link_north[horiz_active_links]
        horiz_link_south = self.link_south[horiz_active_links]
        vert_link_east = self.link_east[vert_active_links]
        vert_link_west = self.link_west[vert_active_links]
        vert_link_north = self.link_north[vert_active_links]
        vert_link_south = self.link_south[vert_active_links]

        self.dhdx[core_nodes] = (self.h[node_east] - self.h[node_west]) / (2 *
                                                                           dx)
        self.dhdy[core_nodes] = (self.h[node_north] -
                                 self.h[node_south]) / (2 * dy)
        self.dChdx[core_nodes] = (self.Ch[node_east] -
                                  self.Ch[node_west]) / (2 * dx)
        self.dChdy[core_nodes] = (self.Ch[node_north] -
                                  self.Ch[node_south]) / (2 * dy)
        self.dudx[horiz_active_links] = (self.u[horiz_link_east] -
                                         self.u[horiz_link_west]) / (2 * dx)
        self.dudy[horiz_active_links] = (self.u[horiz_link_north] -
                                         self.u[horiz_link_south]) / (2 * dy)
        self.dvdx[vert_active_links] = (self.v[vert_link_east] -
                                        self.v[vert_link_west]) / (2 * dx)
        self.dvdy[vert_active_links] = (self.v[vert_link_north] -
                                        self.v[vert_link_south]) / (2 * dy)

    def update_gradients(self):
        """update gradient terms when main variables were changed
        """
        update_gradient(self.u,
                        self.u_temp,
                        self.dudx_temp,
                        self.dudy_temp,
                        self.wet_pwet_horizontal_links,
                        self.link_north[self.wet_pwet_horizontal_links],
                        self.link_south[self.wet_pwet_horizontal_links],
                        self.link_east[self.wet_pwet_horizontal_links],
                        self.link_west[self.wet_pwet_horizontal_links],
                        self.grid.dx,
                        self.dt_local,
                        out_dfdx=self.dudx_temp,
                        out_dfdy=self.dudy_temp)
        update_gradient(self.v,
                        self.v_temp,
                        self.dvdx_temp,
                        self.dvdy_temp,
                        self.wet_pwet_vertical_links,
                        self.link_north[self.wet_pwet_vertical_links],
                        self.link_south[self.wet_pwet_vertical_links],
                        self.link_east[self.wet_pwet_vertical_links],
                        self.link_west[self.wet_pwet_vertical_links],
                        self.grid.dx,
                        self.dt_local,
                        out_dfdx=self.dvdx_temp,
                        out_dfdy=self.dvdy_temp)
        update_gradient(self.h,
                        self.h_temp,
                        self.dhdx_temp,
                        self.dhdy_temp,
                        self.wet_pwet_nodes,
                        self.node_north[self.wet_pwet_nodes],
                        self.node_south[self.wet_pwet_nodes],
                        self.node_east[self.wet_pwet_nodes],
                        self.node_west[self.wet_pwet_nodes],
                        self.grid.dx,
                        self.dt_local,
                        out_dfdx=self.dhdx_temp,
                        out_dfdy=self.dhdy_temp)
        update_gradient(self.Ch,
                        self.Ch_temp,
                        self.dChdx_temp,
                        self.dChdy_temp,
                        self.wet_pwet_nodes,
                        self.node_north[self.wet_pwet_nodes],
                        self.node_south[self.wet_pwet_nodes],
                        self.node_east[self.wet_pwet_nodes],
                        self.node_west[self.wet_pwet_nodes],
                        self.grid.dx,
                        self.dt_local,
                        out_dfdx=self.dChdx_temp,
                        out_dfdy=self.dChdy_temp)

    def update_gradients2(self):
        """update gradient terms for df/dx du/dy
        """
        update_gradient2(self.u,
                         self.dudx_temp,
                         self.dudy_temp,
                         self.u,
                         self.v,
                         self.wet_pwet_horizontal_links,
                         self.link_north[self.wet_pwet_horizontal_links],
                         self.link_south[self.wet_pwet_horizontal_links],
                         self.link_east[self.wet_pwet_horizontal_links],
                         self.link_west[self.wet_pwet_horizontal_links],
                         self.grid.dx,
                         self.dt_local,
                         out_dfdx=self.dudx_temp,
                         out_dfdy=self.dudy_temp)
        update_gradient2(self.v,
                         self.dvdx_temp,
                         self.dvdy_temp,
                         self.u,
                         self.v,
                         self.wet_pwet_vertical_links,
                         self.link_north[self.wet_pwet_vertical_links],
                         self.link_south[self.wet_pwet_vertical_links],
                         self.link_east[self.wet_pwet_vertical_links],
                         self.link_west[self.wet_pwet_vertical_links],
                         self.grid.dx,
                         self.dt_local,
                         out_dfdx=self.dvdx_temp,
                         out_dfdy=self.dvdy_temp)
        update_gradient2(self.h,
                         self.dhdx_temp,
                         self.dhdy_temp,
                         self.u_node,
                         self.v_node,
                         self.wet_pwet_nodes,
                         self.node_north[self.wet_pwet_nodes],
                         self.node_south[self.wet_pwet_nodes],
                         self.node_east[self.wet_pwet_nodes],
                         self.node_west[self.wet_pwet_nodes],
                         self.grid.dx,
                         self.dt_local,
                         out_dfdx=self.dhdx_temp,
                         out_dfdy=self.dhdy_temp)
        update_gradient2(self.Ch,
                         self.dChdx_temp,
                         self.dChdy_temp,
                         self.u_node,
                         self.v_node,
                         self.wet_pwet_nodes,
                         self.node_north[self.wet_pwet_nodes],
                         self.node_south[self.wet_pwet_nodes],
                         self.node_east[self.wet_pwet_nodes],
                         self.node_west[self.wet_pwet_nodes],
                         self.grid.dx,
                         self.dt_local,
                         out_dfdx=self.dChdx_temp,
                         out_dfdy=self.dChdy_temp)
        self.update_values()

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
        # nodes = np.where(self.h > self.h_w * 0.1)[0]
        nodes = self.wet_nodes

        # k1 = self.G_eta_k1
        # k2 = self.G_eta_k2
        # k3 = self.G_eta_k3
        # k4 = self.G_eta_k4

        # self.calc_G_eta(h, u_node, v_node, Ch, U_node, nodes, out_geta=k1)

        # self.calc_G_eta(h,
        #                 u_node,
        #                 v_node,
        #                 Ch + 0.5 * self.dt_local * (-k1),
        #                 U_node,
        #                 nodes,
        #                 out_geta=k2)

        # self.calc_G_eta(h,
        #                 u_node,
        #                 v_node,
        #                 Ch + 0.5 * self.dt_local * (-k2),
        #                 U_node,
        #                 nodes,
        #                 out_geta=k3)

        # self.calc_G_eta(h,
        #                 u_node,
        #                 v_node,
        #                 Ch + self.dt_local * (-k3),
        #                 U_node,
        #                 nodes,
        #                 out_geta=k4)

        # self.G_eta = 1. / 6. * (k1 + 2. * k2 + 2. * k3 + k4)

        # out_Ch[nodes] = Ch[nodes] \
        #     + self.dt_local * (
        #     - self.G_eta[nodes])
        # out_eta[nodes] = eta[nodes] \
        #     + self.dt_local * (
        #     self.G_eta[nodes] / (
        #         1 - self.lambda_p))

        # # dirty trick to avoid extremely high sediment concentration
        # C_max = 0.1
        # C_abnormal = out_Ch[nodes] > C_max * h[nodes]
        # out_Ch[nodes[C_abnormal]] = C_max * h[nodes][C_abnormal]
        # self.G_eta[nodes[C_abnormal]] = (Ch[nodes][C_abnormal] - C_max *
        #                                  h[nodes][C_abnormal]) / self.dt_local
        # out_eta[nodes[C_abnormal]] = eta[nodes[C_abnormal]] + self.dt_local * (
        #     self.G_eta[nodes[C_abnormal]] / (1 - self.lambda_p))

        # # test of ode solver
        # Ch_new = odeint(test_ode_grad,
        #                 Ch[nodes],
        #                 np.array([0, self.dt_local]),
        #                 args=(self, h, u_node, v_node, U_node, nodes))

        dt = self.dt_local
        ws = self.ws
        r0 = self.r0
        u_star = np.sqrt(self.Cf * U_node[nodes] * U_node[nodes])
        es = get_es(self.R,
                    self.g,
                    self.Ds,
                    self.nu,
                    u_star,
                    function=self.sed_entrainment_func)

        out_Ch[nodes] = (Ch[nodes] + ws * es * dt)
        out_Ch[nodes] /= (1 + ws * r0 / h[nodes] * dt)
        out_eta[nodes] = eta[nodes] + Ch[nodes] - out_Ch[nodes]

        return out_Ch, out_eta

    def copy_values_to_temp(self):
        self.h_temp[:] = self.h[:]
        self.dhdx_temp[:] = self.dhdx[:]
        self.dhdy_temp[:] = self.dhdy[:]
        self.h_link_temp[:] = self.h_link[:]
        self.u_temp[:] = self.u[:]
        self.dudx_temp[:] = self.dudx[:]
        self.dudy_temp[:] = self.dudy[:]
        self.u_node_temp[:] = self.u_node[:]
        self.v_temp[:] = self.v[:]
        self.dvdx_temp[:] = self.dvdx[:]
        self.dvdy_temp[:] = self.dvdy[:]
        self.v_node_temp[:] = self.v_node[:]
        self.Ch_temp[:] = self.Ch[:]
        self.dChdx_temp[:] = self.dChdx[:]
        self.dChdy_temp[:] = self.dChdy[:]
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
        all_wet_nodes = np.where(self.h > self.C_init)
        self.C[all_wet_nodes] = self.Ch[all_wet_nodes] / self.h[all_wet_nodes]
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
        self.grid.at_node['flow_depth__horizontal_gradient'] = self.dhdx
        self.grid.at_node['flow_depth__vertical_gradient'] = self.dhdy
        self.grid.at_node[
            'flow_sediment_volume__horizontal_gradient'] = self.dChdx
        self.grid.at_node[
            'flow_sediment_volume__vertical_gradient'] = self.dChdy

    def update_values(self):
        """Update variables from temporally variables and
           apply boundary conditions
        """

        # adjust illeagal values
        # self.h_temp[np.where(self.h_temp < self.h_init)] = self.h_init
        # self.Ch_temp[np.where(self.Ch_temp <= 0)] = self.C_init * self.h_init
        adjust_negative_values(
            self.h_temp,
            self.Ch_temp,
            self.wet_pwet_nodes,
            self.node_east,
            self.node_west,
            self.node_north,
            self.node_south,
            out_h=self.h_temp,
            out_Ch=self.Ch_temp,
        )

        # copy values from temp to grid values
        self.h[:] = self.h_temp[:]
        self.dhdx[:] = self.dhdx_temp[:]
        self.dhdy[:] = self.dhdy_temp[:]
        self.u[:] = self.u_temp[:]
        self.dudx[:] = self.dudx_temp[:]
        self.dudy[:] = self.dudy_temp[:]
        self.v[:] = self.v_temp[:]
        self.dvdx[:] = self.dvdx_temp[:]
        self.dvdy[:] = self.dvdy_temp[:]
        self.Ch[:] = self.Ch_temp[:]
        self.dChdx[:] = self.dChdx_temp[:]
        self.dChdy[:] = self.dChdy_temp[:]
        self.eta[:] = self.eta_temp[:]
        self.U[:] = self.U_temp[:]
        self.U_node[:] = self.U_node_temp[:]

        # update boundary conditions
        self.update_boundary_conditions(h=self.h,
                                        u=self.u,
                                        v=self.v,
                                        Ch=self.Ch,
                                        h_link=self.h_link,
                                        Ch_link=self.Ch_link,
                                        u_node=self.u_node,
                                        v_node=self.v_node,
                                        eta=self.eta)

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

        if self.water_entrainment is True:
            self.ew_node[core_nodes] = get_ew(U_node[core_nodes],
                                              Ch[core_nodes], self.R, self.g)
        else:
            self.ew_node[core_nodes] = 0

        self.G_h[core_nodes] = self.ew_node[core_nodes] * U_node[core_nodes]

    def calc_G_Ch(self, Ch, Ch_link, u, v, core_nodes):
        """Calculate non-advection term for Ch
        """
        self.G_Ch[core_nodes] = 0

    def calc_G_u(self,
                 h,
                 h_link,
                 u,
                 v,
                 Ch,
                 Ch_link,
                 eta,
                 U,
                 link_horiz,
                 scheme='CCUP'):
        """Calculate non-advection term for u
        """
        east_node = self.east_node_at_horizontal_link[link_horiz]
        west_node = self.west_node_at_horizontal_link[link_horiz]
        Rg = self.R * self.g
        dx = self.grid.dx

        eta_grad_at_link = self.grid.calc_grad_at_link(eta)
        eta_grad_x = eta_grad_at_link[link_horiz]

        h_link_horiz = h_link[link_horiz]
        Ch_link_horiz = Ch_link[link_horiz]
        C_horiz = np.zeros(h_link_horiz.shape)
        wet_horiz = h_link_horiz > self.h_w
        C_horiz[~wet_horiz] = 0
        C_horiz[wet_horiz] = Ch_link_horiz[wet_horiz] / h_link_horiz[wet_horiz]

        if scheme == 'CentralDifference':
            self.G_u[link_horiz] = -Rg * (
                C_horiz * eta_grad_x + 0.5 *
                (Ch[east_node] - Ch[west_node]) / dx + C_horiz *
                (h[east_node] - h[west_node]) / dx)
        else:
            self.G_u[link_horiz] = -Rg * C_horiz * eta_grad_x

    def calc_G_v(self,
                 h,
                 h_link,
                 u,
                 v,
                 Ch,
                 Ch_link,
                 eta,
                 U,
                 link_vert,
                 scheme='CCUP'):
        """Calculate non-advection term for v
        """

        north_node = self.north_node_at_vertical_link[link_vert]
        south_node = self.south_node_at_vertical_link[link_vert]
        Rg = self.R * self.g
        dx = self.grid.dx

        eta_grad_at_link = self.grid.calc_grad_at_link(eta)
        eta_grad_y = eta_grad_at_link[link_vert]

        h_link_vert = h_link[link_vert]
        Ch_link_vert = Ch_link[link_vert]
        C_vert = np.zeros(h_link_vert.shape)
        wet_vert = h_link_vert > self.h_w
        C_vert[~wet_vert] = 0
        C_vert[wet_vert] = Ch_link_vert[wet_vert] / h_link_vert[wet_vert]

        if scheme == 'CentralDifference':
            self.G_v[link_vert] = -Rg * (
                C_vert * eta_grad_y + 0.5 *
                (Ch[north_node] - Ch[south_node]) / dx + C_vert *
                (h[north_node] - h[south_node]) / dx)
        else:
            self.G_v[link_vert] = -Rg * C_vert * eta_grad_y

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
        u_star_at_node = np.sqrt(self.Cf * U_node[core] * U_node[core])
        self.es = get_es(self.R,
                         self.g,
                         self.Ds,
                         self.nu,
                         u_star_at_node,
                         function=self.sed_entrainment_func)

        out_geta[core] = ws * (r0 * Ch[core] / h[core] - self.es)

        return out_geta

        # remove too large gradients
        # maxC = 0.05
        # illeagal_val = np.where(out_geta[core] * self.dt_local > Ch[core])
        # out_geta[core][illeagal_val] = Ch[core[illeagal_val]] / self.dt_local
        # illeagal_val2 = np.where(
        #     Ch[core] - out_geta[core] * self.dt_local > maxC * h[core])
        # out_geta[core][illeagal_val2] = (
        #     maxC * h[core][illeagal_val2] -
        #     Ch[core][illeagal_val2]) / self.dt_local

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
        write_netcdf(filename,
                     self.grid,
                     names=[
                         'topographic__elevation',
                         'flow__depth',
                         'flow__horizontal_velocity_at_node',
                         'flow__vertical_velocity_at_node',
                         'flow__surface_elevation',
                         'flow__sediment_concentration',
                         'bed__thickness',
                     ],
                     at='node')

    def update_boundary_conditions(
            self,
            h=None,
            u=None,
            v=None,
            Ch=None,
            h_link=None,
            Ch_link=None,
            u_node=None,
            v_node=None,
            eta=None,
            p=None,
    ):
        """Update boundary conditions
        """
        edge_nodes = self.grid.nodes_at_link[self.fixed_value_edge_links]

        if h is not None:
            h[self.fixed_grad_nodes] = h[self.fixed_grad_anchor_nodes]

        if Ch is not None:
            Ch[self.fixed_grad_nodes] = Ch[self.fixed_grad_anchor_nodes]

        if u is not None:
            u[self.fixed_grad_links] = u[self.fixed_grad_anchor_links]
            u[self.fixed_grad_link_at_east[
                u[self.fixed_grad_link_at_east] < 0]] = 0
            u[self.fixed_grad_link_at_west[
                u[self.fixed_grad_link_at_west] > 0]] = 0
            u[self.fixed_value_links] = (
                2. / 3.) * u_node[self.fixed_value_nodes] + (
                    1. / 3.) * u[self.fixed_value_anchor_links]
            u[self.fixed_value_edge_links] = np.mean(u_node[edge_nodes],
                                                     axis=1)

        if v is not None:
            v[self.fixed_grad_links] = v[self.fixed_grad_anchor_links]
            v[self.fixed_grad_link_at_north[
                v[self.fixed_grad_link_at_north] < 0]] = 0
            v[self.fixed_grad_link_at_south[
                v[self.fixed_grad_link_at_south] > 0]] = 0
            v[self.fixed_value_links] = (
                2. / 3.) * v_node[self.fixed_value_nodes] + (
                    1. / 3.) * v[self.fixed_value_anchor_links]
            v[self.fixed_value_edge_links] = np.mean(v_node[edge_nodes],
                                                     axis=1)

        if h_link is not None:
            h_link[self.fixed_grad_links] = h_link[
                self.fixed_grad_anchor_links]
            h_link[self.fixed_value_links] = (
                h[self.fixed_value_nodes] +
                h[self.fixed_value_anchor_nodes]) / 2.0
            h_link[self.fixed_value_edge_links] = np.mean(h[edge_nodes],
                                                          axis=1)

        if Ch_link is not None:
            Ch_link[self.fixed_grad_links] = Ch_link[
                self.fixed_grad_anchor_links]
            Ch_link[self.fixed_value_links] = (
                Ch[self.fixed_value_nodes] +
                Ch[self.fixed_value_anchor_nodes]) / 2.0
            Ch_link[self.fixed_value_edge_links] = np.mean(Ch[edge_nodes],
                                                           axis=1)

        if u_node is not None:
            u_node[self.fixed_grad_nodes] = u_node[
                self.fixed_grad_anchor_nodes]

        if v_node is not None:
            v_node[self.fixed_grad_nodes] = v_node[
                self.fixed_grad_anchor_nodes]

        if p is not None:
            p[self.fixed_grad_nodes] = p[self.fixed_grad_anchor_nodes]

        if eta is not None:
            eta[self.fixed_grad_nodes] = eta[self.fixed_grad_anchor_nodes]
            eta[self.fixed_value_nodes] = eta[self.fixed_value_anchor_nodes]


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
        canyon='parabola',
):
    # making grid
    # size of calculation domain is 4 x 8 km with dx = 20 m
    lgrids = int(length / spacing)
    wgrids = int(width / spacing)
    grid = RasterModelGrid((lgrids, wgrids), xy_spacing=[spacing, spacing])
    grid.add_zeros('flow__depth', at='node')
    grid.add_zeros('topographic__elevation', at='node')
    grid.add_zeros('flow__horizontal_velocity_at_node', at='node')
    grid.add_zeros('flow__vertical_velocity_at_node', at='node')
    grid.add_zeros('flow__horizontal_velocity', at='link')
    grid.add_zeros('flow__vertical_velocity', at='link')
    grid.add_zeros('flow__sediment_concentration', at='node')
    grid.add_zeros('bed__thickness', at='node')

    # making topography
    # set the slope
    grid.at_node['topographic__elevation'] = (
        grid.node_y - slope_basin_break) * slope_outside

    if canyon == 'parabola':
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
                                   spacing=500,
                                   filter_size=[1, 1]):

    # read a geotiff file into ndarray
    topo_file = gdal.Open(geotiff_filename, gdalconst.GA_ReadOnly)
    topo_data = topo_file.GetRasterBand(1).ReadAsArray()
    if (xlim is not None) and (ylim is not None):
        topo_data = topo_data[xlim[0]:xlim[1], ylim[0]:ylim[1]]

    # Smoothing by median filter
    topo_data = median_filter(topo_data, size=filter_size)

    grid = RasterModelGrid(topo_data.shape, xy_spacing=[spacing, spacing])
    grid.add_zeros('flow__depth', at='node')
    grid.add_zeros('topographic__elevation', at='node')
    grid.add_zeros('flow__horizontal_velocity', at='link')
    grid.add_zeros('flow__vertical_velocity', at='link')
    grid.add_zeros('flow__sediment_concentration', at='node')
    grid.add_zeros('bed__thickness', at='node')

    grid.at_node['topographic__elevation'][grid.nodes] = topo_data

    return grid


def test_ode_grad(Ch, t, tc, h, u_node, v_node, U_node, nodes):

    G_eta = tc.calc_G_eta(h, u_node, v_node, Ch, U_node, nodes)
    return G_eta


if __name__ == '__main__':
    """This is a script to run the model of TurbidityCurrent2D
    """
    import os
    os.environ['MKL_NUM_THREADS'] = '6'
    os.environ['OMP_NUM_THREADS'] = '6'
    from landlab import FIXED_GRADIENT_BOUNDARY, FIXED_VALUE_BOUNDARY
    from tqdm import tqdm

    grid = create_topography(
        length=8000,
        width=2000,
        spacing=10,
        slope_outside=0.2,
        slope_inside=0.05,
        slope_basin_break=1000,  #2000
        canyon_basin_break=1200,  #2200
        canyon_center=1000,
        canyon_half_width=100,
    )
    # grid = create_topography_from_geotiff('depth500.tif',
    #                                       xlim=[200, 800],
    #                                       ylim=[400, 1200],
    #                                       spacing=500)

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
    #     initial_flow_thickness=200,
    #     initial_region_radius=30000,
    #     initial_region_center=[100000, 125000],
    # )

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
    last = 3

    for i in tqdm(range(1, last + 1), disable=False):
        tc.run_one_step(dt=10.0)
        tc.save_nc('tc{:04d}.nc'.format(i))
        if np.sum(tc.Ch) / Ch_init < 0.01:
            break
    tc.save_grid('tc{:04d}.nc'.format(i))
    print('elapsed time: {} sec.'.format(time.time() - t))
