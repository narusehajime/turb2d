from .gridutils import set_up_neighbor_arrays, update_up_down_links_and_nodes
from .gridutils import map_values, map_links_to_nodes, map_nodes_to_links
from .gridutils import find_boundary_links_nodes, adjust_negative_values
from .gridutils import write_netcdf
from .utils import (
    create_init_flow_region,
    create_topography,
    create_topography_from_geotiff,
)
from .wetdry import find_wet_grids, process_partial_wet_grids
from .sediment_func import get_es, get_ew, get_ws, get_det_rate, get_bedload
from .cip import update_gradient, update_gradient2
from .cip import CIP2D, Jameson, SOR
from landlab.io.native_landlab import save_grid

# from landlab.io.netcdf import write_netcdf, write_raster_netcdf

from . import _links as links
from landlab import Component, FieldError
import numpy as np
import time
from tqdm import tqdm


"""A component of landlab that simulates a turbidity current on 2D grids

This component simulates turbidity currents using the 2-D numerical model of
shallow-water equations over topography on the basis of 3 equation model of
Parker et al. (1986).

.. codeauthor: : Hajime Naruse

Examples
---------
import turb2d
turb2d.run()

"""


class TurbidityCurrent2D(Component):
    """Simulate a turbidity current using the CIP method.

    Landlab component that simulates turbidity current using the CIP method
    for solving the 2D shallow water equations.

    This component calculates flow depth, shear stress across any raster grid.
    Default input file is named "turbidity_current_input.txt" and is
    contained in the turb2d directory.

    The primary method of this class is: func: 'run_one_step'
    """

    _name = "TurbidityCurrent2D"

    _cite_as = """Naruse in prep."""

    _input_var_names = (
        "flow__depth",
        "flow__horizontal_velocity",
        "flow__vertical_velocity",
        "flow__sediment_concentration_i",
        "flow__sediment_concentration_total",
        "flow__TKE",
        "topographic__elevation",
        "bed__thickness",
        "flow__surface_elevation",
        "bed__active_layer_fraction_i",
        "bed__sediment_volume_per_unit_area_i",
    )

    _output_var_names = (
        "flow__depth",
        "flow__horizontal_velocity",
        "flow__vertical_velocity",
        "flow__sediment_concentration_i",
        "flow__sediment_concentration_total",
        "flow__TKE",
        "topographic__elevation",
        "bed__thickness",
        "flow__surface_elevation",
        "bed__active_layer_fraction_i",
        "bed__sediment_volume_per_unit_area_i",
    )

    _var_units = {
        "flow__depth": "m",
        "flow__horizontal_velocity": "m/s",
        "flow__vertical_velocity": "m/s",
        "flow__sediment_concentration_i": "1",
        "flow__sediment_concentration_total": "1",
        "flow__TKE": "m^2/s^2",
        "topographic__elevation": "m",
        "bed__thickness": "m",
        "flow__surface_elevation": "m",
        "bed__active_layer_fraction_i": "1",
        "bed__sediment_volume_per_unit_area_i": "m",
    }

    _var_mapping = {
        "flow__depth": "node",
        "flow__horizontal_velocity": "link",
        "flow__vertical_velocity": "link",
        "flow__TKE": "link",
        "flow__sediment_concentration_i": "node",
        "flow__sediment_concentration_total": "node",
        "topographic__elevation": "node",
        "bed__thickness": "node",
        "flow__surface_elevation": "node",
        "bed__active_layer_fraction_i": "node",
        "bed__sediment_volume_per_unit_area_i": "node",
    }

    _var_doc = {
        "flow__depth": "The depth of flow at each node.",
        "flow__horizontal_velocity": "Horizontal component of"
        " flow velocity at each link",
        "flow__vertical_velocity": "Vertical component of flow velocity at each link",
        "flow__sediment_concentration_i": "Sediment concentration"
        " of the ith grain size class in flow",
        "flow__sediment_concentration_total": "Total sediment concentration",
        "flow__TKE": "Turbulent kinetic energy",
        "topographic__elevation": "The land surface elevation.",
        "bed__thickness": "The bed thickness",
        "flow__surface_elevation": "Elevation of flow surface",
        "bed__active_layer_fraction_i": "Fraction of "
        "the ith grain size class in active layer",
        "bed__sediment_volume_per_unit_area_i": "Sediment volume per"
        " unit area of the ith grain size class",
    }

    def __init__(
            self,
            grid,
            h_init=0.0,
            Ch_w=10 ** (-4),
            h_w=0.01,
            alpha=0.1,
            Cf=0.004,
            g=9.81,
            R=1.65,
            Ds=100 * 10**-6,
            lambda_p=0.4,
            r0=1.5,
            nu=1.010 * 10**-6,
            kappa=0.01,
            nu_a=0.8,
            implicit_num=100,
            implicit_threshold=1.0 * 10**-12,
            C_init=0.0,
            gamma=0.35,
            la=0.01,
            water_entrainment=True,
            water_detrainment=True,
            det_factor=1.0,
            suspension=True,
            bedload_transport=True,
            sed_entrainment_func="GP1991field",
            no_erosion=True,
            model="3eq",
            dt_local_limit=5.0,
            **kwds
    ):
        """Create a component of turbidity current

        Parameters
        ----------
        grid: RasterModelGrid
            A landlab grid.
        h_init: float, optional
            Thickness of initial thin layer of flow to prevent divide by zero
            errors(m).
        Ch_w: float, optional
            Volume-per-unit-area of suspended load (Ch) to judge "wet" nodes
             and links(m).
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
        Ds: float of list or ndarray, optional
            Sediment diameter(m). If a list of grain sizes is given, the model deals
             with multiple grain size classes
        lambda_p: float, optional
            Bed sediment porosity(1)
        nu: float, optional
            Kinematic viscosity of water(at 293K)
        kappa: float, optional
            Second order artificial viscosity. This value, alpha and
            h_w affect to calculation stability.
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
        la: float, optional
            Thickness of active layer on the bed
        suspension: boolean, optional
            turn on the function for entrainment/settling of suspension
        bedload_transport: boolean, optional
            turn on the funciton for bedload deposition/erosion
        water_entrainment: boolean, optional
            turn on the function for ambient water entrainment
        water_detrainment: boolean, optional
            turn on the function for ambient water detrainment
        sed_entrainment_func: string, optional
            Choose the function to be used for sediment entrainment. Default
            is 'GP1991field', and other options are: 'GP1991exp', 'vanRijn1984'
        no_erosion: boolean, optional
            If True, bed cannot be eroded
        model: string, optional
            Choose "3eq" or "4eq" for the three or four equation model of Parker
            (1986)
        """
        # First we copy our grid
        self._grid = grid

        # Copy model parameters
        self.h_init = h_init
        self.alpha = alpha
        self.Cf = Cf
        self.g = g
        self.R = R
        if type(Ds) is float:
            self.Ds = np.array([Ds]).reshape(1, 1)
        else:
            self.Ds = np.array(Ds).reshape(len(Ds), 1)
        self.number_gclass = len(self.Ds)
        self.Ch_w = Ch_w
        self.h_w = h_w
        self.nu = nu
        self.kappa = kappa
        self.nu_a = nu_a
        self.r0 = r0
        self.lambda_p = lambda_p
        self.implicit_num = implicit_num
        self.implicit_threshold = implicit_threshold
        self.C_init = C_init
        self.gamma = gamma
        self.la = la
        self.water_entrainment = water_entrainment
        self.water_detrainment = water_detrainment
        self.det_factor = det_factor
        self.suspension = suspension
        self.bedload_transport = bedload_transport
        self.sed_entrainment_func = sed_entrainment_func
        self.model = model
        self.karman = 0.4
        self.no_erosion = no_erosion
        self.dt_local_limit = dt_local_limit
        self.alpha_TKE = kwds.pop("alpha_TKE", 0.1)

        # Perform the parent class initializer
        super(TurbidityCurrent2D, self).__init__(grid, **kwds)


        # Now setting up fields at nodes and links
        try:
            self.eta = grid.add_zeros(
                "topographic__elevation",
                at="node",
                units=self._var_units["topographic__elevation"],
            )

        except FieldError:
            # Field was already set
            self.eta = self._grid.at_node["topographic__elevation"]

        try:
            self.bed_thick = grid.add_zeros(
                "bed__thickness", at="node", units=self._var_units["bed__thickness"]
            )
            # Record initial topography
            self.eta_init = self.eta.copy()
        except FieldError:
            # Field was already set
            self.bed_thick = self._grid.at_node["bed__thickness"]
            self.eta_init = self.eta[:] - self.bed_thick[:]

        try:
            self.bed_thick_i = np.empty(
                [self.number_gclass, grid.number_of_nodes], dtype=float
            )
            for i in range(self.number_gclass):
                self.bed_thick_i[i, :] = grid.add_zeros(
                    "bed__sediment_volume_per_unit_area_" + str(i),
                    at="node",
                    units=self._var_units["bed__sediment_volume_per_unit_area_i"],
                )
        except FieldError:
            for i in range(self.number_gclass):
                self.bed_thick_i[i, :] = grid.at_node[
                    "bed__sediment_volume_per_unit_area_" + str(i)
                ]

        try:
            self.bed_active_layer = np.empty(
                [self.number_gclass, grid.number_of_nodes], dtype=float
            )
            for i in range(self.number_gclass):
                self.bed_active_layer[i, :] = (
                    grid.add_ones(
                        "bed__active_layer_fraction_" + str(i),
                        at="node",
                        units=self._var_units["bed__active_layer_fraction_i"],
                    )
                    / self.number_gclass
                )
        except FieldError:
            for i in range(self.number_gclass):
                self.bed_active_layer[i, :] = grid.at_node[
                    "bed__active_layer_fraction_" + str(i)
                ]

        try:
            self.h = grid.add_zeros(
                "flow__depth", at="node", units=self._var_units["flow__depth"]
            )
        except FieldError:
            # Field was already set
            self.h = grid.at_node["flow__depth"]

        try:
            self.C_i = np.empty([self.number_gclass, grid.number_of_nodes])
            self.Ch_i = np.empty([self.number_gclass, grid.number_of_nodes])
            for i in range(self.number_gclass):
                self.C_i[i, :] = grid.add_zeros(
                    "flow__sediment_concentration_" + str(i),
                    at="node",
                    units=self._var_units["flow__sediment_concentration_i"],
                )
                self.Ch_i[i, :] = self.C_i[i, :] * self.h
            self.C = grid.add_zeros(
                "flow__sediment_concentration_total",
                at="node",
                units=self._var_units["flow__sediment_concentration_total"],
            )
            self.Ch = self.C * self.h
        except FieldError:
            # Field was already set
            for i in range(self.number_gclass):
                self.C_i[i, :] = grid.at_node["flow__sediment_concentration_" + str(i)]
                self.Ch_i[i, :] = self.C_i[i, :] * self.h
            self.C = np.sum(self.C_i, axis=0)
            self.Ch = np.sum(self.Ch_i, axis=0)

        try:
            grid.add_zeros(
                "flow__surface_elevation",
                at="node",
                units=self._var_units["flow__surface_elevation"],
            )
            grid.at_node["flow__surface_elevation"] = self.h + self.eta

        except FieldError:
            grid.at_node["flow__surface_elevation"] = self.h + self.eta

        try:
            self.u = grid.add_zeros(
                "flow__horizontal_velocity",
                at="link",
                units=self._var_units["flow__horizontal_velocity"],
            )
            self.v = grid.add_zeros(
                "flow__vertical_velocity",
                at="link",
                units=self._var_units["flow__vertical_velocity"],
            )

        except FieldError:
            # Field was already set
            self.u = grid.at_link["flow__horizontal_velocity"]
            self.v = grid.at_link["flow__vertical_velocity"]

        try:
            self.Kh = grid.add_zeros(
                "flow__TKE", at="link", units=self._var_units["flow__TKE"]
            )
        except FieldError:
            # Field was already set
            self.Kh = grid.at_link["flow__TKE"]

        try:
            self.u_node = grid.add_zeros(
                "flow__horizontal_velocity_at_node",
                at="node",
                units=self._var_units["flow__horizontal_velocity"],
            )
            self.v_node = grid.add_zeros(
                "flow__vertical_velocity_at_node",
                at="node",
                units=self._var_units["flow__vertical_velocity"],
            )
        except FieldError:
            self.u_node = grid.at_node["flow__horizontal_velocity_at_node"]
            self.v_node = grid.at_node["flow__vertical_velocity_at_node"]

        self.h += self.h_init
        self.C += self.C_init
        self.Ch += self.h_init * self.C_init
        self.C_i += self.C_init
        self.Ch_i += self.C_init * self.h_init

        # For gradient of parameters at nodes and links
        try:
            self.dudx = grid.add_zeros(
                "flow_horizontal_velocity__horizontal_gradient", at="link"
            )
            self.dudy = grid.add_zeros(
                "flow_horizontal_velocity__vertical_gradient", at="link"
            )
            self.dvdx = grid.add_zeros(
                "flow_vertical_velocity__horizontal_gradient", at="link"
            )
            self.dvdy = grid.add_zeros(
                "flow_vertical_velocity__vertical_gradient", at="link"
            )

        except FieldError:
            self.dudx = grid.at_link["flow_horizontal_velocity__horizontal_gradient"]
            self.dudy = grid.at_link["flow_horizontal_velocity__vertical_gradient"]
            self.dvdx = grid.at_link["flow_vertical_velocity__horizontal_gradient"]
            self.dvdy = grid.at_link["flow_vertical_velocity__vertical_gradient"]

        try:
            self.dhdx = grid.add_zeros("flow_depth__horizontal_gradient", at="node")
            self.dhdy = grid.add_zeros("flow_depth__vertical_gradient", at="node")

        except FieldError:
            self.dhdx = grid.at_node["flow_depth__horizontal_gradient"]
            self.dhdy = grid.at_node["flow_depth__vertical_gradient"]

        try:
            self.dChdx_i = np.empty([self.number_gclass, grid.number_of_nodes])
            self.dChdy_i = np.empty([self.number_gclass, grid.number_of_nodes])
            for i in range(self.number_gclass):
                self.dChdx_i[i, :] = grid.add_zeros(
                    "flow_sediment_volume__horizontal_gradient_" + str(i), at="node"
                )
                self.dChdy_i[i, :] = grid.add_zeros(
                    "flow_sediment_volume__vertical_gradient_" + str(i), at="node"
                )
        except FieldError:
            for i in range(self.number_gclass):
                self.dChdx_i[i, :] = grid.at_node[
                    "flow_sediment_volume__horizontal_gradient_" + str(i)
                ]
                self.dChdy_i[i, :] = grid.at_node[
                    "flow_sediment_volume__vertical_gradient_" + str(i)
                ]

        try:
            self.dKhdx = grid.add_zeros("flow_TKE__horizontal_gradient", at="link")
            self.dKhdy = grid.add_zeros("flow_TKE__vertical_gradient", at="link")

        except FieldError:
            self.dKhdx = grid.at_link["flow_TKE__horizontal_gradient"]
            self.dKhdy = grid.at_link["flow_TKE__vertical_gradient"]

        # record active links
        self.active_link_ids = links.active_link_ids(
            self.grid.shape, self.grid.status_at_node
        )
        self.horizontal_active_links = links.horizontal_active_link_ids(
            self.grid.shape, self.active_link_ids
        )
        self.horizontal_active_links = np.delete(
            self.horizontal_active_links, np.where(self.horizontal_active_links < 0)
        )
        self.vertical_active_links = links.vertical_active_link_ids(
            self.grid.shape, self.active_link_ids
        )
        self.vertical_active_links = np.delete(
            self.vertical_active_links, np.where(self.vertical_active_links < 0)
        )

        # Set up temporal parameters
        # self.Ch = self.C * self.h
        self.u_node = np.zeros(grid.number_of_nodes)
        self.v_node = np.zeros(grid.number_of_nodes)
        self.Kh_node = np.zeros(grid.number_of_nodes)
        self.h_link = np.zeros(grid.number_of_links)
        self.Ch_link = np.zeros(grid.number_of_links)
        self.Ch_link_i = np.zeros([self.number_gclass, grid.number_of_links])

        # friction coefficients at grids
        self.Cf_link = np.ones(grid.number_of_links) * self.Cf
        self.Cf_node = np.ones(grid.number_of_nodes) * self.Cf

        # composite velocity
        self.U = np.zeros(grid.number_of_links)
        self.U_node = np.zeros(grid.number_of_nodes)

        # water and sediment entrainment rates
        self.ew_node = np.zeros(grid.number_of_nodes)
        self.ew_link = np.zeros(grid.number_of_links)

        # diffusion coefficient of velocity
        self.nu_t = np.zeros(grid.number_of_links)

        # temporary values for flow depth
        self.h_temp = np.zeros(grid.number_of_nodes)
        self.dhdx_temp = np.zeros(grid.number_of_nodes)
        self.dhdy_temp = np.zeros(grid.number_of_nodes)
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

        # temporary values for TKE
        self.Kh_temp = np.zeros(grid.number_of_links)
        self.Kh_node_temp = np.zeros(grid.number_of_nodes)
        self.dKhdx_temp = np.zeros(grid.number_of_links)
        self.dKhdy_temp = np.zeros(grid.number_of_links)

        # temporary value to store divergence of velocity
        self.div = np.zeros(grid.number_of_nodes)

        # temporary variable to store pressure
        self.p = np.zeros(grid.number_of_nodes)
        self.p_temp = np.zeros(grid.number_of_nodes)

        # temporary values for wave celerity
        self.Cs = np.zeros(grid.number_of_nodes)

        # volume of suspended sediment
        self.Ch_i_temp = np.zeros([self.number_gclass, grid.number_of_nodes])
        self.Ch_temp = np.zeros(grid.number_of_nodes)
        self.dChdx_i_temp = np.zeros([self.number_gclass, grid.number_of_nodes])
        self.dChdy_i_temp = np.zeros([self.number_gclass, grid.number_of_nodes])
        self.Ch_i_prev = np.zeros([self.number_gclass, grid.number_of_nodes])
        self.Ch_link_i_temp = np.zeros([self.number_gclass, grid.number_of_links])
        self.Ch_link_temp = np.zeros(grid.number_of_links)

        # topographic elevation and slope
        self.eta_temp = self.eta.copy()
        self.bed_thick_i_temp = self.bed_thick_i.copy()
        self.S = grid.calc_grad_at_link(self.eta)
        self.bed_change_i = np.zeros_like(self.bed_thick_i)
        self.es = np.zeros_like(self.bed_thick_i)
        self.bedload_total = np.zeros_like(self.bed_thick_i)
        self.bedload_x = np.zeros_like(self.bed_thick_i)
        self.bedload_y = np.zeros_like(self.bed_thick_i)

        # length of flow velocity vector
        self.U_temp = self.U.copy()
        self.U_node_temp = self.U_node.copy()

        # artificial viscosity
        self.q = np.zeros(grid.number_of_nodes)

        # arrays to record upcurrent and downcurrent nodes
        self.horizontal_up_nodes = np.zeros(grid.number_of_nodes, dtype=np.int64)
        self.vertical_up_nodes = np.zeros(grid.number_of_nodes, dtype=np.int64)
        self.horizontal_down_nodes = np.zeros(grid.number_of_nodes, dtype=np.int64)
        self.vertical_down_nodes = np.zeros(grid.number_of_nodes, dtype=np.int64)

        self.horizontal_up_links = np.zeros(grid.number_of_links, dtype=np.int64)
        self.vertical_up_links = np.zeros(grid.number_of_links, dtype=np.int64)
        self.horizontal_down_links = np.zeros(grid.number_of_links, dtype=np.int64)
        self.vertical_down_links = np.zeros(grid.number_of_links, dtype=np.int64)

        # arrays to refer adjacent links and nodes
        self.east_link_at_node = np.zeros(grid.number_of_nodes, dtype=np.int64)
        self.north_link_at_node = np.zeros(grid.number_of_nodes, dtype=np.int64)
        self.west_link_at_node = np.zeros(grid.number_of_nodes, dtype=np.int64)
        self.south_link_at_node = np.zeros(grid.number_of_nodes, dtype=np.int64)
        self.west_node_at_horizontal_link = np.zeros(
            grid.number_of_links, dtype=np.int64
        )
        self.east_node_at_horizontal_link = np.zeros(
            grid.number_of_links, dtype=np.int64
        )
        self.south_node_at_vertical_link = np.zeros(
            grid.number_of_links, dtype=np.int64
        )
        self.north_node_at_vertical_link = np.zeros(
            grid.number_of_links, dtype=np.int64
        )

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
        self.wettest_nodes = None
        self.dry_nodes = None
        self.dry_links = None
        self.horizontal_overspill_velocity = None
        self.vertical_overspill_velocity = None

        # Calculate subordinate parameters
        self.ws = get_ws(self.R, self.g, self.Ds, self.nu)

        # Start time of simulation is at 0 s
        grid.at_grid["time__elapsed"] = 0.0
        self.elapsed_time = grid.at_grid["time__elapsed"]

        self.dt = None
        self.dt_local = None
        self.first_step = True
        # self.first_stage_count = 0

        self.neighbor_flag = False

    def calc_time_step(self):
        """Calculate time step"""

        Ch_is_positive = self.Ch > 0
        self.Cs[Ch_is_positive] = np.sqrt(self.R * self.g * self.Ch[Ch_is_positive])
        self.Cs[~Ch_is_positive] = 0

        dt_local = (
            self.alpha
            * self.grid.dx
            / np.amax(
                np.array(
                    [
                        np.amax(np.abs(self.u_node) + self.Cs),
                        np.amax(np.abs(self.v_node) + self.Cs),
                        1.0,
                    ]
                )
            )
        )
        if dt_local > self.dt_local_limit:
            dt_local = self.dt_local_limit

        if self.first_step is True:
            dt_local *= 0.1

        return dt_local

    def run_one_step(self, dt=None):
        """Generate a turbidity current across a grid.

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
        local_elapsed_time = 0.0
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
        self.h = self.grid["node"]["flow__depth"]
        self.eta = self.grid["node"]["topographic__elevation"]
        for i in range(self.number_gclass):
            self.bed_thick_i[i, :] = self.grid["node"][
                "bed__sediment_volume_per_unit_area_" + str(i)
            ]
        self.bed_thick[:] = np.sum(self.bed_thick_i, axis=0)
        self.u = self.grid["link"]["flow__horizontal_velocity"]
        self.v = self.grid["link"]["flow__vertical_velocity"]
        self.u_node = self.grid["node"]["flow__horizontal_velocity_at_node"]
        self.v_node = self.grid["node"]["flow__vertical_velocity_at_node"]
        for i in range(self.number_gclass):
            self.C_i[i, :] = self.grid["node"]["flow__sediment_concentration_" + str(i)]
            self.Ch_i[i, :] = self.C_i[i, :] * self.h
        self.C[:] = np.sum(self.C_i, axis=0)
        self.Ch[:] = np.sum(self.Ch_i, axis=0)

        # create numerical solver objects
        self.cip2d = CIP2D(self.grid.number_of_links)
        self.sor = SOR(
            self.grid.number_of_nodes,
            self.node_east,
            self.node_west,
            self.node_north,
            self.node_south,
            self.implicit_threshold,
            self.implicit_num,
            self.alpha,
            self.update_boundary_conditions,
        )
        self.jameson = Jameson(self)

        # Initialize boundary conditions and wet/dry grids
        find_boundary_links_nodes(self)
        find_wet_grids(self)
        map_values(
            self,
            h=self.h,
            dhdx=self.dhdx,
            dhdy=self.dhdy,
            u=self.u,
            dudx=self.dudx,
            v=self.v,
            dvdy=self.dvdy,
            Ch_i=self.Ch_i,
            dChdx_i=self.dChdx_i,
            dChdy_i=self.dChdy_i,
            Ch=self.Ch,
            eta=self.eta,
            h_link=self.h_link,
            u_node=self.u_node,
            v_node=self.v_node,
            Ch_link=self.Ch_link,
            Ch_link_i=self.Ch_link_i,
            U=self.U,
            U_node=self.U_node,
            Cf_link=self.Cf_link,
            Cf_node=self.Cf_node,
        )
        self.update_boundary_conditions(
            h=self.h,
            u=self.u,
            v=self.v,
            Ch_i=self.Ch_i,
            h_link=self.h_link,
            Ch_link_i=self.Ch_link_i,
            u_node=self.u_node,
            v_node=self.v_node,
            eta=self.eta,
        )
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
            map_values(
                self,
                h=self.h,
                dhdx=self.dhdx,
                dhdy=self.dhdy,
                u=self.u,
                dudx=self.dudx,
                v=self.v,
                dvdy=self.dvdy,
                Ch_i=self.Ch_i,
                dChdx_i=self.dChdx_i,
                dChdy_i=self.dChdy_i,
                Ch=self.Ch,
                eta=self.eta,
                h_link=self.h_link,
                u_node=self.u_node,
                v_node=self.v_node,
                Ch_link=self.Ch_link,
                Ch_link_i=self.Ch_link_i,
                U=self.U,
                U_node=self.U_node,
            )

            # Process partial wet grids
            self._process_wet_dry_boundary()

            # calculate non-advection terms using implicit method
            self._nonadvection_phase()

            # calculate advection terms using cip method
            self._advection_phase()

            # # apply the shock dissipation scheme
            self._shock_dissipation_phase()

            # the end of the loop of one local time step
            self.first_step = False

            # Calculation is terminated if global dt is not specified.
            if dt is np.inf:
                break

            # add local dt to the timer "local elapsed time"
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
        self.cip2d.run(
            self.u,
            self.dudx,
            self.dudy,
            self.u,
            self.v,
            self.wet_pwet_horizontal_links,
            self.horizontal_up_links.astype(np.int64),
            self.vertical_up_links.astype(np.int64),
            self.grid.dx,
            self.dt_local,
            out_f=self.u_temp,
            out_dfdx=self.dudx_temp,
            out_dfdy=self.dudy_temp,
        )

        self.cip2d.run(
            self.v,
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
            out_dfdy=self.dvdy_temp,
        )

        self.cip2d.run(
            self.h,
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
            out_dfdy=self.dhdy_temp,
        )

        for i in range(self.number_gclass):
            self.cip2d.run(
                self.Ch_i[i, :],
                self.dChdx_i[i, :],
                self.dChdy_i[i, :],
                self.u_node,
                self.v_node,
                self.wet_pwet_nodes,
                self.horizontal_up_nodes,
                self.vertical_up_nodes,
                self.grid.dx,
                self.dt_local,
                out_f=self.Ch_i_temp[i, :],
                out_dfdx=self.dChdx_i_temp[i, :],
                out_dfdy=self.dChdy_i_temp[i, :],
            )

        if self.model == "4eq":
            self.cip2d.run(
                self.Kh,
                self.dKhdx,
                self.dKhdy,
                self.u,
                self.v,
                self.wet_pwet_links,
                self.horizontal_up_links,
                self.vertical_up_links,
                self.grid.dx,
                self.dt_local,
                out_f=self.Kh_temp,
                out_dfdx=self.dKhdx_temp,
                out_dfdy=self.dKhdy_temp,
            )

        # remove abnormal values
        self._remove_abnormal_values()

        # update gradient terms
        self.update_gradients2()

        # update values after calculating advection terms
        # map node values to links, and link values to nodes.
        self.update_values()
        map_values(
            self,
            h=self.h,
            dhdx=self.dhdx,
            dhdy=self.dhdy,
            u=self.u,
            dudx=self.dudx,
            dudy=self.dudy,
            v=self.v,
            dvdx=self.dvdx,
            dvdy=self.dvdy,
            Ch_i=self.Ch_i,
            dChdx_i=self.dChdx_i,
            dChdy_i=self.dChdy_i,
            Ch=self.Ch,
            eta=self.eta,
            h_link=self.h_link,
            u_node=self.u_node,
            v_node=self.v_node,
            Ch_link=self.Ch_link,
            U=self.U,
            U_node=self.U_node,
        )
        update_up_down_links_and_nodes(self)

    def _nonadvection_phase(self):
        """Calculate non-advection phase of the model
        Pressure terms for velocities and mass conservation equations
        are solved implicitly by CCUP method
        """

        # Solve pressure term and associated velocity change by CCUP method
        self._CCUP()

        # Solve gravity terms
        self._gravity()

        # Solve artificial viscosity for momentum
        self._artificial_viscosity(
            self.h_temp,
            self.h_link_temp,
            self.u_temp,
            self.v_temp,
            self.Ch_temp,
            self.Ch_link_temp,
        )
        self.update_boundary_conditions(
            u=self.u_temp,
            v=self.v_temp,
            u_node=self.u_node_temp,
            v_node=self.v_node_temp,
        )

        # Solve friction terms
        self._friction()

        # Solve mass conservation terms
        self._fluid_mass_conservation()

        # Solve diffusion terms for momentum
        self._momentum_diffusion()

        # for 4 equation model
        if self.model == "4eq":
            self._calculate_turbulent_kinetic_energy()
            self._TKE_diffusion()
            self._update_friction_coefficient(self.U_temp, self.Kh_temp)

        # Solve deposition/erosion
        if self.suspension:
            self.calc_deposition(
                self.h_temp,
                self.Ch_i_temp,
                self.u_node_temp,
                self.v_node_temp,
                self.eta_temp,
                self.bed_thick_i,
                self.U_node_temp,
                out_Ch_i=self.Ch_i_temp,
                out_eta=self.eta_temp,
                out_bed_thick_i=self.bed_thick_i_temp,
            )
            self.S[self.active_links] = self.grid.calc_grad_at_link(self.eta_temp)[
                self.active_links
            ]

        # update gradient terms
        self.update_gradients()

        # update values
        self.update_values()
        map_values(
            self,
            h=self.h,
            dhdx=self.dhdx,
            dhdy=self.dhdy,
            u=self.u,
            dudx=self.dudx,
            dudy=self.dudy,
            v=self.v,
            dvdx=self.dvdx,
            dvdy=self.dvdy,
            Ch_i=self.Ch_i,
            dChdx_i=self.dChdx_i,
            dChdy_i=self.dChdy_i,
            Ch=self.Ch,
            eta=self.eta,
            h_link=self.h_link,
            u_node=self.u_node,
            v_node=self.v_node,
            Ch_link=self.Ch_link,
            U=self.U,
            U_node=self.U_node,
        )
        update_up_down_links_and_nodes(self)

    def _gravity(self):
        """solve the gravity term"""
        Rg = self.R * self.g

        # calculate gravity force
        self.u_temp[self.wet_horizontal_links] -= (
            Rg
            * self.Ch_link[self.wet_horizontal_links]
            / self.h_link[self.wet_horizontal_links]
            * self.S[self.wet_horizontal_links]
            * self.dt_local
        )
        self.v_temp[self.wet_vertical_links] -= (
            Rg
            * self.Ch_link[self.wet_vertical_links]
            / self.h_link[self.wet_vertical_links]
            * self.S[self.wet_vertical_links]
            * self.dt_local
        )
        self.update_boundary_conditions(
            u=self.u_temp,
            v=self.v_temp,
            u_node=self.u_node_temp,
            v_node=self.v_node_temp,
        )

    def _friction(self):
        """solve the friction term with semi-implicit scheme"""

        # calculate water entrainment coefficients
        if self.water_entrainment:
            self.ew_link[self.wet_horizontal_links] = get_ew(
                self.U[self.wet_horizontal_links],
                self.Ch_link[self.wet_horizontal_links],
                self.R,
                self.g,
            )
            self.ew_link[self.wet_vertical_links] = get_ew(
                self.U[self.wet_vertical_links],
                self.Ch_link[self.wet_vertical_links],
                self.R,
                self.g,
            )
            self.ew_node[self.wet_nodes] = get_ew(
                self.U_node[self.wet_nodes], self.Ch[self.wet_nodes], self.R, self.g
            )
        else:
            self.ew_link[self.wet_horizontal_links] = 0
            self.ew_link[self.wet_vertical_links] = 0
            self.ew_node[self.wet_nodes] = 0

        if self.water_detrainment:
            det_rate_u = get_det_rate(
                self.ws,
                self.Ch_link_i[:, self.wet_horizontal_links],
                self.h_link[self.wet_horizontal_links],
                det_factor=self.det_factor,
            )
            det_rate_v = get_det_rate(
                self.ws,
                self.Ch_link_i[:, self.wet_vertical_links],
                self.h_link[self.wet_vertical_links],
                det_factor=self.det_factor,
            )
        else:
            det_rate_u = 0.0
            det_rate_v = 0.0

        # calculate friction terms using semi-implicit scheme
        self.u_temp[self.wet_horizontal_links] /= (
            1
            + (
                (
                    self.Cf_link[self.wet_horizontal_links]
                    + self.ew_link[self.wet_horizontal_links]
                )
                * self.U[self.wet_horizontal_links]
                - det_rate_u
            )
            * self.dt_local
            / self.h_link[self.wet_horizontal_links]
        )
        self.v_temp[self.wet_vertical_links] /= (
            1
            + (
                (
                    self.Cf_link[self.wet_vertical_links]
                    + self.ew_link[self.wet_vertical_links]
                )
                * self.U[self.wet_vertical_links]
                - det_rate_v
            )
            * self.dt_local
            / self.h_link[self.wet_vertical_links]
        )

        self.update_boundary_conditions(
            u=self.u_temp,
            v=self.v_temp,
            u_node=self.u_node_temp,
            v_node=self.v_node_temp,
        )

    def _fluid_mass_conservation(self):
        """solve fluid mass conservation"""
        dx = self.grid.dx
        dy = self.grid.dy
        dt = self.dt_local

        wet_pwet_nodes = self.wet_pwet_nodes
        east_link_at_node = self.east_link_at_node
        west_link_at_node = self.west_link_at_node
        north_link_at_node = self.north_link_at_node
        south_link_at_node = self.south_link_at_node

        # mass conservation
        self.div[wet_pwet_nodes] = (
            self.u_temp[east_link_at_node[wet_pwet_nodes]]
            - self.u_temp[west_link_at_node[wet_pwet_nodes]]
        ) / (dx) + (
            self.v_temp[north_link_at_node[wet_pwet_nodes]]
            - self.v_temp[south_link_at_node[wet_pwet_nodes]]
        ) / (
            dy
        )

        # remove negative values
        self._remove_abnormal_values()

        self.h_temp[self.wet_pwet_nodes] /= 1 + self.div[wet_pwet_nodes] * dt
        self.Ch_i_temp[:, self.wet_pwet_nodes] /= 1 + self.div[wet_pwet_nodes] * dt

        # calculate flow expansion by water entrainment
        if self.water_entrainment is True:
            self.h_temp[self.wet_nodes] += (
                self.ew_node[self.wet_nodes]
                * self.U_node[self.wet_nodes]
                * self.dt_local
            )

        # calculate flow contraction by water detrainment
        if self.water_detrainment is True:
            self.h_temp[self.wet_nodes] -= (
                get_det_rate(
                    self.ws,
                    self.Ch_i[:, self.wet_nodes],
                    self.h[self.wet_nodes],
                    self.det_factor,
                )
            ) * self.dt_local

        # map nodes to links
        map_nodes_to_links(
            self,
            h=self.h_temp,
            dhdx=self.dhdx_temp,
            dhdy=self.dhdy_temp,
            Ch_i=self.Ch_i_temp,
            dChdx_i=self.dChdx_i_temp,
            dChdy_i=self.dChdy_i_temp,
            Ch=self.Ch_temp,
            eta=self.eta_temp,
            h_link=self.h_link_temp,
            Ch_link_i=self.Ch_link_i_temp,
            Ch_link=self.Ch_link_temp,
        )

        # update boundary conditions
        self.update_boundary_conditions(
            h=self.h_temp,
            Ch=self.Ch_temp,
            h_link=self.h_link_temp,
            Ch_link=self.Ch_link_temp,
            eta=self.eta_temp,
        )

    def _momentum_diffusion(self):
        """solve diffusion terms of momentum"""

        # copy grid ids and other variables
        wet_pwet_h_links = self.wet_pwet_horizontal_links
        wet_pwet_v_links = self.wet_pwet_vertical_links
        link_east = self.link_east
        link_west = self.link_west
        link_north = self.link_north
        link_south = self.link_south

        dx = self.grid.dx
        dy = self.grid.dy
        dt = self.dt_local
        dx2 = dx * dx
        dy2 = dy * dy

        # diffusion of momentum
        self.calc_nu_t(self.u_temp, self.v_temp, self.h_link_temp, out=self.nu_t)
        self.u_temp[wet_pwet_h_links] += (
            self.nu_t[wet_pwet_h_links]
            * dt
            * (
                (
                    self.u_temp[link_east[wet_pwet_h_links]]
                    - 2 * self.u_temp[wet_pwet_h_links]
                    + self.u_temp[link_west[wet_pwet_h_links]]
                )
                + (
                    self.u_temp[link_north[wet_pwet_h_links]]
                    - 2 * self.u_temp[wet_pwet_h_links]
                    + self.u_temp[link_south[wet_pwet_h_links]]
                )
            )
            / dx2
        )
        self.v_temp[wet_pwet_v_links] += (
            self.nu_t[wet_pwet_v_links]
            * dt
            * (
                (
                    self.v_temp[link_east[wet_pwet_v_links]]
                    - 2 * self.v_temp[wet_pwet_v_links]
                    + self.v_temp[link_west[wet_pwet_v_links]]
                )
                + (
                    self.v_temp[link_north[wet_pwet_v_links]]
                    - 2 * self.v_temp[wet_pwet_v_links]
                    + self.v_temp[link_south[wet_pwet_v_links]]
                )
            )
            / dy2
        )
        self.update_boundary_conditions(
            u=self.u_temp,
            v=self.v_temp,
            u_node=self.u_node_temp,
            v_node=self.v_node_temp,
        )

        # map values
        map_links_to_nodes(
            self,
            u=self.u_temp,
            dudx=self.dudx_temp,
            dudy=self.dudy_temp,
            v=self.v_temp,
            dvdx=self.dvdx_temp,
            dvdy=self.dvdy_temp,
            u_node=self.u_node_temp,
            v_node=self.v_node_temp,
            U=self.U_temp,
            U_node=self.U_node_temp,
        )
        self.update_boundary_conditions(
            u_node=self.u_node_temp,
            v_node=self.v_node_temp,
        )

    def _TKE_diffusion(self):
        """solve diffusion terms of TKE"""

        # copy grid ids and other variables
        wet_pwet_h_links = self.wet_pwet_horizontal_links
        wet_pwet_v_links = self.wet_pwet_vertical_links
        link_east = self.link_east
        link_west = self.link_west
        link_north = self.link_north
        link_south = self.link_south

        dx = self.grid.dx
        dy = self.grid.dy
        dt = self.dt_local
        dx2 = dx * dx
        dy2 = dy * dy

        # diffusion of TKE
        self.calc_nu_t(self.u_temp, self.v_temp, self.h_link_temp, out=self.nu_t)
        self.Kh_temp[wet_pwet_h_links] += (
            self.nu_t[wet_pwet_h_links]
            * dt
            * (
                (
                    self.Kh_temp[link_east[wet_pwet_h_links]]
                    - 2 * self.Kh_temp[wet_pwet_h_links]
                    + self.Kh_temp[link_west[wet_pwet_h_links]]
                )
                + (
                    self.Kh_temp[link_north[wet_pwet_h_links]]
                    - 2 * self.Kh_temp[wet_pwet_h_links]
                    + self.Kh_temp[link_south[wet_pwet_h_links]]
                )
            )
            / dx2
        )
        self.Kh_temp[wet_pwet_v_links] += (
            self.nu_t[wet_pwet_v_links]
            * dt
            * (
                (
                    self.Kh_temp[link_east[wet_pwet_v_links]]
                    - 2 * self.Kh_temp[wet_pwet_v_links]
                    + self.Kh_temp[link_west[wet_pwet_v_links]]
                )
                + (
                    self.Kh_temp[link_north[wet_pwet_v_links]]
                    - 2 * self.Kh_temp[wet_pwet_v_links]
                    + self.Kh_temp[link_south[wet_pwet_v_links]]
                )
            )
            / dy2
        )
        self.update_boundary_conditions(
            Kh=self.Kh_temp
        )

        # map values
        map_links_to_nodes(
            self,
            Kh=self.Kh_temp,
        )


    def _CCUP(self):
        """solve the pressure term by CCUP method"""

        self.copy_values_to_temp()
        map_nodes_to_links(
            self,
            h=self.h_temp,
            dhdx=self.dhdx_temp,
            dhdy=self.dhdy_temp,
            Ch_i=self.Ch_i_temp,
            dChdx_i=self.dChdx_i_temp,
            dChdy_i=self.dChdy_i_temp,
            Ch=self.Ch_temp,
            eta=self.eta_temp,
            h_link=self.h_link_temp,
            Ch_link_i=self.Ch_link_i_temp,
            Ch_link=self.Ch_link_temp,
        )

        # copy grid ids and other variables
        wet_nodes = self.wet_nodes
        wet_pwet_nodes = self.wet_pwet_nodes
        east_link_at_node = self.east_link_at_node
        west_link_at_node = self.west_link_at_node
        north_link_at_node = self.north_link_at_node
        south_link_at_node = self.south_link_at_node

        h = self.h
        h_link = self.h_link
        u = self.u
        v = self.v
        Ch = self.Ch
        Ch_link = self.Ch_link

        dx = self.grid.dx
        dy = self.grid.dy
        dt = self.dt_local
        Rg = self.R * self.g
        dx2 = dx * dx
        dy2 = dy * dy
        dt2 = dt * dt

        # calculate pressure
        self.p[self.dry_nodes] = 0
        self.p[wet_pwet_nodes] = h[wet_pwet_nodes] * Ch[wet_pwet_nodes]
        self.update_boundary_conditions(p=self.p)
        self.p_temp[:] = self.p[:]

        # set coefficients to SOR solver
        self.sor.a[wet_nodes] = (
            -Rg
            * (
                1 / (2.0 * h_link[east_link_at_node[wet_nodes]])
                + 1 / (2.0 * h_link[west_link_at_node[wet_nodes]])
            )
            / dx2
            - Rg
            * (
                1 / (2.0 * h_link[north_link_at_node[wet_nodes]])
                + 1 / (2.0 * h_link[south_link_at_node[wet_nodes]])
            )
            / dy2
            - 1 / (2 * self.p[wet_nodes] * dt2)
        )
        self.sor.b[wet_nodes] = Rg / (dx2 * 2.0 * h_link[east_link_at_node[wet_nodes]])
        self.sor.c[wet_nodes] = Rg / (dx2 * 2.0 * h_link[west_link_at_node[wet_nodes]])
        self.sor.d[wet_nodes] = Rg / (dx2 * 2.0 * h_link[north_link_at_node[wet_nodes]])
        self.sor.e[wet_nodes] = Rg / (dx2 * 2.0 * h_link[south_link_at_node[wet_nodes]])
        self.sor.g[wet_nodes] = (
            -1 / (2 * dt2)
            + (
                (u[east_link_at_node[wet_nodes]] - u[west_link_at_node[wet_nodes]]) / dx
                + (v[north_link_at_node[wet_nodes]] - v[south_link_at_node[wet_nodes]])
                / dy
            )
            / dt
            - Rg
            * (
                (
                    Ch_link[east_link_at_node[wet_nodes]]
                    / h_link[east_link_at_node[wet_nodes]]
                    * self.S[east_link_at_node[wet_nodes]]
                    - Ch_link[west_link_at_node[wet_nodes]]
                    / h_link[west_link_at_node[wet_nodes]]
                    * self.S[west_link_at_node[wet_nodes]]
                )
                / dx2
                + (
                    Ch_link[north_link_at_node[wet_nodes]]
                    / h_link[north_link_at_node[wet_nodes]]
                    * self.S[north_link_at_node[wet_nodes]]
                    - Ch_link[south_link_at_node[wet_nodes]]
                    / h_link[south_link_at_node[wet_nodes]]
                    * self.S[south_link_at_node[wet_nodes]]
                )
                / dy2
            )
        )

        # calculate pressure using SOR method
        self.sor.run(self.p, wet_nodes, out=self.p_temp)

        # calculate u, v from pressure
        self.u_temp[self.wet_horizontal_links] -= (
            Rg
            / (2 * self.h_link[self.wet_horizontal_links])
            * (
                self.p_temp[
                    self.east_node_at_horizontal_link[self.wet_horizontal_links]
                ]
                - self.p_temp[
                    self.west_node_at_horizontal_link[self.wet_horizontal_links]
                ]
            )
            / dx
            * dt
        )
        self.v_temp[self.wet_vertical_links] -= (
            Rg
            / (2 * self.h_link[self.wet_vertical_links])
            * (
                self.p_temp[self.north_node_at_vertical_link[self.wet_vertical_links]]
                - self.p_temp[self.south_node_at_vertical_link[self.wet_vertical_links]]
            )
            / dy
            * dt
        )

    def _calculate_turbulent_kinetic_energy(self):
        """Calculate time development of layer-averaged turbulent kinetic energy
        based on Parker et al. (1986)
        """

        map_values(
            self,
            h=self.h_temp,
            h_link=self.h_link_temp,
            u=self.u_temp,
            v=self.v_temp,
            U=self.U_temp,
        )
        self.Kh_temp[self.wet_pwet_links[self.Kh_temp[self.wet_pwet_links] < 0]] = 0.0

        # development of turbulent kinetic energy
        alpha = self.alpha_TKE
        # Ri = self.R * self.g * self.Ch_link[self.wet_pwet_links] \
        #      / self.U[self.wet_pwet_links] / self.U[self.wet_pwet_links]
        # beta = (0.5 * self.ew_link[self.wet_pwet_links] *
        #         (1 - Ri - 2.0 * self.Cf / alpha) + self.Cf) / (self.Cf /
        #                                                        alpha)**1.5

        beta = alpha**1.5 / self.Cf**0.5
        # K = self.Kh[self.wet_pwet_links] / \
        #     self.h_link_temp[self.wet_pwet_links]

        # self.Kh_temp[self.wet_pwet_links] += self.dt_local * (
        #     alpha * K * self.U_temp[self.wet_pwet_links]
        #     + 0.5
        #     * self.ew_link[self.wet_pwet_links]
        #     * self.U_temp[self.wet_pwet_links] ** 3
        #     - beta * K ** 1.5

        # self.Kh_temp[self.wet_pwet_links] += self.dt_local * (
        #     (self.Cf + 0.5 * self.ew_link[self.wet_pwet_links])
        #    * self.U_temp[self.wet_pwet_links]
        #    * self.U_temp[self.wet_pwet_links]
        #    * self.U_temp[self.wet_pwet_links]
        #    - beta
        #    * (self.Kh[self.wet_pwet_links] / self.h_link[self.wet_pwet_links]) ** 1.5
        #    - self.R
        #    * self.g
        #    * (
        #        np.sum(self.Ch_link_i_temp[:, self.wet_pwet_links] * self.ws, axis=0)
        #        + 0.5
        #        * self.U_temp[self.wet_pwet_links]
        #        * self.ew_link[self.wet_pwet_links]
        #        * self.Ch_link_temp[self.wet_pwet_links]
        #    )
        # )

        self.Kh_temp[self.wet_pwet_links] += self.dt_local * (
            (self.Cf + 0.5 * self.ew_link[self.wet_pwet_links])
            * self.U_temp[self.wet_pwet_links]
            * self.U_temp[self.wet_pwet_links]
            * self.U_temp[self.wet_pwet_links]
            - beta
            * (self.Kh[self.wet_pwet_links] / self.h_link[self.wet_pwet_links]) ** 1.5
            - self.R
            * self.g
            * (
                np.sum(self.Ch_link_i_temp[:, self.wet_pwet_links] * self.ws, axis=0)
                + 0.5
                * self.U_temp[self.wet_pwet_links]
                * self.ew_link[self.wet_pwet_links]
                * self.Ch_link_temp[self.wet_pwet_links]
            )
        )

        self.Kh_temp[self.partial_wet_horizontal_links] = 0.0
        self.Kh_temp[self.partial_wet_vertical_links] = 0.0
        
        # remove negative values
        self.Kh_temp[self.wet_pwet_links] = np.where(
            self.Kh_temp[self.wet_pwet_links] < 0.0,
            0.0,
            self.Kh_temp[self.wet_pwet_links]
        )

        # adjust_negative_values(self.Kh,
        #                        self.wet_pwet_links,
        #                        self.link_east,
        #                        self.link_west,
        #                        self.link_north,
        #                        self.link_south,
        #                        out_f=self.Kh_temp)

    def _update_friction_coefficient(self, U, Kh):
        
        # update friction coefficient Cf_link and Cf_nodes
        U_exist = U[self.wet_pwet_links] > 1.0e-10
        self.Cf_link[self.wet_pwet_links[U_exist]] = (
            self.alpha_TKE
            * Kh[self.wet_pwet_links[U_exist]]
            / (self.h_link[self.wet_pwet_links[U_exist]] + 1.0e-7)
            / U[self.wet_pwet_links[U_exist]]
            / U[self.wet_pwet_links[U_exist]]
        ) 
        # self.Cf_link[self.Cf_link > 0.1] = 0.1
        self.Cf_link[self.wet_pwet_links[~U_exist]] = self.Cf
        map_values(self, Cf_link=self.Cf_link, Cf_node=self.Cf_node)



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
        east_link = self.east_link_at_node
        west_link = self.west_link_at_node
        north_link = self.north_link_at_node
        south_link = self.south_link_at_node
        east_node = self.east_node_at_horizontal_link
        west_node = self.west_node_at_horizontal_link
        north_node = self.north_node_at_vertical_link
        south_node = self.south_node_at_vertical_link
        div = self.div
        nu_a = self.nu_a  # artificial viscosity coefficient
        Rg = self.R * self.g
        Cs = self.Cs

        # basic parameters for artificial viscosity
        Cs[wet_nodes] = np.sqrt(Rg * Ch[wet_nodes])

        # no directional dependent expression of artificial viscosity
        # calculated only at grids where flow divergence is negative
        div[:] = 0
        div[wet_nodes] = (u[east_link[wet_nodes]] - u[west_link[wet_nodes]]) / dx + (
            v[north_link[wet_nodes]] - v[south_link[wet_nodes]]
        ) / dy
        compress = div[wet_nodes] < 0
        self.q[wet_nodes[~compress]] = 0
        self.q[wet_nodes[compress]] = (
            (2.0 * h[wet_nodes[compress]])
            / Rg
            * nu_a
            * dx
            * (
                -Cs[wet_nodes[compress]] * div[wet_nodes[compress]]
                + 1.5 * div[wet_nodes[compress]] * div[wet_nodes[compress]] * dx
            )
        )

        # modify flow velocity based on artificial viscosity
        u[wet_horiz] -= (
            0.5
            * Rg
            / h_link[wet_horiz]
            * (self.q[east_node[wet_horiz]] - self.q[west_node[wet_horiz]])
            / dx
            * dt
        )
        v[wet_vert] -= (
            0.5
            * Rg
            / h_link[wet_vert]
            * (self.q[north_node[wet_vert]] - self.q[south_node[wet_vert]])
            / dy
            * dt
        )

    def _remove_abnormal_values(self):
        """remove abnormal values from variables that should not be negative"""

        adjust_negative_values(
            self.h_temp,
            self.wet_pwet_nodes,
            self.node_east,
            self.node_west,
            self.node_north,
            self.node_south,
            out_f=self.h_temp,
        )
        for i in range(self.number_gclass):
            adjust_negative_values(
                self.Ch_i_temp[i, :],
                self.wet_pwet_nodes,
                self.node_east,
                self.node_west,
                self.node_north,
                self.node_south,
                out_f=self.Ch_i_temp[i, :],
            )

        # if self.model == "4eq":
        #     adjust_negative_values(
        #         self.Kh_temp,
        #         self.wet_pwet_nodes,
        #         self.node_east,
        #         self.node_west,
        #         self.node_north,
        #         self.node_south,
        #         out_f=self.Kh_temp,
        #     )

    def _process_wet_dry_boundary(self):
        """Calculate processes at wet and dry boundary"""
        process_partial_wet_grids(
            self,
            self.h,
            self.u,
            self.v,
            self.Ch,
            h_out=self.h_temp,
            u_out=self.u_temp,
            v_out=self.v_temp,
            Ch_out=self.Ch_temp,
        )
        update_up_down_links_and_nodes(self)

        # update gradient terms
        self.update_gradients()

        # update values
        self.update_values()
        map_values(
            self,
            h=self.h,
            dhdx=self.dhdx,
            dhdy=self.dhdy,
            u=self.u,
            dudx=self.dudx,
            v=self.v,
            dvdx=self.dvdx,
            dvdy=self.dvdy,
            Ch=self.Ch,
            Ch_i=self.Ch_i,
            dChdx_i=self.dChdx_i,
            dChdy_i=self.dChdy_i,
            eta=self.eta,
            h_link=self.h_link,
            u_node=self.u_node,
            v_node=self.v_node,
            Ch_link=self.Ch_link,
            U=self.U,
            U_node=self.U_node,
        )

    def _shock_dissipation_phase(self):
        """Calculate shock dissipation phase of the model"""
        # update artificial viscosity coefficients for Jameson scheme
        self.jameson.update_artificial_viscosity(
            self.R * self.g * self.Ch * self.h,
            self.R * self.g * self.Ch_link * self.h_link,
        )

        # apply Jameson's filter
        for i in range(self.number_gclass):
            self.jameson.run(
                self.Ch_i[i, :],
                self.wet_pwet_nodes,
                at="node",
                out=self.Ch_i_temp[i, :],
            )
        self.jameson.run(self.h, self.wet_pwet_nodes, at="node", out=self.h_temp)
        # self.jameson.run(
        #    self.u, self.wet_pwet_horizontal_links, at="hlink", out=self.u_temp
        # )
        # self.jameson.run(
        #     self.v, self.wet_pwet_vertical_links, at="vlink", out=self.v_temp
        # )

        # if self.model == "4eq":
        #     self.jameson.run(
        #         self.Kh, self.wet_pwet_vertical_links, at="vlink", out=self.Kh_temp
        #     )
        #     self.jameson.run(
        #         self.Kh, self.wet_pwet_horizontal_links, at="hlink", out=self.Kh_temp
        #     )

        # update gradient terms
        self.update_gradients()

        # Reset our field values with the newest flow depth and
        # discharge.
        self.update_values()
        map_values(
            self,
            h=self.h,
            dhdx=self.dhdx,
            dhdy=self.dhdy,
            Ch=self.Ch,
            Ch_i=self.Ch_i,
            dChdx_i=self.dChdx_i,
            dChdy_i=self.dChdy_i,
            u=self.u,
            dudx=self.dudx,
            dudy=self.dudy,
            u_node=self.u_node,
            v=self.v,
            dvdx=self.dvdx,
            dvdy=self.dvdy,
            v_node=self.v_node,
            U=self.U,
            U_node=self.U_node,
            eta=self.eta,
            h_link=self.h_link,
            Ch_link=self.Ch_link,
        )

    def update_gradients(self):
        """update gradient terms when main variables were changed"""
        update_gradient(
            self.u,
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
            out_dfdy=self.dudy_temp,
        )
        update_gradient(
            self.v,
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
            out_dfdy=self.dvdy_temp,
        )
        update_gradient(
            self.h,
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
            out_dfdy=self.dhdy_temp,
        )
        for i in range(self.number_gclass):
            update_gradient(
                self.Ch_i[i, :],
                self.Ch_i_temp[i, :],
                self.dChdx_i_temp[i, :],
                self.dChdy_i_temp[i, :],
                self.wet_pwet_nodes,
                self.node_north[self.wet_pwet_nodes],
                self.node_south[self.wet_pwet_nodes],
                self.node_east[self.wet_pwet_nodes],
                self.node_west[self.wet_pwet_nodes],
                self.grid.dx,
                self.dt_local,
                out_dfdx=self.dChdx_i_temp[i, :],
                out_dfdy=self.dChdy_i_temp[i, :],
            )
        if self.model == "4eq":
            update_gradient(
                self.Kh,
                self.Kh_temp,
                self.dKhdx_temp,
                self.dKhdy_temp,
                self.wet_pwet_links,
                self.link_north[self.wet_pwet_links],
                self.link_south[self.wet_pwet_links],
                self.link_east[self.wet_pwet_links],
                self.link_west[self.wet_pwet_links],
                self.grid.dx,
                self.dt_local,
                out_dfdx=self.dKhdx_temp,
                out_dfdy=self.dKhdy_temp,
            )

    def update_gradients2(self):
        """update gradient terms for df/dx du/dy"""
        update_gradient2(
            self.u,
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
            out_dfdy=self.dudy_temp,
        )
        update_gradient2(
            self.v,
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
            out_dfdy=self.dvdy_temp,
        )
        update_gradient2(
            self.h,
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
            out_dfdy=self.dhdy_temp,
        )
        for i in range(self.number_gclass):
            update_gradient2(
                self.Ch_i[i, :],
                self.dChdx_i_temp[i, :],
                self.dChdy_i_temp[i, :],
                self.u_node,
                self.v_node,
                self.wet_pwet_nodes,
                self.node_north[self.wet_pwet_nodes],
                self.node_south[self.wet_pwet_nodes],
                self.node_east[self.wet_pwet_nodes],
                self.node_west[self.wet_pwet_nodes],
                self.grid.dx,
                self.dt_local,
                out_dfdx=self.dChdx_i_temp[i, :],
                out_dfdy=self.dChdy_i_temp[i, :],
            )
        if self.model == "4eq":
            update_gradient2(
                self.Kh,
                self.dKhdx_temp,
                self.dKhdy_temp,
                self.u,
                self.v,
                self.wet_pwet_links,
                self.link_north[self.wet_pwet_links],
                self.link_south[self.wet_pwet_links],
                self.link_east[self.wet_pwet_links],
                self.link_west[self.wet_pwet_links],
                self.grid.dx,
                self.dt_local,
                out_dfdx=self.dKhdx_temp,
                out_dfdy=self.dKhdy_temp,
            )

        self.update_values()

    def calc_deposition(
        self,
        h,
        Ch_i,
        u_node,
        v_node,
        eta,
        bed_thick_i,
        U_node,
        out_Ch_i=None,
        out_eta=None,
        out_bed_thick_i=None,
    ):
        """Calculate deposition/erosion processes

        Parameters
        ----------------
        h
        Ch_i
        u_node
        v_node
        eta
        bed_thick_i
        out_Ch_i
        out_eta
        """
        if out_Ch_i is None:
            out_Ch_i = np.empty_like(Ch_i)
        if out_eta is None:
            out_eta = np.empty_like(eta)
        if out_bed_thick_i is None:
            out_bed_thick_i = np.empty_like(bed_thick_i)

        nodes = self.wet_nodes
        dt = self.dt_local
        ws = self.ws
        r0 = self.r0
        node_east = self.node_east[nodes]
        node_west = self.node_west[nodes]
        node_north = self.node_north[nodes]
        node_south = self.node_south[nodes]
        dx = self.grid.dx

        # copy previous values of Ch
        self.Ch_i_prev[:, nodes] = Ch_i[:, nodes]

        # Calculate shear velocity
        u_star = np.sqrt(self.Cf_node[nodes] * U_node[nodes] * U_node[nodes])

        # Calculate entrainment rate
        self.es[:, nodes] = get_es(
            self.R,
            self.g,
            self.Ds,
            self.nu,
            u_star,
            function=self.sed_entrainment_func,
        )

        # Calculate the change of volume of suspended sediment
        # Settling is solved explicitly, and entrainment is
        # solved semi-implicitly
        out_Ch_i[:, nodes] = (
            Ch_i[:, nodes]
            + ws * self.bed_active_layer[:, nodes] * self.es[:, nodes] * dt
        )
        out_Ch_i[:, nodes] /= 1 + ws * r0 / (h[nodes] + 1.0e-7) * dt

        # Obtain sedimentation rate
        self.bed_change_i[:, nodes] = self.Ch_i_prev[:, nodes] - out_Ch_i[:, nodes]

        # if erosion is forbidden, out_Ch_i is modified
        if self.no_erosion is True:
            eroded_region = np.sum(self.bed_change_i[:, nodes], axis=0) < 0.0
            out_Ch_i[:, nodes[eroded_region]] = self.Ch_i_prev[:, nodes[eroded_region]]
            self.bed_change_i[:, nodes[eroded_region]] = 0.0


        # bedload transport
        if self.bedload_transport == True:
           self.bedload_total[:, nodes] = get_bedload(
               u_star,
               self.Ds,
               R=self.R,
               g=self.g,
               function="MPM"
           )
           self.bedload_total[:, nodes] *= self.bed_active_layer[:, nodes]
           self.bedload_x[:, nodes] = (
               u_node[nodes] / (U_node[nodes] + 1.0e-7) * self.bedload_total[:, nodes]
           )
           self.bedload_y[:, nodes] = (
               v_node[nodes] / (U_node[nodes] + 1.0e-7) * self.bedload_total[:, nodes]
           )
           self.bed_change_i[:, nodes] += dt / dx / 2.0 * (
               self.bedload_x[:, node_east] - self.bedload_x[:, node_west]
               + self.bedload_y[:, node_north] - self.bedload_y[:, node_south])
            

        # Apply diffusion to avoid slope steeper than angle of repose
        self._bed_diffusion_at_high_slope()

        # Time development of active layer
        self.bed_active_layer[:, nodes] += 1.0 / self.la * (
            self.bed_change_i[:, nodes]
            - np.sum(self.bed_change_i[:, nodes], axis=0)
        )
        # remove negative values
        self.bed_active_layer[:, nodes] = np.where(
            self.bed_active_layer[:, nodes] < 0.0,
            1.0e-7,
            self.bed_active_layer[:, nodes]
        ) 
        # Adjust abnormal values in the active layer
        self.bed_active_layer[:, nodes] /= np.sum(
            self.bed_active_layer[:, nodes], axis=0
        )

        # Calculate deposition
        out_bed_thick_i[:, nodes] = bed_thick_i[:, nodes] + self.bed_change_i[:, nodes]

        # Calculate bed elevation change
        out_eta[nodes] = (
            eta[nodes] + np.sum(self.bed_change_i[:, nodes], axis=0) / self.lambda_p
        )

        return out_Ch_i, out_eta, out_bed_thick_i

    def _bed_diffusion_at_high_slope(self):
        """diffusion sediment transport in the region where slope is close to angle
        of repose
        """
        diffusion_coeff = 1.0e-3
        high_slope = 0.6

        # for numerical stabililty
        stable_diffusion_coeff = self.grid.dx ** 2 / 4.0 / self.dt_local * 0.1
        if diffusion_coeff > stable_diffusion_coeff:
            diffusion_coeff = stable_diffusion_coeff
        


        high_slope_horizontal_links = (
            np.abs(self.S[self.wet_pwet_horizontal_links]) > high_slope
        )
        east_node = self.east_node_at_horizontal_link[
            self.wet_pwet_horizontal_links[high_slope_horizontal_links]
        ]
        west_node = self.west_node_at_horizontal_link[
            self.wet_pwet_horizontal_links[high_slope_horizontal_links]
        ]
        horiz_change = (
            diffusion_coeff
            * self.dt_local
            / self.grid.dx
            * self.S[self.wet_pwet_horizontal_links[high_slope_horizontal_links]]
        ) 
        self.bed_change_i[:, east_node] -= (
            horiz_change * self.bed_active_layer[:, east_node]
        )
        self.bed_change_i[:, west_node] += (
            horiz_change * self.bed_active_layer[:, west_node]
        )
        
        high_slope_vertical_links = (
            np.abs(self.S[self.wet_pwet_vertical_links]) > high_slope
        )
        north_node = self.north_node_at_vertical_link[
            self.wet_pwet_vertical_links[high_slope_vertical_links]
        ]
        south_node = self.south_node_at_vertical_link[
            self.wet_pwet_vertical_links[high_slope_vertical_links]
        ]
        vert_change = (
            diffusion_coeff
            * self.dt_local
            / self.grid.dx
            * self.S[self.wet_pwet_vertical_links[high_slope_vertical_links]]
            )
        self.bed_change_i[:, north_node] -= (
            vert_change * self.bed_active_layer[:, north_node]
            )
        self.bed_change_i[:, south_node] += (
            vert_change * self.bed_active_layer[:, south_node]
            )

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
        self.Ch_i_temp[:, :] = self.Ch_i[:, :]
        self.dChdx_i_temp[:, :] = self.dChdx_i[:, :]
        self.dChdy_i_temp[:, :] = self.dChdy_i[:, :]
        self.Ch = np.sum(self.Ch_i_temp, axis=0)
        self.Ch_temp[:] = self.Ch[:]
        self.Ch_link_i_temp[:, :] = self.Ch_link_i[:, :]
        self.Ch_link = np.sum(self.Ch_link_i, axis=0)
        self.Ch_link_temp[:] = self.Ch_link[:]
        self.eta_temp[:] = self.eta[:]
        self.bed_thick_i_temp[:, :] = self.bed_thick_i[:, :]
        self.U_temp[:] = self.U[:]
        self.U_node_temp[:] = self.U_node[:]
        if self.model == "4eq":
            self.Kh_temp = self.Kh[:]
            self.dKhdx_temp = self.dKhdx[:]
            self.dKhdy_temp = self.dKhdy[:]

    def copy_values_to_grid(self):
        """Copy flow parameters to grid"""
        self.grid.at_node["flow__depth"] = self.h
        self.grid.at_link["flow__horizontal_velocity"] = self.u
        self.grid.at_link["flow__vertical_velocity"] = self.v
        self.C_i[:, :] = self.C_init
        all_wet_nodes = np.where(self.h > self.h_w)
        self.C_i[:, all_wet_nodes] = self.Ch_i[:, all_wet_nodes] / self.h[all_wet_nodes]
        for i in range(self.number_gclass):
            self.grid.at_node["flow__sediment_concentration_" + str(i)] = self.C_i[i, :]
            self.grid.at_node["flow_sediment_volume__horizontal_gradient_" + str(i)] = (
                self.dChdx_i[i, :]
            )
            self.grid.at_node["flow_sediment_volume__vertical_gradient_" + str(i)] = (
                self.dChdy_i[i, :]
            )
            self.grid.at_node["bed__sediment_volume_per_unit_area_" + str(i)] = (
                self.bed_thick_i[i, :]
            )
            self.grid.at_node["bed__active_layer_fraction_" + str(i)] = (
                self.bed_active_layer[i, :]
            )
        self.C[:] = np.sum(self.C_i, axis=0)
        self.grid.at_node["flow__sediment_concentration_total"] = self.C
        self.grid.at_node["topographic__elevation"] = self.eta
        self.grid.at_node["bed__thickness"] = self.bed_thick
        self.grid.at_node["flow__surface_elevation"] = self.eta + self.h
        self.grid.at_node["flow__horizontal_velocity_at_node"] = self.u_node
        self.grid.at_node["flow__vertical_velocity_at_node"] = self.v_node
        self.grid.at_link["flow_horizontal_velocity__horizontal_gradient"] = self.dudx
        self.grid.at_link["flow_horizontal_velocity__vertical_gradient"] = self.dudy
        self.grid.at_link["flow_vertical_velocity__horizontal_gradient"] = self.dvdx
        self.grid.at_link["flow_vertical_velocity__vertical_gradient"] = self.dvdy
        self.grid.at_node["flow_depth__horizontal_gradient"] = self.dhdx
        self.grid.at_node["flow_depth__vertical_gradient"] = self.dhdy
        if self.model == "4eq":
            self.grid.at_link["flow__TKE"] = self.Kh
            self.grid.at_link["flow_TKE__horizontal_gradient"] = self.dKhdx
            self.grid.at_link["flow_TKE__vertical_gradient"] = self.dKhdy

    def update_values(self):
        """Update variables from temporally variables and
        apply boundary conditions
        """

        # adjust abnormal values
        self._remove_abnormal_values()

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
        self.Ch_i[:, :] = self.Ch_i_temp[:, :]
        self.dChdx_i[:, :] = self.dChdx_i_temp[:, :]
        self.dChdy_i[:, :] = self.dChdy_i_temp[:, :]
        self.Ch_temp[:] = np.sum(self.Ch_i, axis=0)
        self.Ch[:] = self.Ch_temp[:]
        self.bed_thick_i[:, :] = self.bed_thick_i_temp[:, :]
        self.eta[:] = self.eta_temp[:]
        self.U[:] = self.U_temp[:]
        self.U_node[:] = self.U_node_temp[:]
        if self.model == "4eq":
            self.Kh[:] = self.Kh_temp[:]
            self.dKhdx[:] = self.dKhdx_temp[:]
            self.dKhdy[:] = self.dKhdy_temp[:]

        # update boundary conditions
        self.update_boundary_conditions(
            h=self.h,
            u=self.u,
            v=self.v,
            Ch_i=self.Ch_i,
            h_link=self.h_link,
            Ch_link_i=self.Ch_link_i,
            Ch=self.Ch,
            Ch_link=self.Ch_link,
            u_node=self.u_node,
            v_node=self.v_node,
            eta=self.eta,
            Kh=self.Kh,
        )

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
            out = np.zeros_like(u)

        karman = self.karman

        out = 1 / 6.0 * karman * np.sqrt(u * u + v * v) * h_link

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

    def save_nc(self, filename, time=None, format="NETCDF3_64BIT"):
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
        variable_names = [
            "topographic__elevation",
            "flow__depth",
            "flow__horizontal_velocity_at_node",
            "flow__vertical_velocity_at_node",
            "flow__surface_elevation",
            "flow__sediment_concentration_total",
            "bed__thickness",
        ]
        # add sediment concentration of each grain size class
        variable_names.extend(
            [
                "flow__sediment_concentration_{}".format(i)
                for i in range(self.number_gclass)
            ]
        )

        # add sediment volume per unit area of each grain
        #  size class
        variable_names.extend(
            [
                "bed__sediment_volume_per_unit_area_{}".format(i)
                for i in range(self.number_gclass)
            ]
        )

        # add grain size distribution in active layer
        variable_names.extend(
            [
                "bed__active_layer_fraction_{}".format(i)
                for i in range(self.number_gclass)
            ]
        )

        write_netcdf(
            filename,
            self.grid,
            names=variable_names,
            at="node",
            raster=True,
            time=time,
            format=format,
        )
        # write_raster_netcdf(filename, self.grid, time=time, names=variable_names, at="node", format=format)

    def update_boundary_conditions(
        self,
        h=None,
        u=None,
        v=None,
        Ch_i=None,
        Ch=None,
        h_link=None,
        Ch_link_i=None,
        Ch_link=None,
        u_node=None,
        v_node=None,
        eta=None,
        p=None,
        Kh=None,
    ):
        """Update boundary conditions"""
        edge_nodes = self.grid.nodes_at_link[self.fixed_value_edge_links]

        if h is not None:
            h[self.fixed_grad_nodes] = h[self.fixed_grad_anchor_nodes]

        if Ch is not None:
            Ch[self.fixed_grad_nodes] = Ch[self.fixed_grad_anchor_nodes]

        if Ch_i is not None:
            Ch_i[:, self.fixed_grad_nodes] = Ch_i[:, self.fixed_grad_anchor_nodes]

        if u is not None:
            u[self.fixed_grad_links] = u[self.fixed_grad_anchor_links]
            u[self.fixed_grad_link_at_east[u[self.fixed_grad_link_at_east] < 0]] = 0
            u[self.fixed_grad_link_at_west[u[self.fixed_grad_link_at_west] > 0]] = 0
            u[self.fixed_value_links] = (2.0 / 3.0) * u_node[self.fixed_value_nodes] + (
                1.0 / 3.0
            ) * u[self.fixed_value_anchor_links]
            u[self.fixed_value_edge_links] = np.mean(u_node[edge_nodes], axis=1)

        if v is not None:
            v[self.fixed_grad_links] = v[self.fixed_grad_anchor_links]
            v[self.fixed_grad_link_at_north[v[self.fixed_grad_link_at_north] < 0]] = 0
            v[self.fixed_grad_link_at_south[v[self.fixed_grad_link_at_south] > 0]] = 0
            v[self.fixed_value_links] = (2.0 / 3.0) * v_node[self.fixed_value_nodes] + (
                1.0 / 3.0
            ) * v[self.fixed_value_anchor_links]
            v[self.fixed_value_edge_links] = np.mean(v_node[edge_nodes], axis=1)

        if Kh is not None:
            Kh[self.fixed_grad_links] = Kh[self.fixed_grad_anchor_links]
            Kh[self.fixed_value_links] = Kh[self.fixed_value_anchor_links]

        if h_link is not None:
            h_link[self.fixed_grad_links] = h_link[self.fixed_grad_anchor_links]
            h_link[self.fixed_value_links] = (
                h[self.fixed_value_nodes] + h[self.fixed_value_anchor_nodes]
            ) / 2.0
            h_link[self.fixed_value_edge_links] = np.mean(h[edge_nodes], axis=1)

        if Ch_link is not None:
            Ch_link[self.fixed_grad_links] = Ch_link[self.fixed_grad_anchor_links]
            Ch_link[self.fixed_value_links] = (
                Ch[self.fixed_value_nodes] + Ch[self.fixed_value_anchor_nodes]
            ) / 2.0
            Ch_link[self.fixed_value_edge_links] = np.mean(Ch[edge_nodes], axis=1)

        if Ch_link_i is not None:
            Ch_link_i[:, self.fixed_grad_links] = Ch_link_i[
                :, self.fixed_grad_anchor_links
            ]
            Ch_link_i[:, self.fixed_value_links] = (
                Ch_i[:, self.fixed_value_nodes] + Ch_i[:, self.fixed_value_anchor_nodes]
            ) / 2.0
            Ch_link_i[:, self.fixed_value_edge_links] = np.mean(
                Ch_i[:, edge_nodes], axis=2
            )

        if u_node is not None:
            u_node[self.fixed_grad_nodes] = u_node[self.fixed_grad_anchor_nodes]

        if v_node is not None:
            v_node[self.fixed_grad_nodes] = v_node[self.fixed_grad_anchor_nodes]

        if p is not None:
            p[self.fixed_grad_nodes] = p[self.fixed_grad_anchor_nodes]

        if eta is not None:
            eta[self.fixed_grad_nodes] = eta[self.fixed_grad_anchor_nodes] + (
                self.eta_init[self.fixed_grad_nodes]
                - self.eta_init[self.fixed_grad_anchor_nodes]
            )
            eta[self.fixed_value_nodes] = eta[self.fixed_value_anchor_nodes] + (
                self.eta_init[self.fixed_value_nodes]
                - self.eta_init[self.fixed_value_anchor_nodes]
            )


def run(
    geotiff_filename=None,
    xlim=None,
    ylim=None,
    filter_size=None,
    grid_spacing=10,
    initial_flow_concentration=[0.005, 0.005],
    initial_flow_thickness=100,
    initial_region_radius=100,
    initial_region_center=[1000, 4000],
    dt=50,
    number_of_steps=50,
    model="3eq",
    progress=True,
):
    """
    Run the model of 2DH turbidity current

    Parameters
    ---------------------------
    geotiff_filename: string, optional
        Name of a file in geotiff format. If this parameter is specified,
        digital elevation model (DEM) is loaded from the file, and simulation is
         conducted on the topography.

    xlim: [integer, integer], optional
        Range of X axis used in calculation [xmin, xmax]. If this parameter is
         specified, the numerical calculations are performed in the rang of
         x grids specified by this parameter in the DEM. If this parameter
         is not specified, whole range of DEM is used.

    ylim: [integer, integer], optional
        Range of Y axis used in calculation [ymin, ymax]. If this parameter is
         specified, the numerical calculations are performed in the rang of
         y grids specified by this parameter in the DEM. If this parameter
         is not specified, whole range of DEM is used.

    filter_size: integer, optional
        Size of the median filter window. If this parameter is specified,
        the median filter is applied on the DEM before calculation.

    grid_spacing=10,
    initial_flow_concentration=[0.005, 0.005],
    initial_flow_thickness=100,
    initial_region_radius=100,
    initial_region_center=[1000, 4000],
    dt=50,
    number_of_steps=50,
    model="3eq",
    progress=True,
    """
    if geotiff_filename is None:
        grid = create_topography(
            length=6000,
            width=2000,
            spacing=grid_spacing,
            slope_outside=0.2,  # 0.2
            slope_inside=0.05,  # 0.02
            slope_basin_break=2000,  # 2000
            canyon_basin_break=2200,  # 2200
            canyon_center=1000,
            canyon_half_width=100,
        )

    else:
        grid = create_topography_from_geotiff(
            "depth500.tif",
            xlim=xlim,
            ylim=ylim,
            spacing=grid_spacing,
            filter_size=filter_size,
        )

    grid.set_status_at_node_on_edges(
        top=grid.BC_NODE_IS_FIXED_GRADIENT,
        bottom=grid.BC_NODE_IS_FIXED_GRADIENT,
        right=grid.BC_NODE_IS_FIXED_GRADIENT,
        left=grid.BC_NODE_IS_FIXED_GRADIENT,
    )

    create_init_flow_region(
        grid,
        initial_flow_concentration=initial_flow_concentration,
        initial_flow_thickness=initial_flow_thickness,
        initial_region_radius=initial_region_radius,
        initial_region_center=initial_region_center,  # 1000, 4000
    )

    # grid.status_at_node[grid.nodes_at_top_edge] = grid.BC_NODE_IS_FIXED_GRADIENT
    # grid.status_at_node[grid.nodes_at_bottom_edge] = grid.BC_NODE_IS_FIXED_GRADIENT
    # grid.status_at_node[grid.nodes_at_left_edge] = grid.BC_NODE_IS_FIXED_GRADIENT
    # grid.status_at_node[grid.nodes_at_right_edge] = grid.BC_NODE_IS_FIXED_GRADIENT

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

    # making turbidity current object
    tc = TurbidityCurrent2D(
        grid,
        Cf=0.004,
        alpha=0.4,
        kappa=0.1,  # 0.01
        nu_a=0.75,
        Ds=[80 * 10**-6, 40 * 10**-6],
        h_init=0.0,
        Ch_w=10 ** (-5),
        h_w=0.01,
        C_init=0.0,
        implicit_num=100,
        implicit_threshold=1.0 * 10**-10,
        r0=1.5,
        water_entrainment=True,
        suspension=True,
        model=model,
    )

    # start calculation
    t = time.time()
    tc.save_nc("tc{:04d}.nc".format(0), time=0.0)
    Ch_init = np.sum(tc.C * tc.h)

    for i in tqdm(range(1, number_of_steps + 1), disable=(progress is False)):
        tc.run_one_step(dt=dt)
        tc.save_nc("tc{:04d}.nc".format(i), time=i * dt)
        if np.sum(tc.C * tc.h) / Ch_init < 0.01:
            break
    tc.save_grid("tc{:04d}.grid".format(i))
    print("elapsed time: {} sec.".format(time.time() - t))


if __name__ == "__main__":
    """This is a script to run the model of TurbidityCurrent2D"""
    import os

    os.environ["MKL_NUM_THREADS"] = "6"
    os.environ["OMP_NUM_THREADS"] = "6"

    run()
    # run(geotiff_filename='depth500.tif',
    #     xlim=[200, 800],
    #     ylim=[400, 1200],
    #     grid_spacing=500,
    #     filter_size=[5, 5],
    #     initial_flow_concentration=0.01,
    #     initial_flow_thickness=200,
    #     initial_region_radius=30000,
    #     initial_region_center=[100000, 125000])
