"""A module for TurbidityCurrent2D to produce a grid object from a geotiff
   file or from scratch.

   codeauthor: : Hajime Naruse
"""

from landlab import RasterModelGrid
import numpy as np
from scipy.ndimage import median_filter, zoom
from landlab import FieldError
import rasterio


def create_topography(
    length=8000,
    width=2000,
    spacing=20,
    slope_outside=0.1,
    slope_inside=0.05,
    slope_basin=0.02,
    slope_basin_break=2000,
    canyon_basin_break=2200,
    canyon_center=1000,
    canyon_half_width=100,
    canyon="parabola",
    noise=0.01,
):
    """create an artificial topography where a turbidity current flow down
    A slope and a flat basn plain are set in calculation domain, and a
    parabola or v-shaped canyon is created in the slope.

    Parameters
    ------------------
     length: float, optional
        length of calculation domain [m]

     width: float, optional
        width of calculation domain [m]

     spacing: float, optional
        grid spacing [m]

     slope_outside: float, optional
        topographic inclination in the region outside the canyon

     slope_inside: float, optional
        topographic inclination in the region inside the thalweg of
        the canyon

     slope_basin: float, optional
        topographic inclination of the basin plain

     slope_basin_break: float, optional
        location of slope-basin break

     canyon_basin_break: float, optional
        location of canyon-basin break. This value must be
        larger than slope-basin break.

     canyon_center: float, optional
        location of center of the canyon

     canyon_half_width: float, optional
        half width of the canyon

     canyon: String, optional
        Style of the canyon. 'parabola' or 'V' can be chosen.

     random: float, optional
        Range of random noise to be added on generated topography

     Return
     -------------------------
     grid: RasterModelGrid
        a landlab grid object. Topographic elevation is stored as
        grid.at_node['topographic__elevation']


    """
    # making grid
    # size of calculation domain is 4 x 8 km with dx = 20 m
    lgrids = int(length / spacing)
    wgrids = int(width / spacing)
    grid = RasterModelGrid((lgrids, wgrids), xy_spacing=[spacing, spacing])
    grid.add_zeros("flow__depth", at="node")
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_zeros("flow__horizontal_velocity_at_node", at="node")
    grid.add_zeros("flow__vertical_velocity_at_node", at="node")
    grid.add_zeros("flow__horizontal_velocity", at="link")
    grid.add_zeros("flow__vertical_velocity", at="link")
    grid.add_zeros("bed__thickness", at="node")

    # making topography
    # set the slope
    grid.at_node["topographic__elevation"] = (
        grid.node_y - slope_basin_break
    ) * slope_outside

    if canyon == "parabola":
        # set canyon
        d0 = slope_inside * (canyon_basin_break - slope_basin_break)
        d = slope_inside * (grid.node_y - canyon_basin_break) - d0
        a = d0 / canyon_half_width**2
        canyon_elev = a * (grid.node_x - canyon_center) ** 2 + d
        inside = np.where(canyon_elev < grid.at_node["topographic__elevation"])
        grid.at_node["topographic__elevation"][inside] = canyon_elev[inside]

    # set basin
    basin_height = (grid.node_y - slope_basin_break) * slope_basin
    basin_region = grid.at_node["topographic__elevation"] < basin_height
    grid.at_node["topographic__elevation"][basin_region] = basin_height[basin_region]

    # add random value on topographic elevation (+- noise)
    grid.at_node["topographic__elevation"] += (
        2.0 * noise * (np.random.rand(grid.number_of_nodes) - 0.5)
    )

    grid.set_closed_boundaries_at_grid_edges(False, False, False, False)

    return grid


def create_init_flow_region(
    grid,
    initial_flow_concentration=0.02,
    initial_flow_thickness=200,
    initial_region_radius=200,
    initial_region_center=[1000, 7000],
):
    """making initial flow region in a grid, assuming lock-exchange type initiation
    of a turbidity current. Plan-view morphology of a suspended cloud is a circle,

    Parameters
    ----------------------
    grid: RasterModelGrid
       a landlab grid object

    initial_flow_concentration: float, optional
       initial flow concentration

    initial_flow_thickness: float, optional
       initial flow thickness

    initial_region_radius: float, optional
       radius of initial flow region

    initial_region_center: list, optional
       [x, y] coordinates of center of initial flow region
    """
    # check number of grain size classes
    if type(initial_flow_concentration) is float:
        initial_flow_concentration_i = np.array([initial_flow_concentration])
    else:
        initial_flow_concentration_i = np.array(initial_flow_concentration).reshape(
            len(initial_flow_concentration), 1
        )

    # initialize flow parameters
    for i in range(len(initial_flow_concentration_i)):
        try:
            grid.add_zeros("flow__sediment_concentration_{}".format(i), at="node")
        except FieldError:
            grid.at_node["flow__sediment_concentration_{}".format(i)][:] = 0.0
        try:
            grid.add_zeros("bed__sediment_volume_per_unit_area_{}".format(i), at="node")
        except FieldError:
            grid.at_node["bed__sediment_volume_per_unit_area_{}".format(i)][:] = 0.0

    try:
        grid.add_zeros("flow__sediment_concentration_total", at="node")
    except FieldError:
        grid.at_node["flow__sediment_concentration_total"][:] = 0.0
    try:
        grid.add_zeros("flow__depth", at="node")
    except FieldError:
        grid.at_node["flow__depth"][:] = 0.0
    try:
        grid.add_zeros("flow__horizontal_velocity_at_node", at="node")
    except FieldError:
        grid.at_node["flow__horizontal_velocity_at_node"][:] = 0.0
    try:
        grid.add_zeros("flow__vertical_velocity_at_node", at="node")
    except FieldError:
        grid.at_node["flow__vertical_velocity_at_node"][:] = 0.0
    try:
        grid.add_zeros("flow__horizontal_velocity", at="link")
    except FieldError:
        grid.at_link["flow__horizontal_velocity"][:] = 0.0
    try:
        grid.add_zeros("flow__vertical_velocity", at="link")
    except FieldError:
        grid.at_link["flow__vertical_velocity"][:] = 0.0

    # set initial flow region
    initial_flow_region = (
        (grid.node_x - initial_region_center[0]) ** 2
        + (grid.node_y - initial_region_center[1]) ** 2
    ) < initial_region_radius**2
    grid.at_node["flow__depth"][initial_flow_region] = initial_flow_thickness
    grid.at_node["flow__depth"][~initial_flow_region] = 0.0
    for i in range(len(initial_flow_concentration_i)):
        grid.at_node["flow__sediment_concentration_{}".format(i)][
            initial_flow_region
        ] = initial_flow_concentration_i[i]
        grid.at_node["flow__sediment_concentration_{}".format(i)][
            ~initial_flow_region
        ] = 0.0
    grid.at_node["flow__sediment_concentration_total"][initial_flow_region] = np.sum(
        initial_flow_concentration_i
    )


def create_topography_from_geotiff(
    geotiff_filename, xlim=None, ylim=None, spacing=None, filter_size=[1, 1]
):
    """create a landlab grid file from a geotiff file

    Parameters
    -----------------------
    geotiff_filename: String
       Name of a geotiff-format file to import.
       DEM coordinates must be in a projected coordinate system (e.g. UTM).

    xlim: list, optional
       list [xmin, xmax] to specify x coordinates of a region of interest
          in a geotiff file to import

    ylim: list, optional
       list [ymin, ymax] to specify y coordinates of a region of interest
          in a geotiff file to import

    spacing: float, optional
       grid spacing.
       Normally, the grid interval is automatically read from the geotif file,
       so there is no need to specify this parameter. However, if you do
       specify it, an interpolation process will be carried out to convert
       the DEM data to match the specified value. This process can take a long
       time.

    filter_size: list, optional
       [x, y] size of a window used in a median filter.
         This filter is applied for smoothing DEM data.

    Return
    ------------------------
    grid: RasterModelGrid
       a landlab grid object to be used in TurbidityCurrent2D

    """

    # read a geotiff file into ndarray
    with rasterio.open(geotiff_filename) as src:
        topo_data = src.read(1)[::-1, :]
        profile = src.profile
        width = profile["width"]
        height = profile["height"]
        transform = src.transform
        dx = transform[0]
        min_x, max_y = transform * (0, 0)
        max_x, min_y = transform * (width, height)
        xy_of_lower_left = (min_x, min_y)

    # print(topo_data.shape)
    if (xlim is not None) and (ylim is not None):
        topo_data = topo_data[xlim[0] : xlim[1], ylim[0] : ylim[1]]

    # Smoothing by median filter
    topo_data = median_filter(topo_data, size=filter_size)

    # change grid size if the parameter spacing is specified
    if spacing is not None and spacing != dx:
        zoom_factor = dx / spacing
        topo_data = zoom(topo_data, zoom_factor)
        dx = spacing

    grid = RasterModelGrid(
        topo_data.shape, xy_spacing=[dx, dx], xy_of_lower_left=xy_of_lower_left
    )
    grid.add_zeros("flow__depth", at="node")
    grid.add_zeros("topographic__elevation", at="node")
    grid.add_zeros("flow__horizontal_velocity", at="link")
    grid.add_zeros("flow__vertical_velocity", at="link")
    grid.add_zeros("bed__thickness", at="node")
    grid.at_node["topographic__elevation"][grid.nodes] = topo_data
    grid.add_zeros("flow__horizontal_velocity_at_node", at="node")
    grid.add_zeros("flow__vertical_velocity_at_node", at="node")

    return grid


def create_init_flow_region_surge(
    grid,
    initial_flow_concentration=0.02,
    initial_flow_thickness=200,
    initial_region_radius=200,
    initial_region_center=[1000, 7000],
    shallow_region_source=None,
):
    """making initial flow region in a grid, assuming lock-exchange type initiation
    of a turbidity current. Plan-view morphology of a suspended cloud is a circle,

    Parameters
    ----------------------
    grid: RasterModelGrid
       a landlab grid object

    initial_flow_concentration: float, optional
       initial flow concentration

    initial_flow_thickness: float, optional
       initial flow thickness

    initial_region_radius: float, optional
       radius of initial flow region

    initial_region_center: list, optional
       [x, y] coordinates of center of initial flow region
    """
    # check number of grain size classes
    if type(initial_flow_concentration) is float:
        initial_flow_concentration_i = np.array([initial_flow_concentration])
    else:
        initial_flow_concentration_i = np.array(initial_flow_concentration).reshape(
            len(initial_flow_concentration), 1
        )

    # initialize flow parameters
    for i in range(len(initial_flow_concentration_i)):
        try:
            grid.add_zeros("flow__sediment_concentration_{}".format(i), at="node")
        except FieldError:
            grid.at_node["flow__sediment_concentration_{}".format(i)][:] = 0.0
        try:
            grid.add_zeros("bed__sediment_volume_per_unit_area_{}".format(i), at="node")
        except FieldError:
            grid.at_node["bed__sediment_volume_per_unit_area_{}".format(i)][:] = 0.0

    try:
        grid.add_zeros("flow__sediment_concentration_total", at="node")
    except FieldError:
        grid.at_node["flow__sediment_concentration_total"][:] = 0.0
    try:
        grid.add_zeros("flow__depth", at="node")
    except FieldError:
        grid.at_node["flow__depth"][:] = 0.0
    try:
        grid.add_zeros("flow__horizontal_velocity_at_node", at="node")
    except FieldError:
        grid.at_node["flow__horizontal_velocity_at_node"][:] = 0.0
    try:
        grid.add_zeros("flow__vertical_velocity_at_node", at="node")
    except FieldError:
        grid.at_node["flow__vertical_velocity_at_node"][:] = 0.0
    try:
        grid.add_zeros("flow__horizontal_velocity", at="link")
    except FieldError:
        grid.at_link["flow__horizontal_velocity"][:] = 0.0
    try:
        grid.add_zeros("flow__vertical_velocity", at="link")
    except FieldError:
        grid.at_link["flow__vertical_velocity"][:] = 0.0

    # check circular initial flow region (initial_flow_radius = float) or rectangular (initial_flow_radius = [x_half_length, y_half_length])
    if type(initial_region_radius) is float:
        initial_flow_region = (
            (grid.node_x - initial_region_center[0]) ** 2
            + (grid.node_y - initial_region_center[1]) ** 2
        ) < initial_region_radius**2
    else:
        initial_flow_region = (
            (grid.node_x - initial_region_center[0]) ** 2
            < initial_region_radius[0] ** 2
        ) & (
            (grid.node_y - initial_region_center[1]) ** 2
            < initial_region_radius[1] ** 2
        )

    if shallow_region_source is not None:
        initial_flow_region = (
            grid.at_node["topographic__elevation"] > shallow_region_source
        ) & (grid.at_node["topographic__elevation"] < 0)

    # set initial flow region
    grid.at_node["flow__depth"][initial_flow_region] = initial_flow_thickness
    if shallow_region_source is not None:
        very_shallow_region = (
            -grid.at_node["topographic__elevation"] < initial_flow_thickness
        ) & (grid.at_node["topographic__elevation"] < 0.0)
        grid.at_node["flow__depth"][very_shallow_region] = -grid.at_node[
            "topographic__elevation"
        ][very_shallow_region]

    grid.at_node["flow__depth"][~initial_flow_region] = 0.0
    for i in range(len(initial_flow_concentration_i)):
        grid.at_node["flow__sediment_concentration_{}".format(i)][
            initial_flow_region
        ] = initial_flow_concentration_i[i]
        grid.at_node["flow__sediment_concentration_{}".format(i)][
            ~initial_flow_region
        ] = 0.0
    grid.at_node["flow__sediment_concentration_total"][initial_flow_region] = np.sum(
        initial_flow_concentration_i
    )


def create_init_flow_region_cont(
    grid,
    node_y_0,
    node_y_1,
    node_x_1,
    flow_sediment_concentration=0.005,
    flow_depth=30.0,
    flow_vertical_velocity=4.0,
    flow_horizontal_velocity=0.0,
):
    """making initial flow region in a grid, assuming continuous type initiation
    of a turbidity current. Plan-view morphology of a suspended cloud is a rectangle,

    Parameters
    ----------------------
    grid: RasterModelGrid
       a landlab grid object

    node_y_0: int
       uppermost grid of initiation area

    node_y_1: int
       lowermost grid of initiation area

    node_x_1: int
       rightmost grid of initiation area

    flow_sediment_concentration: float, optional
       initial flow concentration

    flow_depth: float, optional
       initial flow thickness

    flow_vertical_velocity; float, optional
       initial flow velocity in vertical direction

    flow_horizontal_velocity; float, optional
       initial flow velocity in horizontal direction
    """

    # check number of grain size classes
    if type(flow_sediment_concentration) is float:
        flow_sediment_concentration_i = np.array([flow_sediment_concentration])
    else:
        flow_sediment_concentration_i = np.array(flow_sediment_concentration).reshape(
            len(flow_sediment_concentration), 1
        )

    # initialize flow parameters
    for i in range(len(flow_sediment_concentration_i)):
        try:
            grid.add_zeros("flow__sediment_concentration_{}".format(i), at="node")
        except FieldError:
            grid.at_node["flow__sediment_concentration_{}".format(i)][:] = 0.0
        try:
            grid.add_zeros("bed__sediment_volume_per_unit_area_{}".format(i), at="node")
        except FieldError:
            grid.at_node["bed__sediment_volume_per_unit_area_{}".format(i)][:] = 0.0

    try:
        grid.add_zeros("flow__sediment_concentration_total", at="node")
    except FieldError:
        grid.at_node["flow__sediment_concentration_total"][:] = 0.0
    try:
        grid.add_zeros("flow__depth", at="node")
    except FieldError:
        grid.at_node["flow__depth"][:] = 0.0
    try:
        grid.add_zeros("flow__horizontal_velocity_at_node", at="node")
    except FieldError:
        grid.at_node["flow__horizontal_velocity_at_node"][:] = 0.0
    try:
        grid.add_zeros("flow__vertical_velocity_at_node", at="node")
    except FieldError:
        grid.at_node["flow__vertical_velocity_at_node"][:] = 0.0
    try:
        grid.add_zeros("flow__horizontal_velocity", at="link")
    except FieldError:
        grid.at_link["flow__horizontal_velocity"][:] = 0.0
    try:
        grid.add_zeros("flow__vertical_velocity", at="link")
    except FieldError:
        grid.at_link["flow__vertical_velocity"][:] = 0.0

    # set inlet region
    inlet = np.where(
        (grid.y_of_node > node_y_0)
        & (grid.y_of_node < node_y_1)
        & (grid.x_of_node < node_x_1)
    )
    inlet_link = np.where(
        (grid.midpoint_of_link[:, 1] > node_y_0)
        & (grid.midpoint_of_link[:, 1] < node_y_1)
        & (grid.midpoint_of_link[:, 0] < node_x_1)
    )

    grid.at_node["flow__depth"][inlet] = flow_depth
    grid.at_node["flow__horizontal_velocity_at_node"][inlet] = flow_horizontal_velocity
    grid.at_node["flow__vertical_velocity_at_node"][inlet] = -flow_vertical_velocity
    grid.at_link["flow__horizontal_velocity"][inlet_link] = flow_horizontal_velocity
    grid.at_link["flow__vertical_velocity"][inlet_link] = -flow_vertical_velocity
    for i in range(len(flow_sediment_concentration_i)):
        grid.at_node["flow__sediment_concentration_{}".format(i)][inlet] = (
            flow_sediment_concentration_i[i]
        )
    grid.at_node["flow__sediment_concentration_total"][inlet] = np.sum(
        flow_sediment_concentration_i
    )
