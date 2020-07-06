"""A module for TurbidityCurrent2D to produce a grid object from a geotiff
   file or from scratch. 

   codeauthor: : Hajime Naruse
"""

from landlab import RasterModelGrid
import numpy as np
from osgeo import gdal, gdalconst
from scipy.ndimage import median_filter


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
        canyon='parabola',
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
    basin_height = (grid.node_y - slope_basin_break) * slope_basin
    basin_region = grid.at_node['topographic__elevation'] < basin_height
    grid.at_node['topographic__elevation'][basin_region] = basin_height[
        basin_region]

    # add random value on topographic elevation (+- noise)
    grid.at_node['topographic__elevation'] += 2.0 * noise * (
        np.random.rand(grid.number_of_nodes) - 0.5)

    grid.set_closed_boundaries_at_grid_edges(False, False, False, False)

    return grid


def create_init_flow_region(
        grid,
        initial_flow_concentration=0.02,
        initial_flow_thickness=200,
        initial_region_radius=200,
        initial_region_center=[1000, 7000],
):
    """ making initial flow region in a grid, assuming lock-exchange type initiation
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

    initial_flow_region = (
        (grid.node_x - initial_region_center[0])**2 +
        (grid.node_y - initial_region_center[1])**2) < initial_region_radius**2
    grid.at_node['flow__depth'][initial_flow_region] = initial_flow_thickness
    grid.at_node['flow__depth'][~initial_flow_region] = 0.0
    grid.at_node['flow__sediment_concentration'][
        initial_flow_region] = initial_flow_concentration
    grid.at_node['flow__sediment_concentration'][~initial_flow_region] = 0.0


def create_topography_from_geotiff(geotiff_filename,
                                   xlim=None,
                                   ylim=None,
                                   spacing=500,
                                   filter_size=[1, 1]):
    """create a landlab grid file from a geotiff file

       Parameters
       -----------------------
       geotiff_filename: String
          name of a geotiff-format file to import 

       xlim: list, optional
          list [xmin, xmax] to specify x coordinates of a region of interest 
             in a geotiff file to import

       ylim: list, optional
          list [ymin, ymax] to specify y coordinates of a region of interest 
             in a geotiff file to import

       spacing: float, optional
          grid spacing

       filter_size: list, optional
          [x, y] size of a window used in a median filter.
            This filter is applied for smoothing DEM data.

       Return
       ------------------------
       grid: RasterModelGrid
          a landlab grid object to be used in TurbidityCurrent2D

    """

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
