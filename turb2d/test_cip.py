from .cip import cip_1d_advection, rcip_1d_advection, tangentf, atangentf, cip_2d_advection, cip_2d_M_advection, rcip_2d_M_advection, cubic_interp_1d, rcubic_interp_1d, rcip_2d_advection, forester_filter, Jameson, CIP2D
from landlab import RasterModelGrid
from landlab.grid.structured_quad import links
from landlab.grid.raster_mappers import map_mean_of_links_to_node
import matplotlib.pyplot as plt
from .turb2d import TurbidityCurrent2D
from .gridutils import set_up_neighbor_arrays
import numpy as np


def test_cip_1d_advection(h, dhdx, loop=10, out_h=None, out_dhdx=None):

    assert h.shape[0] == dhdx.shape[0]

    dx = 1.0
    dt = 0.1

    h_temp = h.copy()
    dhdx_temp = dhdx.copy()

    n = h.shape[0]
    u = np.ones(n, dtype=np.float)
    core = np.arange(1, n - 1, 1, dtype=np.int)
    up = np.arange(0, n - 2, 1, dtype=np.int)
    down = np.arange(2, n, 1, dtype=np.int)

    u[0:int(n / 2)] = -1.0
    up[np.where(u < 0)] += 2
    down[np.where(u < 0)] -= 2

    for i in range(loop):

        cip_1d_advection(h_temp,
                         dhdx_temp,
                         u,
                         core,
                         up,
                         down,
                         dx,
                         dt,
                         out_f=out_h,
                         out_dfdx=out_dhdx)

        h_temp[core] = out_h[core] - out_h[core] * (u[2:] - u[:-2]) / (2 *
                                                                       dx) * dt
        dhdx_temp[core] = out_dhdx[core] + (
            (h_temp[2:] - out_h[2:]) - (h_temp[:-2] - out_h[:-2])) / (
                2 * dx) - out_dhdx[core] * (u[2:] - u[:-2]) / (2 * dx) * dt

    out_h[core] = h_temp[core]
    out_dhdx[core] = dhdx_temp[core]


def test_rcip_1d_advection(h, dhdx, loop=10, out_h=None, out_dhdx=None):

    assert h.shape[0] == dhdx.shape[0]

    dx = 1.0
    dt = 0.1

    h_temp = h.copy()
    dhdx_temp = dhdx.copy()

    n = h.shape[0]
    u = np.ones(n, dtype=np.float)
    core = np.arange(1, n - 1, 1, dtype=np.int)
    up = np.arange(0, n - 2, 1, dtype=np.int)
    down = np.arange(2, n, 1, dtype=np.int)

    u[0:int(n / 2)] = -1.0
    up[np.where(u < 0)] += 2
    down[np.where(u < 0)] -= 2

    for i in range(loop):

        rcip_1d_advection(h_temp,
                          dhdx_temp,
                          u,
                          core,
                          up,
                          down,
                          dx,
                          dt,
                          out_f=out_h,
                          out_dfdx=out_dhdx)

        h_temp[core] = out_h[core] - out_h[core] * (u[2:] - u[:-2]) / (2 *
                                                                       dx) * dt
        dhdx_temp[core] = out_dhdx[core] + (
            (h_temp[2:] - out_h[2:]) - (h_temp[:-2] - out_h[:-2])) / (
                2 * dx) - out_dhdx[core] * (u[2:] - u[:-2]) / (2 * dx) * dt

    out_h[core] = h_temp[core]
    out_dhdx[core] = dhdx_temp[core]


def test_tcip_1d_advection(h, dhdx, loop=10, out_h=None, out_dhdx=None):

    assert h.shape[0] == dhdx.shape[0]

    dx = 1.0
    dt = 0.1

    h_temp = tangentf(h / (np.max(h) * 1.1))
    import ipdb
    ipdb.set_trace()
    h_temp2 = h.copy()
    dhdx_temp = np.zeros(h.shape[0])
    h_temp3 = h_temp.copy()
    dhdx_temp2 = dhdx_temp.copy()

    n = h.shape[0]
    u = np.ones(n, dtype=np.float)
    core = np.arange(1, n - 1, 1, dtype=np.int)
    up = np.arange(0, n - 2, 1, dtype=np.int)
    down = np.arange(2, n, 1, dtype=np.int)

    u[0:int(n / 2)] = -1.0
    up[np.where(u < 0)] += 2
    down[np.where(u < 0)] -= 2

    for i in range(loop):

        cip_1d_advection(h_temp,
                         dhdx_temp,
                         u,
                         core,
                         up,
                         down,
                         dx,
                         dt,
                         out_f=h_temp2,
                         out_dfdx=dhdx_temp2)

        h_temp3[core] = atangentf(h_temp2[core]) * np.max(h) * 1.1
        h_temp3[core] -= h_temp3[core] * (u[2:] - u[:-2]) / (2 * dx) * dt
        h_temp2 = tangentf(h_temp3 / (np.max(h) * 1.1))
        dhdx_temp[core] = dhdx_temp2[core] + (
            (h_temp2[2:] - h_temp[2:]) - (h_temp2[:-2] - h_temp[:-2])) / (
                2 * dx) - dhdx_temp2[core] * (u[2:] - u[:-2]) / (2 * dx) * dt
        h_temp[:] = h_temp2[:]
        dhdx_temp[:] = dhdx_temp2[:]

    out_h[:] = atangentf(h_temp[:]) * np.max(h) * 1.1
    out_dhdx[:] = dhdx_temp[:]

    return out_h, out_dhdx


def test_cip_2d_advection(grid=None,
                          h=None,
                          dhdx=None,
                          dhdy=None,
                          loop=10,
                          out_h=None,
                          out_dhdx=None):
    if grid is None:
        grid = RasterModelGrid([100, 100], xy_spacing=[1.0, 1.0])
        grid.add_zeros('flow__depth', at='node')
        grid.add_zeros('flow_depth__horizontal_gradient', at='node')
        grid.add_zeros('flow_depth__vertical_gradient', at='node')
        grid.add_zeros('flow__horizontal_velocity', at='link')
        grid.add_zeros('flow__vertical_velocity', at='link')
        grid.at_link['flow__horizontal_velocity'][:] = 1.0
        grid.at_link['flow__vertical_velocity'][:] = 1.0
        grid.at_link['flow__horizontal_velocity'][
            grid.xy_of_link[:, 0] < 50.] = -1.0
        grid.at_link['flow__vertical_velocity'][
            grid.xy_of_link[:, 1] < 50.] = -1.0

    if h is None:
        h = grid.at_node['flow__depth']
        h[np.where((grid.x_of_node - 50.)**2 +
                   (grid.y_of_node - 50.)**2 < 15.**2)] = 1.0

    if dhdx is None:
        dhdx = grid.at_node['flow_depth__horizontal_gradient']

    if dhdy is None:
        dhdy = grid.at_node['flow_depth__vertical_gradient']

    out_h = np.zeros(h.shape)
    out_dhdx = np.zeros(h.shape)
    out_dhdy = np.zeros(h.shape)

    core = grid.core_nodes
    h_up = grid.adjacent_nodes_at_node[:, 2].copy()
    v_up = grid.adjacent_nodes_at_node[:, 3].copy()
    u = grid.at_link['flow__horizontal_velocity']
    v = grid.at_link['flow__vertical_velocity']
    dx = 1.0
    dt = 0.1

    east = grid.adjacent_nodes_at_node[:, 0][core]
    west = grid.adjacent_nodes_at_node[:, 2][core]
    north = grid.adjacent_nodes_at_node[:, 1][core]
    south = grid.adjacent_nodes_at_node[:, 3][core]

    east_link = grid.links_at_node[:, 0][core]
    north_link = grid.links_at_node[:, 1][core]
    west_link = grid.links_at_node[:, 2][core]
    south_link = grid.links_at_node[:, 3][core]

    u_node = map_mean_of_links_to_node(grid, 'flow__horizontal_velocity')
    v_node = map_mean_of_links_to_node(grid, 'flow__vertical_velocity')

    h_up[np.where(u_node < 0.)] = grid.adjacent_nodes_at_node[:, 0][np.where(
        u_node < 0.)]
    v_up[np.where(v_node < 0.)] = grid.adjacent_nodes_at_node[:, 1][np.where(
        v_node < 0.)]

    for i in range(loop):

        div = (u[east_link] - u[west_link]) / (dx) + (v[north_link] -
                                                      v[south_link]) / (dx)

        out_h[core] = h[core] - h[core] * div * dt

        out_dhdx[core] = dhdx[core] + (
            (out_h[east] - h[east]) - (out_h[west] - h[west])) / (2 * dx) - (
                dhdx[core] * (u[east_link] - u[west_link]) /
                (dx) + dhdy[core] * (v[east_link] - v[east_link]) / (dx)) * dt

        out_dhdy[core] = dhdy[core] + (
            (out_h[north] - h[north]) - (out_h[south] - h[south])) / (
                2 * dx) - (dhdx[core] * (u[north_link] - u[south_link]) /
                           (dx) + dhdy[core] *
                           (v[north_link] - v[south_link]) / (dx)) * dt

        h[:] = out_h[:]
        dhdx[:] = out_dhdx[:]
        dhdy[:] = out_dhdy[:]

        cip_2d_advection(h,
                         dhdx,
                         dhdy,
                         u_node,
                         v_node,
                         core,
                         h_up,
                         v_up,
                         dx,
                         dt,
                         out_f=out_h,
                         out_dfdx=out_dhdx,
                         out_dfdy=out_dhdy)

        h[:] = out_h[:]
        dhdx[:] = out_dhdx[:]
        dhdy[:] = out_dhdy[:]

    return grid, h, dhdx, dhdy


def test_rcip_2d_advection(grid=None,
                           h=None,
                           dhdx=None,
                           dhdy=None,
                           loop=10,
                           out_h=None,
                           out_dhdx=None):
    if grid is None:
        grid = RasterModelGrid([100, 100], xy_spacing=[1.0, 1.0])
        grid.add_zeros('flow__depth', at='node')
        grid.add_zeros('flow_depth__horizontal_gradient', at='node')
        grid.add_zeros('flow_depth__vertical_gradient', at='node')
        grid.add_zeros('flow__horizontal_velocity', at='link')
        grid.add_zeros('flow__vertical_velocity', at='link')
        grid.at_link['flow__horizontal_velocity'][:] = 1.0
        grid.at_link['flow__vertical_velocity'][:] = 1.0
        grid.at_link['flow__horizontal_velocity'][
            grid.xy_of_link[:, 0] < 50.] = -1.0
        grid.at_link['flow__vertical_velocity'][
            grid.xy_of_link[:, 1] < 50.] = -1.0

    if h is None:
        h = grid.at_node['flow__depth']
        h[np.where((grid.x_of_node - 50.)**2 +
                   (grid.y_of_node - 50.)**2 < 15.**2)] = 1.0

    if dhdx is None:
        dhdx = grid.at_node['flow_depth__horizontal_gradient']

    if dhdy is None:
        dhdy = grid.at_node['flow_depth__vertical_gradient']

    out_h = np.zeros(h.shape)
    out_dhdx = np.zeros(h.shape)
    out_dhdy = np.zeros(h.shape)

    core = grid.core_nodes
    h_up = grid.adjacent_nodes_at_node[:, 2].copy()
    v_up = grid.adjacent_nodes_at_node[:, 3].copy()
    u = grid.at_link['flow__horizontal_velocity']
    v = grid.at_link['flow__vertical_velocity']
    dx = 1.0
    dt = 0.1

    east = grid.adjacent_nodes_at_node[:, 0][core]
    west = grid.adjacent_nodes_at_node[:, 2][core]
    north = grid.adjacent_nodes_at_node[:, 1][core]
    south = grid.adjacent_nodes_at_node[:, 3][core]

    east_link = grid.links_at_node[:, 0][core]
    north_link = grid.links_at_node[:, 1][core]
    west_link = grid.links_at_node[:, 2][core]
    south_link = grid.links_at_node[:, 3][core]

    u_node = map_mean_of_links_to_node(grid, 'flow__horizontal_velocity')
    v_node = map_mean_of_links_to_node(grid, 'flow__vertical_velocity')

    h_up[np.where(u_node < 0.)] = grid.adjacent_nodes_at_node[:, 0][np.where(
        u_node < 0.)]
    v_up[np.where(v_node < 0.)] = grid.adjacent_nodes_at_node[:, 1][np.where(
        v_node < 0.)]

    for i in range(loop):

        div = (u[east_link] - u[west_link]) / (dx) + (v[north_link] -
                                                      v[south_link]) / (dx)

        out_h[core] = h[core] - h[core] * div * dt

        out_dhdx[core] = dhdx[core] + (
            (out_h[east] - h[east]) - (out_h[west] - h[west])) / (2 * dx) - (
                dhdx[core] * (u[east_link] - u[west_link]) /
                (dx) + dhdy[core] * (v[east_link] - v[east_link]) / (dx)) * dt

        out_dhdy[core] = dhdy[core] + (
            (out_h[north] - h[north]) - (out_h[south] - h[south])) / (
                2 * dx) - (dhdx[core] * (u[north_link] - u[south_link]) /
                           (dx) + dhdy[core] *
                           (v[north_link] - v[south_link]) / (dx)) * dt

        h[:] = out_h[:]
        dhdx[:] = out_dhdx[:]
        dhdy[:] = out_dhdy[:]

        rcip_2d_advection(h,
                          dhdx,
                          dhdy,
                          u_node,
                          v_node,
                          core,
                          h_up,
                          v_up,
                          dx,
                          dt,
                          out_f=out_h,
                          out_dfdx=out_dhdx,
                          out_dfdy=out_dhdy)

        h[:] = out_h[:]
        dhdx[:] = out_dhdx[:]
        dhdy[:] = out_dhdy[:]

    return grid, h, dhdx, dhdy


def test_CIP2D(grid=None,
               h=None,
               dhdx=None,
               dhdy=None,
               loop=10,
               out_h=None,
               out_dhdx=None):

    if grid is None:
        grid = RasterModelGrid([100, 100], xy_spacing=[1.0, 1.0])
        grid.add_zeros('flow__depth', at='node')
        grid.add_zeros('flow_depth__horizontal_gradient', at='node')
        grid.add_zeros('flow_depth__vertical_gradient', at='node')
        grid.add_zeros('flow__horizontal_velocity', at='link')
        grid.add_zeros('flow__vertical_velocity', at='link')
        grid.at_link['flow__horizontal_velocity'][:] = 1.0
        grid.at_link['flow__vertical_velocity'][:] = 1.0
        grid.at_link['flow__horizontal_velocity'][
            grid.xy_of_link[:, 0] < 50.] = -1.0
        grid.at_link['flow__vertical_velocity'][
            grid.xy_of_link[:, 1] < 50.] = -1.0

    if h is None:
        h = grid.at_node['flow__depth']
        h[np.where((grid.x_of_node - 50.)**2 +
                   (grid.y_of_node - 50.)**2 < 15.**2)] = 1.0

    if dhdx is None:
        dhdx = grid.at_node['flow_depth__horizontal_gradient']

    if dhdy is None:
        dhdy = grid.at_node['flow_depth__vertical_gradient']

    cip2d = CIP2D(h.shape[0])

    out_h = np.zeros(h.shape)
    out_dhdx = np.zeros(h.shape)
    out_dhdy = np.zeros(h.shape)

    core = grid.core_nodes
    h_up = grid.adjacent_nodes_at_node[:, 2].copy()
    v_up = grid.adjacent_nodes_at_node[:, 3].copy()
    u = grid.at_link['flow__horizontal_velocity']
    v = grid.at_link['flow__vertical_velocity']
    dx = 1.0
    dt = 0.1

    east = grid.adjacent_nodes_at_node[:, 0][core]
    west = grid.adjacent_nodes_at_node[:, 2][core]
    north = grid.adjacent_nodes_at_node[:, 1][core]
    south = grid.adjacent_nodes_at_node[:, 3][core]

    east_link = grid.links_at_node[:, 0][core]
    north_link = grid.links_at_node[:, 1][core]
    west_link = grid.links_at_node[:, 2][core]
    south_link = grid.links_at_node[:, 3][core]

    u_node = map_mean_of_links_to_node(grid, 'flow__horizontal_velocity')
    v_node = map_mean_of_links_to_node(grid, 'flow__vertical_velocity')

    h_up[np.where(u_node < 0.)] = grid.adjacent_nodes_at_node[:, 0][np.where(
        u_node < 0.)]
    v_up[np.where(v_node < 0.)] = grid.adjacent_nodes_at_node[:, 1][np.where(
        v_node < 0.)]

    import ipdb
    ipdb.set_trace()

    for i in range(loop):

        div = (u[east_link] - u[west_link]) / (dx) + (v[north_link] -
                                                      v[south_link]) / (dx)

        out_h[core] = h[core] - h[core] * div * dt

        out_dhdx[core] = dhdx[core] + (
            (out_h[east] - h[east]) - (out_h[west] - h[west])) / (2 * dx) - (
                dhdx[core] * (u[east_link] - u[west_link]) /
                (dx) + dhdy[core] * (v[east_link] - v[east_link]) / (dx)) * dt

        out_dhdy[core] = dhdy[core] + (
            (out_h[north] - h[north]) - (out_h[south] - h[south])) / (
                2 * dx) - (dhdx[core] * (u[north_link] - u[south_link]) /
                           (dx) + dhdy[core] *
                           (v[north_link] - v[south_link]) / (dx)) * dt

        h[:] = out_h[:]
        dhdx[:] = out_dhdx[:]
        dhdy[:] = out_dhdy[:]

        cip2d.run(h,
                  dhdx,
                  dhdy,
                  u_node,
                  v_node,
                  core,
                  h_up,
                  v_up,
                  dx,
                  dt,
                  out_f=out_h,
                  out_dfdx=out_dhdx,
                  out_dfdy=out_dhdy)

        h[:] = out_h[:]
        dhdx[:] = out_dhdx[:]
        dhdy[:] = out_dhdy[:]

    return grid, h, dhdx, dhdy


def test_cip_2d_M_advection(grid=None,
                            h=None,
                            dhdx=None,
                            dhdy=None,
                            loop=10,
                            out_h=None,
                            out_dhdx=None):
    if grid is None:
        grid = RasterModelGrid([100, 100], spacing=1.0)
        grid.add_zeros('flow__depth', at='node')
        grid.add_zeros('flow_depth__horizontal_gradient', at='node')
        grid.add_zeros('flow_depth__vertical_gradient', at='node')
        grid.add_zeros('flow__horizontal_velocity', at='link')
        grid.add_zeros('flow__vertical_velocity', at='link')
        grid.at_link['flow__horizontal_velocity'][:] = 1.0
        grid.at_link['flow__vertical_velocity'][:] = 1.0
        grid.at_link['flow__horizontal_velocity'][grid.x_of_link < 50.] = -1.0
        grid.at_link['flow__vertical_velocity'][grid.y_of_link < 50.] = -1.0

    if h is None:
        h = grid.at_node['flow__depth']
        h[np.where((grid.x_of_node - 50.)**2 +
                   (grid.y_of_node - 50.)**2 < 15.**2)] = 1.0

    if dhdx is None:
        dhdx = grid.at_node['flow_depth__horizontal_gradient']

    if dhdy is None:
        dhdy = grid.at_node['flow_depth__vertical_gradient']

    out_h = np.zeros(h.shape)
    out_dhdx = np.zeros(h.shape)
    out_dhdy = np.zeros(h.shape)

    core = grid.core_nodes
    h_up = grid.adjacent_nodes_at_node[:, 2].copy()
    v_up = grid.adjacent_nodes_at_node[:, 3].copy()
    u = grid.at_link['flow__horizontal_velocity']
    v = grid.at_link['flow__vertical_velocity']
    dx = 1.0
    dt = 0.1

    east = grid.adjacent_nodes_at_node[:, 0][core]
    west = grid.adjacent_nodes_at_node[:, 2][core]
    north = grid.adjacent_nodes_at_node[:, 1][core]
    south = grid.adjacent_nodes_at_node[:, 3][core]

    east_link = grid.links_at_node[:, 0][core]
    north_link = grid.links_at_node[:, 1][core]
    west_link = grid.links_at_node[:, 2][core]
    south_link = grid.links_at_node[:, 3][core]

    u_node = map_mean_of_links_to_node(grid, 'flow__horizontal_velocity')
    v_node = map_mean_of_links_to_node(grid, 'flow__vertical_velocity')

    h_up[np.where(u_node < 0.)] = grid.adjacent_nodes_at_node[:, 0][np.where(
        u_node < 0.)]
    v_up[np.where(v_node < 0.)] = grid.adjacent_nodes_at_node[:, 1][np.where(
        v_node < 0.)]

    for i in range(loop):

        div = (u[east_link] - u[west_link]) / (dx) + (v[north_link] -
                                                      v[south_link]) / (dx)

        out_h[core] = h[core] - h[core] * div * dt

        out_dhdx[core] = dhdx[core] + (
            (out_h[east] - h[east]) - (out_h[west] - h[west])) / (2 * dx) - (
                dhdx[core] * (u[east_link] - u[west_link]) /
                (dx) + dhdy[core] * (v[east_link] - v[west_link]) / (dx)) * dt

        out_dhdy[core] = dhdy[core] + (
            (out_h[north] - h[north]) - (out_h[south] - h[south])) / (
                2 * dx) - (dhdx[core] * (u[north_link] - u[south_link]) /
                           (dx) + dhdy[core] *
                           (v[north_link] - v[south_link]) / (dx)) * dt

        h[:] = out_h[:]
        dhdx[:] = out_dhdx[:]
        dhdy[:] = out_dhdy[:]

        cip_2d_M_advection(h,
                           dhdx,
                           dhdy,
                           u_node,
                           v_node,
                           core,
                           h_up,
                           v_up,
                           dx,
                           dt,
                           out_f=out_h,
                           out_dfdx=out_dhdx,
                           out_dfdy=out_dhdy)

        h[:] = out_h[:]
        dhdx[:] = out_dhdx[:]
        dhdy[:] = out_dhdy[:]

    return grid, h, dhdx, dhdy


def test_rcip_2d_M_advection(grid=None,
                             h=None,
                             dhdx=None,
                             dhdy=None,
                             loop=10,
                             out_h=None,
                             out_dhdx=None):

    if grid is None:
        grid = RasterModelGrid([100, 100], spacing=1.0)
        grid.add_zeros('flow__depth', at='node')
        grid.add_zeros('flow_depth__horizontal_gradient', at='node')
        grid.add_zeros('flow_depth__vertical_gradient', at='node')
        grid.add_zeros('flow__horizontal_velocity', at='link')
        grid.add_zeros('flow__vertical_velocity', at='link')
        grid.at_link['flow__horizontal_velocity'][:] = 1.0
        grid.at_link['flow__vertical_velocity'][:] = 1.0
        grid.at_link['flow__horizontal_velocity'][grid.x_of_link < 50.] = -1.0
        grid.at_link['flow__vertical_velocity'][grid.y_of_link < 50.] = -1.0

    if h is None:
        h = grid.at_node['flow__depth']
        h[np.where((grid.x_of_node - 50.)**2 +
                   (grid.y_of_node - 50.)**2 < 15.**2)] = 1.0

    if dhdx is None:
        dhdx = grid.at_node['flow_depth__horizontal_gradient']

    if dhdy is None:
        dhdy = grid.at_node['flow_depth__vertical_gradient']

    out_h = np.zeros(h.shape)
    out_dhdx = np.zeros(h.shape)
    out_dhdy = np.zeros(h.shape)

    core = grid.core_nodes
    h_up = grid.adjacent_nodes_at_node[:, 2].copy()
    v_up = grid.adjacent_nodes_at_node[:, 3].copy()
    u = grid.at_link['flow__horizontal_velocity']
    v = grid.at_link['flow__vertical_velocity']
    dx = 1.0
    dt = 0.1

    east = grid.adjacent_nodes_at_node[:, 0][core]
    west = grid.adjacent_nodes_at_node[:, 2][core]
    north = grid.adjacent_nodes_at_node[:, 1][core]
    south = grid.adjacent_nodes_at_node[:, 3][core]

    east_link = grid.links_at_node[:, 0][core]
    north_link = grid.links_at_node[:, 1][core]
    west_link = grid.links_at_node[:, 2][core]
    south_link = grid.links_at_node[:, 3][core]

    u_node = map_mean_of_links_to_node(grid, 'flow__horizontal_velocity')
    v_node = map_mean_of_links_to_node(grid, 'flow__vertical_velocity')

    h_up[np.where(u_node < 0.)] = grid.adjacent_nodes_at_node[:, 0][np.where(
        u_node < 0.)]
    v_up[np.where(v_node < 0.)] = grid.adjacent_nodes_at_node[:, 1][np.where(
        v_node < 0.)]

    for i in range(loop):

        div = (u[east_link] - u[west_link]) / (dx) + (v[north_link] -
                                                      v[south_link]) / (dx)

        out_h[core] = h[core] - h[core] * div * dt

        out_dhdx[core] = dhdx[core] + (
            (out_h[east] - h[east]) - (out_h[west] - h[west])) / (2 * dx) - (
                dhdx[core] * (u[east_link] - u[west_link]) /
                (dx) + dhdy[core] * (v[east_link] - v[east_link]) / (dx)) * dt

        out_dhdy[core] = dhdy[core] + (
            (out_h[north] - h[north]) - (out_h[south] - h[south])) / (
                2 * dx) - (dhdx[core] * (u[north_link] - u[south_link]) /
                           (dx) + dhdy[core] *
                           (v[north_link] - v[south_link]) / (dx)) * dt

        h[:] = out_h[:]
        dhdx[:] = out_dhdx[:]
        dhdy[:] = out_dhdy[:]

        rcip_2d_M_advection(h,
                            dhdx,
                            dhdy,
                            u_node,
                            v_node,
                            core,
                            h_up,
                            v_up,
                            dx,
                            dt,
                            out_f=out_h,
                            out_dfdx=out_dhdx,
                            out_dfdy=out_dhdy)

        h[:] = out_h[:]
        dhdx[:] = out_dhdx[:]
        dhdy[:] = out_dhdy[:]

    return grid, h, dhdx, dhdy


def test_cubic_interp_1d():

    import ipdb
    ipdb.set_trace()

    x = np.arange(0, 100, 1, dtype=float)
    x_half = np.arange(0.5, 99, 1.0, dtype=float)
    dx = 1.0

    core = np.arange(0, 99, 1, dtype=int)
    right = np.arange(1, 100, 1, dtype=int)
    left = np.arange(0, 99, 1, dtype=int)

    u = np.zeros(x.shape)
    u[np.where((x > 40) & (x < 60))] = 1.0
    u_interp = np.zeros(x_half.shape)

    dudx = np.zeros(x.shape)
    dudx[1:-1] = (u[2:] - u[:-2]) / 2.0

    rcubic_interp_1d(u, dudx, core, right, left, dx, out=u_interp)

    plt.plot(x, u, 'bo')
    plt.plot(x_half, u_interp)
    plt.savefig('interp_test.png')

    return x, x_half, u, u_interp


def test_forester_filter():

    grid = RasterModelGrid([10, 10])
    grid.add_zeros('flow__sediment_concentration', at='node')
    grid.at_node['flow__depth'] = np.random.rand(grid.number_of_nodes) - 0.2
    h = grid.at_node['flow__depth']
    core = grid.core_nodes
    east = grid.adjacent_nodes_at_node[:, 0]
    north = grid.adjacent_nodes_at_node[:, 1]
    west = grid.adjacent_nodes_at_node[:, 2]
    south = grid.adjacent_nodes_at_node[:, 3]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    im1 = ax1.imshow(h[grid.nodes], vmax=1.0, vmin=-1.0)
    fig.colorbar(im1)

    forester_filter(h, core, east, west, north, south, out_f=h, loop=100)

    im2 = ax2.imshow(h[grid.nodes], vmax=1.0, vmin=-1.0)
    fig.colorbar(im2)
    plt.savefig('h_forester.png')


def test_jameson():

    fig, (ax1, ax2) = plt.subplots(1, 2)

    grid = RasterModelGrid([10, 10])
    grid.add_zeros('flow__depth', at='node')
    h = grid.at_node['flow__depth']
    h[:] = np.random.rand(10 * 10)
    ax1.imshow(h[grid.nodes])

    core = grid.core_nodes
    tc = TurbidityCurrent2D(grid)
    set_up_neighbor_arrays(tc)

    jm = Jameson(grid.number_of_nodes, grid.number_of_links, tc.node_east,
                 tc.node_west, tc.node_north, tc.node_south,
                 grid.horizontal_links, grid.vertical_links,
                 tc.east_node_at_horizontal_link,
                 tc.west_node_at_horizontal_link,
                 tc.north_node_at_vertical_link,
                 tc.south_node_at_vertical_link, tc.east_link_at_node,
                 tc.west_link_at_node, tc.north_link_at_node,
                 tc.south_link_at_node, 0.2)
    jm.update_artificial_viscosity(h)
    jm.run(h, core, out=h)
    ax2.imshow(h[grid.nodes])

    plt.savefig('jameson.png')
