"""Utility functions for landlab grids
"""
import numpy as np
from landlab.grid.structured_quad import links
from .cip import cubic_interp_1d, rcubic_interp_1d


def set_up_neighbor_arrays(tc):
    """Create and initialize link neighbor arrays.

    Set up arrays of neighboring horizontal and vertical nodes that are
    needed for CIP solution.
    """

    # Find the neighbor nodes
    tc.core_nodes = tc.grid.core_nodes
    neighbor_nodes = tc.grid.adjacent_nodes_at_node.copy()
    tc.node_east = neighbor_nodes[:, 0]
    tc.node_north = neighbor_nodes[:, 1]
    tc.node_west = neighbor_nodes[:, 2]
    tc.node_south = neighbor_nodes[:, 3]

    # Find the neighbor links
    tc.active_links = tc.grid.active_links
    neighbor_links = links.neighbors_at_link(
        tc.grid.shape, np.arange(tc.grid.number_of_links)).copy()
    tc.link_east = neighbor_links[:, 0]
    tc.link_north = neighbor_links[:, 1]
    # This is to fix a bug in links.neighbors_at_link
    tc.link_north[tc.link_north == tc.grid.number_of_links] = -1
    tc.link_west = neighbor_links[:, 2]
    tc.link_south = neighbor_links[:, 3]

    # Find links connected to nodes, and nodes connected to links
    tc.east_link_at_node = tc.grid.links_at_node[:, 0].copy()
    tc.north_link_at_node = tc.grid.links_at_node[:, 1].copy()
    tc.west_link_at_node = tc.grid.links_at_node[:, 2].copy()
    tc.south_link_at_node = tc.grid.links_at_node[:, 3].copy()
    tc.west_node_at_horizontal_link = tc.grid.nodes_at_link[:, 0].copy()
    tc.east_node_at_horizontal_link = tc.grid.nodes_at_link[:, 1].copy()
    tc.south_node_at_vertical_link = tc.grid.nodes_at_link[:, 0].copy()
    tc.north_node_at_vertical_link = tc.grid.nodes_at_link[:, 1].copy()

    # bound_node_north_at_link = np.where(
    #     tc.grid.node_is_boundary(tc.north_node_at_vertical_link))
    # bound_node_south_at_link = np.where(
    #     tc.grid.node_is_boundary(tc.south_node_at_vertical_link))
    # bound_node_east_at_link = np.where(
    #     tc.grid.node_is_boundary(tc.east_node_at_horizontal_link))
    # bound_node_west_at_link = np.where(
    #     tc.grid.node_is_boundary(tc.west_node_at_horizontal_link))
    # tc.north_node_at_vertical_link[
    #     bound_node_north_at_link] = tc.south_node_at_vertical_link[
    #         bound_node_north_at_link]
    # tc.south_node_at_vertical_link[
    #     bound_node_south_at_link] = tc.north_node_at_vertical_link[
    #         bound_node_south_at_link]
    # tc.east_node_at_horizontal_link[
    #     bound_node_east_at_link] = tc.west_node_at_horizontal_link[
    #         bound_node_east_at_link]
    # tc.west_node_at_horizontal_link[
    #     bound_node_west_at_link] = tc.east_node_at_horizontal_link[
    #         bound_node_west_at_link]

    # Find obliquely neighbor vertical links from horizontal links
    # and obliquely neighbor horizontal links from vertical links
    tc.vertical_link_NE = tc.north_link_at_node[
        tc.east_node_at_horizontal_link]
    tc.vertical_link_SE = tc.south_link_at_node[
        tc.east_node_at_horizontal_link]
    tc.vertical_link_NW = tc.north_link_at_node[
        tc.west_node_at_horizontal_link]
    tc.vertical_link_SW = tc.south_link_at_node[
        tc.west_node_at_horizontal_link]
    tc.horizontal_link_NE = tc.east_link_at_node[
        tc.north_node_at_vertical_link]
    tc.horizontal_link_SE = tc.east_link_at_node[
        tc.south_node_at_vertical_link]
    tc.horizontal_link_NW = tc.west_link_at_node[
        tc.north_node_at_vertical_link]
    tc.horizontal_link_SW = tc.west_link_at_node[
        tc.south_node_at_vertical_link]

    # Once the neighbor arrays are set up, we change the flag to True!
    tc.neighbor_flag = True


def update_up_down_links_and_nodes(tc):
    """update location of upcurrent and downcurrent
       nodes and links
    """

    find_horizontal_up_down_nodes(tc,
                                  tc.u_node,
                                  out_up=tc.horizontal_up_nodes,
                                  out_down=tc.horizontal_down_nodes)

    find_vertical_up_down_nodes(tc,
                                tc.v_node,
                                out_up=tc.vertical_up_nodes,
                                out_down=tc.vertical_down_nodes)

    find_horizontal_up_down_links(tc,
                                  tc.u,
                                  out_up=tc.horizontal_up_links,
                                  out_down=tc.horizontal_down_links)

    find_vertical_up_down_links(tc,
                                tc.v,
                                out_up=tc.vertical_up_links,
                                out_down=tc.vertical_down_links)


def map_values(tc, h, dhdx, dhdy, u, dudx, v, dvdy, Ch, dChdx, dChdy, eta,
               h_link, u_node, v_node, Ch_link, U, U_node):
    """map parameters at nodes to links, and those at links to nodes
    """
    map_links_to_nodes(tc, u, dudx, v, dvdy, u_node, v_node, U, U_node)
    map_nodes_to_links(tc, h, dhdx, dhdy, Ch, dChdx, dChdy, eta, h_link,
                       Ch_link)


def map_links_to_nodes(tc, u, dudx, v, dvdy, u_node, v_node, U, U_node):
    """map parameters at links to nodes
    """
    dx = tc.grid.dx
    dry_links = tc.dry_links
    dry_nodes = tc.dry_nodes
    wet_pwet_nodes = tc.wet_pwet_nodes
    wet_pwet_links = tc.wet_pwet_links
    wet_pwet_vertical_links = tc.wet_pwet_vertical_links
    wet_pwet_horizontal_links = tc.wet_pwet_horizontal_links
    horizontal_link_NE = tc.horizontal_link_NE[wet_pwet_vertical_links]
    horizontal_link_NW = tc.horizontal_link_NW[wet_pwet_vertical_links]
    horizontal_link_SE = tc.horizontal_link_SE[wet_pwet_vertical_links]
    horizontal_link_SW = tc.horizontal_link_SW[wet_pwet_vertical_links]
    vertical_link_NE = tc.vertical_link_NE[wet_pwet_horizontal_links]
    vertical_link_NW = tc.vertical_link_NW[wet_pwet_horizontal_links]
    vertical_link_SE = tc.vertical_link_SE[wet_pwet_horizontal_links]
    vertical_link_SW = tc.vertical_link_SW[wet_pwet_horizontal_links]
    north_link_at_node = tc.north_link_at_node[wet_pwet_nodes]
    south_link_at_node = tc.south_link_at_node[wet_pwet_nodes]
    east_link_at_node = tc.east_link_at_node[wet_pwet_nodes]
    west_link_at_node = tc.west_link_at_node[wet_pwet_nodes]

    # set velocity zero at dry links and nodes
    u[dry_links] = 0
    v[dry_links] = 0
    u_node[dry_nodes] = 0
    v_node[dry_nodes] = 0
    dudx[dry_links] = 0
    dvdy[dry_links] = 0

    # Map values of horizontal links to vertical links, and
    # values of vertical links to horizontal links.
    # Horizontal velocity is only updated at horizontal links,
    # and vertical velocity is updated at vertical links during
    # CIP procedures. Therefore we need to map those values each other.
    u[wet_pwet_vertical_links] = (
        u[horizontal_link_NE] + u[horizontal_link_NW] + u[horizontal_link_SE] +
        u[horizontal_link_SW]) / 4.0
    v[wet_pwet_horizontal_links] = (v[vertical_link_NE] + v[vertical_link_NW] +
                                    v[vertical_link_SE] +
                                    v[vertical_link_SW]) / 4.0

    # Calculate composite velocity at links
    U[wet_pwet_links] = np.sqrt(u[wet_pwet_links]**2 + v[wet_pwet_links]**2)

    # map link values (u, v) to nodes
    cubic_interp_1d(u,
                    dudx,
                    wet_pwet_nodes,
                    east_link_at_node,
                    west_link_at_node,
                    dx,
                    out=u_node)
    cubic_interp_1d(v,
                    dvdy,
                    wet_pwet_nodes,
                    north_link_at_node,
                    south_link_at_node,
                    dx,
                    out=v_node)
    # map_mean_of_links_to_node(u,
    #                           wet_pwet_nodes,
    #                           north_link_at_node,
    #                           south_link_at_node,
    #                           east_link_at_node,
    #                           west_link_at_node,
    #                           out=u_node)
    # map_mean_of_links_to_node(v,
    #                           wet_pwet_nodes,
    #                           north_link_at_node,
    #                           south_link_at_node,
    #                           east_link_at_node,
    #                           west_link_at_node,
    #                           out=v_node)
    map_mean_of_links_to_node(U,
                              wet_pwet_nodes,
                              north_link_at_node,
                              south_link_at_node,
                              east_link_at_node,
                              west_link_at_node,
                              out=U_node)
    # tc.grid.map_mean_of_links_to_node(u, out=u_node)
    # tc.grid.map_mean_of_links_to_node(v, out=v_node)
    # tc.grid.map_mean_of_links_to_node(U, out=U_node)


def map_mean_of_links_to_node(f,
                              core,
                              north_link_at_node,
                              south_link_at_node,
                              east_link_at_node,
                              west_link_at_node,
                              out=None):

    n = f.shape[0]
    if out is None:
        out = np.zeros(n)

    out[core] = (f[north_link_at_node] + f[south_link_at_node] +
                 f[east_link_at_node] + f[west_link_at_node]) / 4.0

    return out


def map_nodes_to_links(tc, h, dhdx, dhdy, Ch, dChdx, dChdy, eta, h_link,
                       Ch_link):
    """map parameters at nodes to links
    """

    # remove illeagal values
    h[h < tc.h_init] = tc.h_init
    Ch[Ch < tc.C_init * tc.h_init] = tc.C_init * tc.h_init
    h[tc.dry_nodes] = tc.h_init
    Ch[tc.dry_nodes] = tc.h_init * tc.C_init
    dhdx[tc.dry_nodes] = 0
    dhdy[tc.dry_nodes] = 0
    dChdx[tc.dry_nodes] = 0
    dChdy[tc.dry_nodes] = 0

    north_node_at_vertical_link = tc.north_node_at_vertical_link[
        tc.wet_pwet_vertical_links]
    south_node_at_vertical_link = tc.south_node_at_vertical_link[
        tc.wet_pwet_vertical_links]
    east_node_at_horizontal_link = tc.east_node_at_horizontal_link[
        tc.wet_pwet_horizontal_links]
    west_node_at_horizontal_link = tc.west_node_at_horizontal_link[
        tc.wet_pwet_horizontal_links]
    dx = tc.grid.dx

    # map node values (h, C, eta) to links
    rcubic_interp_1d(h,
                     dhdx,
                     tc.wet_pwet_horizontal_links,
                     east_node_at_horizontal_link,
                     west_node_at_horizontal_link,
                     dx,
                     out=h_link)
    rcubic_interp_1d(h,
                     dhdy,
                     tc.wet_pwet_vertical_links,
                     north_node_at_vertical_link,
                     south_node_at_vertical_link,
                     dx,
                     out=h_link)
    rcubic_interp_1d(Ch,
                     dChdx,
                     tc.wet_pwet_horizontal_links,
                     east_node_at_horizontal_link,
                     west_node_at_horizontal_link,
                     dx,
                     out=Ch_link)
    rcubic_interp_1d(Ch,
                     dChdy,
                     tc.wet_pwet_vertical_links,
                     north_node_at_vertical_link,
                     south_node_at_vertical_link,
                     dx,
                     out=Ch_link)

    # map_mean_of_link_nodes_to_link(h,
    #                                tc.wet_pwet_horizontal_links,
    #                                tc.wet_pwet_vertical_links,
    #                                north_node_at_vertical_link,
    #                                south_node_at_vertical_link,
    #                                east_node_at_horizontal_link,
    #                                west_node_at_horizontal_link,
    #                                out=h_link)
    # map_mean_of_link_nodes_to_link(Ch,
    #                                tc.wet_pwet_horizontal_links,
    #                                tc.wet_pwet_vertical_links,
    #                                north_node_at_vertical_link,
    #                                south_node_at_vertical_link,
    #                                east_node_at_horizontal_link,
    #                                west_node_at_horizontal_link,
    #                                out=Ch_link)


def map_mean_of_link_nodes_to_link(f,
                                   horizontal_link,
                                   vertical_link,
                                   north_node_at_vertical_link,
                                   south_node_at_vertical_link,
                                   east_node_at_horizontal_link,
                                   west_node_at_horizontal_link,
                                   out=None):
    """ map mean of parameters at nodes to link
    """
    if out is None:
        out = np.zeros(f.shape[0], dtype=np.int)

    out[horizontal_link] = (f[east_node_at_horizontal_link] +
                            f[west_node_at_horizontal_link]) / 2.0
    out[vertical_link] = (f[north_node_at_vertical_link] +
                          f[south_node_at_vertical_link]) / 2.0

    return out


def find_horizontal_up_down_nodes(tc, u, out_up=None, out_down=None):
    """Find indeces of nodes that locate
       at horizontally upcurrent and downcurrent directions
    """
    if out_up is None:
        out_up = np.empty(u.shape[0], dtype=np.int)
    if out_down is None:
        out_down = np.empty(u.shape[0], dtype=np.int)

    out_up[:] = tc.node_west[:]
    out_down[:] = tc.node_east[:]
    negative_u_index = np.where(u < 0)[0]
    out_up[negative_u_index] = tc.node_east[negative_u_index]
    out_down[negative_u_index] = tc.node_west[negative_u_index]

    return out_up, out_down


def find_vertical_up_down_nodes(tc, u, out_up=None, out_down=None):
    """Find indeces of nodes that locate
       at vertically upcurrent and downcurrent directions
    """

    if out_up is None:
        out_up = np.empty(u.shape[0], dtype=np.int)
    if out_down is None:
        out_down = np.empty(u.shape[0], dtype=np.int)

    out_up[:] = tc.node_south[:]
    out_down[:] = tc.node_north[:]
    negative_u_index = np.where(u < 0)[0]
    out_up[negative_u_index] = tc.node_north[negative_u_index]
    out_down[negative_u_index] = tc.node_south[negative_u_index]

    return out_up, out_down


def find_horizontal_up_down_links(tc, u, out_up=None, out_down=None):
    """Find indeces of nodes that locate
       at horizontally upcurrent and downcurrent directions
    """
    if out_up is None:
        out_up = np.zeros(u.shape[0], dtype=np.int)
    if out_down is None:
        out_down = np.zeros(u.shape[0], dtype=np.int)

    out_up[:] = tc.link_west[:]
    out_down[:] = tc.link_east[:]
    negative_u_index = np.where(u < 0)[0]
    out_up[negative_u_index] = tc.link_east[negative_u_index]
    out_down[negative_u_index] = tc.link_west[negative_u_index]
    return out_up, out_down


def find_vertical_up_down_links(tc, u, out_up=None, out_down=None):
    """Find indeces of nodes that locate
       at vertically upcurrent and downcurrent directions
    """

    if out_up is None:
        out_up = np.zeros(u.shape[0], dtype=np.int)
    if out_down is None:
        out_down = np.zeros(u.shape[0], dtype=np.int)

    out_up[:] = tc.link_south[:]
    out_down[:] = tc.link_north[:]
    negative_u_index = np.where(u < 0)[0]
    out_up[negative_u_index] = tc.link_north[negative_u_index]
    out_down[negative_u_index] = tc.link_south[negative_u_index]

    return out_up, out_down


def find_boundary_links_nodes(tc):
    """find and record boundary links and nodes
    """
    grid = tc.grid

    # Boundary Types
    FIXED_GRADIENT = grid.BC_NODE_IS_FIXED_GRADIENT
    FIXED_VALUE = grid.BC_NODE_IS_FIXED_VALUE
    CLOSED = grid.BC_NODE_IS_CLOSED

    # Find boundary and edge links
    bound_link_north = tc.south_link_at_node[grid.nodes_at_top_edge]
    bound_link_south = tc.north_link_at_node[grid.nodes_at_bottom_edge]
    bound_link_east = tc.west_link_at_node[grid.nodes_at_right_edge]
    bound_link_west = tc.east_link_at_node[grid.nodes_at_left_edge]
    edge_link_north = links.top_edge_horizontal_ids(grid.shape)
    edge_link_south = links.bottom_edge_horizontal_ids(grid.shape)
    edge_link_east = links.right_edge_vertical_ids(grid.shape)
    edge_link_west = links.left_edge_vertical_ids(grid.shape)
    tc.bound_links = np.unique(
        np.concatenate([
            bound_link_north, bound_link_south, bound_link_east,
            bound_link_west
        ]))
    tc.edge_links = np.unique(
        np.concatenate(
            [edge_link_north, edge_link_south, edge_link_east,
             edge_link_west]))

    ##################################
    # fixed gradient nodes and links #
    ##################################
    fixed_grad_link_at_north = bound_link_north[grid.node_is_boundary(
        tc.north_node_at_vertical_link[bound_link_north],
        boundary_flag=FIXED_GRADIENT)]
    fixed_grad_link_at_south = bound_link_south[grid.node_is_boundary(
        tc.south_node_at_vertical_link[bound_link_south],
        boundary_flag=FIXED_GRADIENT)]
    fixed_grad_link_at_east = bound_link_east[grid.node_is_boundary(
        tc.east_node_at_horizontal_link[bound_link_east],
        boundary_flag=FIXED_GRADIENT)]
    fixed_grad_link_at_west = bound_link_west[grid.node_is_boundary(
        tc.west_node_at_horizontal_link[bound_link_west],
        boundary_flag=FIXED_GRADIENT)]
    fixed_grad_edge_link_at_north = edge_link_north[grid.node_is_boundary(
        tc.east_node_at_horizontal_link[edge_link_north],
        boundary_flag=FIXED_GRADIENT)]
    fixed_grad_edge_link_at_south = edge_link_south[grid.node_is_boundary(
        tc.east_node_at_horizontal_link[edge_link_south],
        boundary_flag=FIXED_GRADIENT)]
    fixed_grad_edge_link_at_east = edge_link_east[grid.node_is_boundary(
        tc.north_node_at_vertical_link[edge_link_east],
        boundary_flag=FIXED_GRADIENT)]
    fixed_grad_edge_link_at_west = edge_link_west[grid.node_is_boundary(
        tc.north_node_at_vertical_link[edge_link_west],
        boundary_flag=FIXED_GRADIENT)]
    fixed_grad_anchor_link_at_north = tc.link_south[fixed_grad_link_at_north]
    fixed_grad_anchor_link_at_south = tc.link_north[fixed_grad_link_at_south]
    fixed_grad_anchor_link_at_east = tc.link_west[fixed_grad_link_at_east]
    fixed_grad_anchor_link_at_west = tc.link_east[fixed_grad_link_at_west]
    fixed_grad_anchor_edge_link_at_north = tc.link_south[
        fixed_grad_edge_link_at_north]
    fixed_grad_anchor_edge_link_at_south = tc.link_north[
        fixed_grad_edge_link_at_south]
    fixed_grad_anchor_edge_link_at_east = tc.link_west[
        fixed_grad_edge_link_at_east]
    fixed_grad_anchor_edge_link_at_west = tc.link_east[
        fixed_grad_edge_link_at_west]

    # Fixed gradient nodes and adjacent nodes (anchor)
    tc.fixed_grad_nodes = grid.fixed_gradient_boundary_nodes
    tc.fixed_grad_anchor_nodes = grid.fixed_gradient_boundary_node_anchor_node

    # Fixed gradient links and adjacent links (anchor)
    tc.fixed_grad_links = np.concatenate([
        fixed_grad_link_at_north, fixed_grad_link_at_south,
        fixed_grad_link_at_east, fixed_grad_link_at_west,
        fixed_grad_edge_link_at_north, fixed_grad_edge_link_at_south,
        fixed_grad_edge_link_at_east, fixed_grad_edge_link_at_west
    ])
    tc.fixed_grad_anchor_links = np.concatenate([
        fixed_grad_anchor_link_at_north, fixed_grad_anchor_link_at_south,
        fixed_grad_anchor_link_at_east, fixed_grad_anchor_link_at_west,
        fixed_grad_anchor_edge_link_at_north,
        fixed_grad_anchor_edge_link_at_south,
        fixed_grad_anchor_edge_link_at_east,
        fixed_grad_anchor_edge_link_at_west
    ])

    ###############################
    # fixed value nodes and links #
    ###############################
    fixed_value_nodes_at_north = tc.grid.nodes_at_top_edge[
        grid.node_is_boundary(tc.grid.nodes_at_top_edge,
                              boundary_flag=FIXED_VALUE)]
    fixed_value_nodes_at_south = tc.grid.nodes_at_bottom_edge[
        grid.node_is_boundary(tc.grid.nodes_at_bottom_edge,
                              boundary_flag=FIXED_VALUE)]
    fixed_value_nodes_at_east = tc.grid.nodes_at_right_edge[
        grid.node_is_boundary(tc.grid.nodes_at_right_edge,
                              boundary_flag=FIXED_VALUE)]
    fixed_value_nodes_at_west = tc.grid.nodes_at_left_edge[
        grid.node_is_boundary(tc.grid.nodes_at_left_edge,
                              boundary_flag=FIXED_VALUE)]
    fixed_value_anchor_nodes_at_north = tc.node_south[
        fixed_value_nodes_at_north]
    fixed_value_anchor_nodes_at_south = tc.node_north[
        fixed_value_nodes_at_south]
    fixed_value_anchor_nodes_at_east = tc.node_west[fixed_value_nodes_at_east]
    fixed_value_anchor_nodes_at_west = tc.node_east[fixed_value_nodes_at_west]
    fixed_value_link_at_north = bound_link_north[grid.node_is_boundary(
        tc.north_node_at_vertical_link[bound_link_north],
        boundary_flag=FIXED_VALUE)]
    fixed_value_link_at_south = bound_link_south[grid.node_is_boundary(
        tc.south_node_at_vertical_link[bound_link_south],
        boundary_flag=FIXED_VALUE)]
    fixed_value_link_at_east = bound_link_east[grid.node_is_boundary(
        tc.east_node_at_horizontal_link[bound_link_east],
        boundary_flag=FIXED_VALUE)]
    fixed_value_link_at_west = bound_link_west[grid.node_is_boundary(
        tc.west_node_at_horizontal_link[bound_link_west],
        boundary_flag=FIXED_VALUE)]
    fixed_value_anchor_link_at_north = tc.link_south[fixed_value_link_at_north]
    fixed_value_anchor_link_at_south = tc.link_north[fixed_value_link_at_south]
    fixed_value_anchor_link_at_east = tc.link_west[fixed_value_link_at_east]
    fixed_value_anchor_link_at_west = tc.link_east[fixed_value_link_at_west]
    fixed_value_edge_link_at_north = edge_link_north[grid.node_is_boundary(
        tc.east_node_at_horizontal_link[edge_link_north],
        boundary_flag=FIXED_VALUE)]
    fixed_value_edge_link_at_south = edge_link_south[grid.node_is_boundary(
        tc.east_node_at_horizontal_link[edge_link_south],
        boundary_flag=FIXED_VALUE)]
    fixed_value_edge_link_at_east = edge_link_east[grid.node_is_boundary(
        tc.north_node_at_vertical_link[edge_link_east],
        boundary_flag=FIXED_VALUE)]
    fixed_value_edge_link_at_west = edge_link_west[grid.node_is_boundary(
        tc.north_node_at_vertical_link[edge_link_west],
        boundary_flag=FIXED_VALUE)]

    # To avoid referreing indeces that do not exist
    tc.link_north[fixed_value_link_at_north] = fixed_value_link_at_north
    tc.link_south[fixed_value_link_at_south] = fixed_value_link_at_south
    tc.link_east[fixed_value_link_at_east] = fixed_value_link_at_east
    tc.link_west[fixed_value_link_at_west] = fixed_value_link_at_west

    # Record fixed value nodes and links with anchor and edge links
    tc.fixed_value_nodes = np.concatenate([
        fixed_value_nodes_at_north, fixed_value_nodes_at_south,
        fixed_value_nodes_at_east, fixed_value_nodes_at_west
    ])
    tc.fixed_value_anchor_nodes = np.concatenate([
        fixed_value_anchor_nodes_at_north, fixed_value_anchor_nodes_at_south,
        fixed_value_anchor_nodes_at_east, fixed_value_anchor_nodes_at_west
    ])
    tc.fixed_value_links = np.concatenate([
        fixed_value_link_at_north, fixed_value_link_at_south,
        fixed_value_link_at_east, fixed_value_link_at_west
    ])
    tc.fixed_value_anchor_links = np.concatenate([
        fixed_value_anchor_link_at_north, fixed_value_anchor_link_at_south,
        fixed_value_anchor_link_at_east, fixed_value_anchor_link_at_west
    ])
    tc.fixed_value_edge_links = np.concatenate([
        fixed_value_edge_link_at_north, fixed_value_edge_link_at_south,
        fixed_value_edge_link_at_east, fixed_value_edge_link_at_west
    ])
