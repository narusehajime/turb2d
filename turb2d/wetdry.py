"""This module process wet/dry grids

   find_wet_grids is used for finding wet, partial wet, and dry grids
"""

import numpy as np


def find_wet_grids(tc):
    """Find wet and partial wet nodes and links on the basis of pressure
    In this model, "dry" nodes are not subject to calculate.
    Only "wet nodes" are considered in the model
    calculation. "wet" is judged by the water pressure (> p_w).
    The "partial wet node" is a dry node but the upcurrent
    node is wet. Flow depth and velocity at partial wet
    nodes are calculated by the YANG's model (YANG et al.,
    2016)

    Parameters
    --------------------------
    tc : TurbidityCurrent2D
        TurbidityCurrent2D object to be checked

    Values set in TurbidityCurrent2D object
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
    Ch = tc.Ch
    core = tc.core_nodes
    # horiz_links = tc.horizontal_active_links
    horiz_links = tc.grid.horizontal_links
    # vert_links = tc.vertical_active_links
    vert_links = tc.grid.vertical_links
    node_east = tc.node_east
    node_west = tc.node_west
    node_north = tc.node_north
    node_south = tc.node_south
    east_link_at_node = tc.east_link_at_node
    west_link_at_node = tc.west_link_at_node
    north_link_at_node = tc.north_link_at_node
    south_link_at_node = tc.south_link_at_node
    east_nodes_at_link = tc.east_node_at_horizontal_link
    west_nodes_at_link = tc.west_node_at_horizontal_link
    north_nodes_at_link = tc.north_node_at_vertical_link
    south_nodes_at_link = tc.south_node_at_vertical_link
    Ch_w = tc.Ch_w
    h_w = tc.h_w

    ############################
    # find wet nodes and links #
    ############################
    # tc.wet_nodes = core[np.where(p[core] > p_w)]
    # tc.wet_horizontal_links = horiz_links[np.where(
    #     (p[west_nodes_at_link[horiz_links]] > p_w) & (p[east_nodes_at_link[horiz_links]] > p_w))]
    # tc.wet_vertical_links = vert_links[np.where(
    #     (p[north_nodes_at_link] > p_w) & (p[south_nodes_at_link] > p_w))]

    wet_nodes = (Ch > Ch_w) & (tc.h > h_w)
    tc.wet_nodes = core[wet_nodes[core]]

    tc.wet_horizontal_links = horiz_links[
        (wet_nodes[west_nodes_at_link[horiz_links]])
        & (wet_nodes[east_nodes_at_link[horiz_links]])
    ]
    tc.wet_vertical_links = vert_links[
        (wet_nodes[north_nodes_at_link[vert_links]])
        & (wet_nodes[south_nodes_at_link[vert_links]])
    ]

    ######################################################
    # find partial wet nodes and links in horizontal axis #
    ######################################################
    wet_at_east = np.where(~(wet_nodes[core]) & (wet_nodes[node_east[core]]))
    # wet_at_east = ~(wet_nodes[core]) & wet_nodes[node_east[core]]
    horizontally_partial_wet_nodes_E = core[wet_at_east]
    horizontally_wettest_nodes_E = node_east[core[wet_at_east]]
    partial_wet_horizontal_links_E = east_link_at_node[core][wet_at_east]
    horizontal_direction_wettest_E = -1.0 * np.ones(wet_at_east[0].shape)

    wet_at_west = np.where(~(wet_nodes[core]) & (wet_nodes[node_west[core]]))
    # wet_at_west = ~(wet_nodes[core]) & wet_nodes[node_west[core]]
    horizontally_partial_wet_nodes_W = core[wet_at_west]
    horizontally_wettest_nodes_W = node_west[core[wet_at_west]]
    partial_wet_horizontal_links_W = west_link_at_node[core][wet_at_west]
    horizontal_direction_wettest_W = 1.0 * np.ones(wet_at_west[0].shape)

    tc.horizontally_partial_wet_nodes = np.concatenate(
        [horizontally_partial_wet_nodes_E, horizontally_partial_wet_nodes_W]
    )
    tc.horizontally_wettest_nodes = np.concatenate(
        [horizontally_wettest_nodes_E, horizontally_wettest_nodes_W]
    )
    tc.partial_wet_horizontal_links = np.concatenate(
        [partial_wet_horizontal_links_E, partial_wet_horizontal_links_W]
    )
    tc.horizontal_direction_wettest = np.concatenate(
        [horizontal_direction_wettest_E, horizontal_direction_wettest_W]
    )

    ######################################################
    # find partial wet nodes and links in vertical axis  #
    ######################################################
    # vertical partial wet check
    wet_at_north = np.where(~(wet_nodes[core]) & (wet_nodes[node_north[core]]))
    vertically_partial_wet_nodes_N = core[wet_at_north]
    vertically_wettest_nodes_N = node_north[core[wet_at_north]]
    partial_wet_vertical_links_N = north_link_at_node[core][wet_at_north]
    vertical_direction_wettest_N = -1.0 * np.ones(wet_at_north[0].shape)

    wet_at_south = np.where(~(wet_nodes[core]) & (wet_nodes[node_south[core]]))
    vertically_partial_wet_nodes_S = core[wet_at_south]
    vertically_wettest_nodes_S = node_south[core[wet_at_south]]
    partial_wet_vertical_links_S = south_link_at_node[core][wet_at_south]
    vertical_direction_wettest_S = 1.0 * np.ones(wet_at_south[0].shape)

    # combine north and south
    tc.vertically_partial_wet_nodes = np.concatenate(
        [vertically_partial_wet_nodes_N, vertically_partial_wet_nodes_S]
    )
    tc.vertically_wettest_nodes = np.concatenate(
        [vertically_wettest_nodes_N, vertically_wettest_nodes_S]
    )
    tc.partial_wet_vertical_links = np.concatenate(
        [partial_wet_vertical_links_N, partial_wet_vertical_links_S]
    )
    tc.vertical_direction_wettest = np.concatenate(
        [vertical_direction_wettest_N, vertical_direction_wettest_S]
    )

    #######################################
    # wet and partial wet nodes and links #
    #######################################
    tc.partial_wet_nodes = np.unique(
        np.concatenate(
            [tc.horizontally_partial_wet_nodes, tc.vertically_partial_wet_nodes]
        )
    )
    tc.wettest_nodes = np.unique(
        np.concatenate([tc.horizontally_wettest_nodes, tc.vertically_wettest_nodes])
    )
    tc.wet_pwet_nodes = np.unique(np.concatenate([tc.wet_nodes, tc.partial_wet_nodes]))
    tc.wet_pwet_horizontal_links = np.concatenate(
        [tc.wet_horizontal_links, tc.partial_wet_horizontal_links]
    )
    tc.wet_pwet_vertical_links = np.concatenate(
        [tc.wet_vertical_links, tc.partial_wet_vertical_links]
    )
    tc.wet_pwet_links = np.concatenate(
        [tc.wet_pwet_horizontal_links, tc.wet_pwet_vertical_links]
    )

    #################################
    #
    #
    tc.wettest_nodes = np.unique(
        np.concatenate([tc.horizontally_wettest_nodes, tc.vertically_wettest_nodes])
    )

    ############################
    # find dry nodes and links #
    ############################
    tc.dry_nodes = np.setdiff1d(core, tc.wet_pwet_nodes, assume_unique=True)
    tc.dry_links = np.setdiff1d(
        tc.active_links,
        np.concatenate([tc.wet_pwet_links, tc.fixed_value_links]),
        assume_unique=True,
    )


def process_partial_wet_grids(
    tc,
    h,
    u,
    v,
    Ch,
    h_out=None,
    u_out=None,
    v_out=None,
    Ch_out=None,
):
    """Process parameters of partial wet nodes and links

    Parameters
    ----------------------------
    tc : TurbidityCurrent2D object
         TurbidityCurrent2D object to be processed

    h_out : ndarray, float
         Flow depth

    u_out : ndarray, float
         flow horizontal velocity

    v_out : ndarray, float
         flow vertical velocity

    Ch_out : ndarray, float
         volume of suspended sediment

    """
    if h_out is None:
        h_out = h.copy()
    if u_out is None:
        u_out = u.copy()
    if v_out is None:
        v_out = v.copy()
    if Ch_out is None:
        Ch_out = Ch.copy()

    #######################################
    # get variables at the current moment #
    #######################################
    g = tc.g
    R = tc.R
    Cf = tc.Cf
    dt = tc.dt_local
    eta = tc.eta
    # empirical coefficient (Homma)
    gamma = tc.gamma

    # grid information
    dry_links = tc.dry_links
    horizontally_partial_wet_nodes = tc.horizontally_partial_wet_nodes
    vertically_partial_wet_nodes = tc.vertically_partial_wet_nodes
    horizontally_wettest_nodes = tc.horizontally_wettest_nodes
    vertically_wettest_nodes = tc.vertically_wettest_nodes
    partial_wet_horizontal_links = tc.partial_wet_horizontal_links
    partial_wet_vertical_links = tc.partial_wet_vertical_links
    horizontal_direction_wettest = tc.horizontal_direction_wettest
    vertical_direction_wettest = tc.vertical_direction_wettest

    ######################################################
    # horizontal and vertical flow discharge between wet #
    # and partial wet nodes                              #
    ######################################################
    # horizontal_overspill_height = (h[horizontally_wettest_nodes] +
    #                                eta[horizontally_wettest_nodes]) - (
    #                                    h[horizontally_partial_wet_nodes] +
    #                                    eta[horizontally_partial_wet_nodes])
    # horizontal_overspill_height[horizontal_overspill_height < 0] = 0
    horizontal_overspill_height = h[horizontally_wettest_nodes]
    # vertical_overspill_height = (
    #     h[vertically_wettest_nodes] + eta[vertically_wettest_nodes]
    # ) - (h[vertically_partial_wet_nodes] + eta[vertically_partial_wet_nodes])
    # vertical_overspill_height[vertical_overspill_height < 0] = 0
    vertical_overspill_height = h[vertically_wettest_nodes]

    horizontal_overspill_velocity = gamma * np.sqrt(
        2.0
        * R
        * g
        * horizontal_overspill_height
        * Ch[horizontally_wettest_nodes]
        / h[horizontally_wettest_nodes]
    )
    vertical_overspill_velocity = gamma * np.sqrt(
        2.0
        * R
        * g
        * vertical_overspill_height
        * Ch[vertically_wettest_nodes]
        / h[vertically_wettest_nodes]
    )

    # tc.M_horiz = horizontal_direction_wettest * M_horiz
    # tc.CM_horiz = horizontal_direction_wettest * CM_horiz
    # tc.M_vert = vertical_direction_wettest * M_vert
    # tc.CM_vert = vertical_direction_wettest * CM_vert

    ################################################################
    # Calculate time development of variables at partial wet links #
    ################################################################
    CfuU = (
        Cf
        * np.sqrt(
            u[partial_wet_horizontal_links] * u[partial_wet_horizontal_links]
            + v[partial_wet_horizontal_links] * v[partial_wet_horizontal_links]
        )
        / (h[horizontally_wettest_nodes] + h[horizontally_partial_wet_nodes])
        * 2
    )
    CfvU = (
        Cf
        * np.sqrt(
            u[partial_wet_vertical_links] * u[partial_wet_vertical_links]
            + v[partial_wet_vertical_links] * v[partial_wet_vertical_links]
        )
        / (h[vertically_wettest_nodes] + h[vertically_partial_wet_nodes])
        * 2
    )

    hdw = horizontal_direction_wettest
    vdw = vertical_direction_wettest

    u_out[partial_wet_horizontal_links] = (
        hdw * horizontal_overspill_velocity + u[partial_wet_horizontal_links]
    )
    u_out[partial_wet_horizontal_links] *= 1 / (1 + CfuU * dt)
    u_out[dry_links] = 0

    v_out[partial_wet_vertical_links] = (
        vdw * vertical_overspill_velocity + v[partial_wet_vertical_links]
    )
    v_out[partial_wet_vertical_links] *= 1 / (1 + CfvU * dt)
    v_out[dry_links] = 0

    return h_out, u_out, v_out, Ch_out
