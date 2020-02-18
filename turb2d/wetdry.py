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
    p = tc.h * tc.Ch
    core = tc.core_nodes
    horiz_links = tc.horizontal_active_links
    vert_links = tc.vertical_active_links
    east_nodes_at_node = tc.node_east[core]
    west_nodes_at_node = tc.node_west[core]
    north_nodes_at_node = tc.node_north[core]
    south_nodes_at_node = tc.node_south[core]
    east_link_at_node = tc.east_link_at_node[core]
    west_link_at_node = tc.west_link_at_node[core]
    north_link_at_node = tc.north_link_at_node[core]
    south_link_at_node = tc.south_link_at_node[core]
    east_nodes_at_link = tc.east_node_at_horizontal_link[horiz_links]
    west_nodes_at_link = tc.west_node_at_horizontal_link[horiz_links]
    north_nodes_at_link = tc.north_node_at_vertical_link[vert_links]
    south_nodes_at_link = tc.south_node_at_vertical_link[vert_links]
    p_w = tc.p_w

    ############################
    # find wet nodes and links #
    ############################
    # tc.wet_nodes = core[np.where(p[core] > p_w)]
    # tc.wet_horizontal_links = horiz_links[np.where(
    #     (p[west_nodes_at_link] > p_w) & (p[east_nodes_at_link] > p_w))]
    # tc.wet_vertical_links = vert_links[np.where(
    #     (p[north_nodes_at_link] > p_w) & (p[south_nodes_at_link] > p_w))]

    wet_nodes = (p > p_w) & (tc.h > 0.001)
    tc.wet_nodes = core[wet_nodes[core]]
    tc.wet_horizontal_links = horiz_links[np.where(
        (wet_nodes[west_nodes_at_link]) & (wet_nodes[east_nodes_at_link]))]
    tc.wet_vertical_links = vert_links[np.where(
        (wet_nodes[north_nodes_at_link]) & (wet_nodes[south_nodes_at_link]))]

    ######################################################
    #find partial wet nodes and links in horizontal axis #
    ######################################################
    wet_at_east = np.where(~(wet_nodes[core])
                           & (wet_nodes[east_nodes_at_node]))
    horizontally_partial_wet_nodes_E = core[wet_at_east]
    horizontally_wettest_nodes_E = east_nodes_at_node[wet_at_east]
    partial_wet_horizontal_links_E = east_link_at_node[wet_at_east]
    horizontal_direction_wettest_E = -1.0 * np.ones(wet_at_east[0].shape)

    wet_at_west = np.where(~(wet_nodes[core])
                           & (wet_nodes[west_nodes_at_node]))
    horizontally_partial_wet_nodes_W = core[wet_at_west]
    horizontally_wettest_nodes_W = west_nodes_at_node[wet_at_west]
    partial_wet_horizontal_links_W = west_link_at_node[wet_at_west]
    horizontal_direction_wettest_W = 1.0 * np.ones(wet_at_west[0].shape)

    tc.horizontally_partial_wet_nodes = np.concatenate(
        [horizontally_partial_wet_nodes_E, horizontally_partial_wet_nodes_W])
    tc.horizontally_wettest_nodes = np.concatenate(
        [horizontally_wettest_nodes_E, horizontally_wettest_nodes_W])
    tc.partial_wet_horizontal_links = np.concatenate(
        [partial_wet_horizontal_links_E, partial_wet_horizontal_links_W])
    tc.horizontal_direction_wettest = np.concatenate(
        [horizontal_direction_wettest_E, horizontal_direction_wettest_W])

    ######################################################
    # find partial wet nodes and links in vertical axis  #
    ######################################################
    # vertical partial wet check
    wet_at_north = np.where(~(wet_nodes[core])
                            & (wet_nodes[north_nodes_at_node]))
    vertically_partial_wet_nodes_N = core[wet_at_north]
    vertically_wettest_nodes_N = north_nodes_at_node[wet_at_north]
    partial_wet_vertical_links_N = north_link_at_node[wet_at_north]
    vertical_direction_wettest_N = -1.0 * np.ones(wet_at_north[0].shape)

    wet_at_south = np.where(~(wet_nodes[core])
                            & (wet_nodes[south_nodes_at_node]))
    vertically_partial_wet_nodes_S = core[wet_at_south]
    vertically_wettest_nodes_S = south_nodes_at_node[wet_at_south]
    partial_wet_vertical_links_S = south_link_at_node[wet_at_south]
    vertical_direction_wettest_S = 1.0 * np.ones(wet_at_south[0].shape)

    # combine north and south
    tc.vertically_partial_wet_nodes = np.concatenate(
        [vertically_partial_wet_nodes_N, vertically_partial_wet_nodes_S])
    tc.vertically_wettest_nodes = np.concatenate(
        [vertically_wettest_nodes_N, vertically_wettest_nodes_S])
    tc.partial_wet_vertical_links = np.concatenate(
        [partial_wet_vertical_links_N, partial_wet_vertical_links_S])
    tc.vertical_direction_wettest = np.concatenate(
        [vertical_direction_wettest_N, vertical_direction_wettest_S])

    #######################################
    # wet and partial wet nodes and links #
    #######################################
    tc.wet_pwet_nodes = np.unique(
        np.concatenate([
            tc.wet_nodes, tc.horizontally_partial_wet_nodes,
            tc.vertically_partial_wet_nodes
        ]))
    tc.wet_pwet_horizontal_links = np.concatenate(
        [tc.wet_horizontal_links, tc.partial_wet_horizontal_links])
    tc.wet_pwet_vertical_links = np.concatenate(
        [tc.wet_vertical_links, tc.partial_wet_vertical_links])
    tc.wet_pwet_links = np.concatenate(
        [tc.wet_pwet_horizontal_links, tc.wet_pwet_vertical_links])

    ############################
    # find dry nodes and links #
    ############################
    tc.dry_nodes = np.setdiff1d(core, tc.wet_pwet_nodes)
    tc.dry_links = np.setdiff1d(
        tc.active_links,
        np.concatenate([tc.wet_pwet_links, tc.fixed_value_links]))


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
    dx = tc.grid.dx
    # empirical coefficient (Homma)
    gamma = tc.gamma

    # grid information
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

    # overspill_velocity_x = gamma * np.sqrt(
    #     2.0 * R * g * Ch[horizontally_wettest_nodes]) / dx * dt
    # overspill_velocity_y = gamma * np.sqrt(
    #     2.0 * R * g * Ch[vertically_wettest_nodes]) / dx * dt

    # M_horiz = h[horizontally_wettest_nodes] * overspill_velocity_x
    # M_vert = h[vertically_wettest_nodes] * overspill_velocity_y
    # CM_horiz = Ch[horizontally_wettest_nodes] * overspill_velocity_x
    # CM_vert = Ch[vertically_wettest_nodes] * overspill_velocity_y

    horizontal_overspill_velocity, M_horiz, CM_horiz = calc_overspill_velocity(
        h[horizontally_wettest_nodes], Ch[horizontally_wettest_nodes], gamma,
        R, g, dx, dt)
    vertical_overspill_velocity, M_vert, CM_vert = calc_overspill_velocity(
        h[vertically_wettest_nodes], Ch[vertically_wettest_nodes], gamma, R, g,
        dx, dt)

    ################################################################
    # Calculate time development of variables at partial wet nodes #
    ################################################################

    # overspilling horizontally
    half_dry = h_out[horizontally_wettest_nodes] < 8.0 * M_horiz
    M_horiz[half_dry] = h_out[horizontally_wettest_nodes][half_dry] / 8.0
    h_out[horizontally_partial_wet_nodes] += M_horiz
    # h_out[horizontally_wettest_nodes] -= M_horiz
    c_half_dry = Ch_out[horizontally_wettest_nodes] < 8.0 * CM_horiz
    CM_horiz[c_half_dry] = Ch_out[horizontally_wettest_nodes][c_half_dry] / 8.
    Ch_out[horizontally_partial_wet_nodes] += CM_horiz
    # Ch_out[horizontally_wettest_nodes] -= CM_horiz

    # overspilling vertically
    half_dry = h_out[vertically_wettest_nodes] < 8.0 * M_vert
    M_vert[half_dry] = h_out[vertically_wettest_nodes][half_dry] / 8.0
    h_out[vertically_partial_wet_nodes] += M_vert
    # h_out[vertically_wettest_nodes] -= M_vert
    c_half_dry = Ch_out[vertically_wettest_nodes] < 8.0 * CM_vert
    CM_vert[c_half_dry] = Ch_out[vertically_wettest_nodes][c_half_dry] / 8.0
    Ch_out[vertically_partial_wet_nodes] += CM_vert
    # Ch_out[vertically_wettest_nodes] -= CM_vert

    ################################################################
    # Calculate time development of variables at partial wet links #
    ################################################################
    CfuU = Cf * np.sqrt(u[partial_wet_horizontal_links]**2 +
                        v[partial_wet_horizontal_links]**2) / (
                            h[horizontally_wettest_nodes] +
                            h[horizontally_partial_wet_nodes]) * 2
    CfvU = Cf * np.sqrt(u[partial_wet_vertical_links]**2 +
                        v[partial_wet_vertical_links]**2) / (
                            h[vertically_wettest_nodes] +
                            h[vertically_partial_wet_nodes]) * 2

    hdw = horizontal_direction_wettest
    vdw = vertical_direction_wettest

    u_out[partial_wet_horizontal_links] = u[partial_wet_horizontal_links]\
                        + hdw * horizontal_overspill_velocity
    u_out[partial_wet_horizontal_links] *= 1 / (1 + CfuU * dt)

    # - CfuU / (h[horizontally_wettest_nodes] / 2) * dt
    # u_out[partial_wet_horizontal_links] = hdw * horizontal_overspill_velocity

    v_out[partial_wet_vertical_links] = v[partial_wet_vertical_links] \
            + vdw * vertical_overspill_velocity
    v_out[partial_wet_vertical_links] *= 1 / (1 + CfvU * dt)

    # - CfvU / (h[vertically_wettest_nodes] / 2) * dt
    # v_out[partial_wet_vertical_links] = vdw * vertical_overspill_velocity

    # tc.horizontal_overspill_velocity[partial_wet_horizontal_links] = h_out[partial_wet_vertical_links]
    # tc.vertical_overspill_velocity[partial_wet_vertical_links] = v_out[partial_wet_vertical_links]

    # h_out[horizontally_partial_wet_nodes] += u_out[
    #     partial_wet_horizontal_links] * h[horizontally_wettest_nodes] * dt
    # Ch_out[horizontally_partial_wet_nodes] += u_out[
    #     partial_wet_horizontal_links] * Ch[horizontally_wettest_nodes] * dt
    # h_out[vertically_partial_wet_nodes] += u_out[
    #     partial_wet_vertical_links] * h[vertically_wettest_nodes] * dt
    # Ch_out[vertically_partial_wet_nodes] += u_out[
    #     partial_wet_vertical_links] * Ch[vertically_wettest_nodes] * dt

    return h_out, u_out, v_out, Ch_out


def calc_overspill_velocity(h, Ch, gamma, R, g, dx, dt):
    """Function to calculate overspilling velocity at flow front by using
       Runge-Kutta method

       Parameters
       -------------
       h : ndarray, float
           Flow depth at wettest nodes

       Ch : ndarray, float
           Sediment volume at wettest nodes

       gamma : double
           Empirical coefficient

       R : double
           Submerged specific density

       g : double
           gravity acceleration

       dx : double
           grid spacing

       dt : double
           time step length


       Returns
       ------------------
       M : ndarray, float
           overspilling velocity of flow depth

       CM : ndarray, float
           overspilling velocity of sediment volume


    """

    ######################################################
    # horizontal and vertical flow discharge between wet #
    # and partial wet nodes                              #
    ######################################################
    overspill_velocity1 = gamma * np.sqrt(2.0 * R * g * Ch)
    M1 = h * overspill_velocity1 / dx * dt
    CM1 = Ch * overspill_velocity1 / dx * dt

    overspill_velocity2 = gamma * np.sqrt(2.0 * R * g * (Ch - CM1 / 2.0))
    M2 = (h - M1 * dt / 2.0) * overspill_velocity2 / dx * dt
    CM2 = (Ch - CM1 * dt / 2.0) * overspill_velocity2 / dx * dt

    overspill_velocity3 = gamma * np.sqrt(2.0 * R * g * (Ch - CM2 / 2.0))
    M3 = (h - M2 * dt / 2.0) * overspill_velocity3 / dx * dt
    CM3 = (Ch - CM2 * dt / 2.0) * overspill_velocity3 / dx * dt

    overspill_velocity4 = gamma * np.sqrt(2.0 * R * g * (Ch - CM3 * dt))
    M4 = (h - M3 * dt / 2.0) * overspill_velocity4 / dx * dt
    CM4 = (Ch - CM3 * dt / 2.0) * overspill_velocity4 / dx * dt

    overspill_velocity = 1 / 6.0 * (
        overspill_velocity1 + 2.0 * overspill_velocity2 +
        2.0 * overspill_velocity3 + overspill_velocity4)
    M = 1 / 6.0 * (M1 + 2.0 * M2 + 2.0 * M3 + M4) * dt
    CM = 1 / 6.0 * (CM1 + 2.0 * CM2 + 2.0 * CM3 + CM4) * dt

    return overspill_velocity, M, CM
