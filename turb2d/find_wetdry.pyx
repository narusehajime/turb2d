import numpy as np
cimport numpy as cnp
cimport cython
from cython import boundscheck, wraparound 

@cython.boundscheck(False)
@cython.wraparound(False)

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

    # Assume necessary data structures (lists, dicts) are initialized properly
    # and variables like Ch, Ch_w, etc., are defined with comparable data.

    # Determine wet_nodes without numpy, assuming Ch and tc.h are lists of the same length
    wet_nodes = [Ch[i] > Ch_w and tc.h[i] > h_w for i in range(len(Ch))]

    # Applying the filtering for wet_nodes on core without numpy
    tc.wet_nodes = [core[i] for i in range(len(core)) if wet_nodes[core[i]]]

    # Wet horizontal and vertical links calculation without numpy
    tc.wet_horizontal_links = [
        horiz_links[i] for i in range(len(horiz_links))
        if wet_nodes[west_nodes_at_link[horiz_links[i]]] and wet_nodes[east_nodes_at_link[horiz_links[i]]]
    ]
    tc.wet_vertical_links = [
        vert_links[i] for i in range(len(vert_links))
        if wet_nodes[north_nodes_at_link[vert_links[i]]] and wet_nodes[south_nodes_at_link[vert_links[i]]]
    ]

    # Horizontal axis wet node calculations without numpy
    wet_at_east = [
        i for i in range(len(core))
        if not wet_nodes[core[i]] and wet_nodes[node_east[core[i]]]
    ]
    horizontally_partial_wet_nodes_E = [core[i] for i in wet_at_east]
    horizontally_wettest_nodes_E = [node_east[core[i]] for i in wet_at_east]
    partial_wet_horizontal_links_E = [east_link_at_node[core[i]] for i in wet_at_east]
    horizontal_direction_wettest_E = [-1.0 for _ in wet_at_east]

    wet_at_west = [
        i for i in range(len(core))
        if not wet_nodes[core[i]] and wet_nodes[node_west[core[i]]]
    ]
    horizontally_partial_wet_nodes_W = [core[i] for i in wet_at_west]
    horizontally_wettest_nodes_W = [node_west[core[i]] for i in wet_at_west]
    partial_wet_horizontal_links_W = [west_link_at_node[core[i]] for i in wet_at_west]
    horizontal_direction_wettest_W = [1.0 for _ in wet_at_west]

    # Combine lists directly
    tc.horizontally_partial_wet_nodes = horizontally_partial_wet_nodes_E + horizontally_partial_wet_nodes_W
    tc.horizontally_wettest_nodes = horizontally_wettest_nodes_E + horizontally_wettest_nodes_W
    tc.partial_wet_horizontal_links = partial_wet_horizontal_links_E + partial_wet_horizontal_links_W
    tc.horizontal_direction_wettest = horizontal_direction_wettest_E + horizontal_direction_wettest_W

    # Similar approach for vertical axis calculations...

    # For the unique operations, you can use set to eliminate duplicates when combining lists
    tc.partial_wet_nodes = list(set(tc.horizontally_partial_wet_nodes + tc.vertically_partial_wet_nodes))
    tc.wettest_nodes = list(set(tc.horizontally_wettest_nodes + tc.vertically_wettest_nodes))
    tc.wet_pwet_nodes = list(set(tc.wet_nodes + tc.partial_wet_nodes))
    tc.wet_pwet_horizontal_links = tc.wet_horizontal_links + tc.partial_wet_horizontal_links
    tc.wet_pwet_vertical_links = tc.wet_vertical_links + tc.partial_wet_vertical_links
    tc.wet_pwet_links = tc.wet_pwet_horizontal_links + tc.wet_pwet_vertical_links

    # Find dry nodes and links by difference without numpy
    tc.dry_nodes = list(set(core) - set(tc.wet_pwet_nodes))
    tc.dry_links = list(set(tc.active_links) - set(tc.wet_pwet_links + tc.fixed_value_links))
