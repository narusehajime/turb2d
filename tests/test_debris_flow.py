"""
Unit tests for marslobes.DebrisFlow
last updated: 8/21/2018
"""

import numpy as np
from matplotlib import pyplot as plt
from landlab import RasterModelGrid
from marslobes.debris_flow import DebrisFlow

(_SHAPE, _SPACING, _ORIGIN) = ((10, 10), (25, 25), (0., 0.))
_ARGS = (_SHAPE, _SPACING, _ORIGIN)


def test_dflow_name(dflow):
    assert dflow.name == 'DebrisFlow'


def test_dflow_cf(dflow):
    assert dflow.cf == 0.004


def test_dflow_calc_time_step(dflow):
    dflow.u = np.ones(dflow.grid.at_node['flow__depth'].shape)
    dflow.v = 10.0 * np.ones(dflow.grid.at_node['flow__depth'].shape)
    dx = dflow.grid.dx
    alpha = dflow.alpha
    dflow.calc_time_step()
    assert dflow.dt == alpha * dx / 10.0


def test_dflow_neighbor(dflow):
    dflow.set_up_neighbor_arrays()
    print(dflow.core_nodes)
    print(dflow.node_east)
    print(dflow.active_links)
    print(dflow.link_west)
    core_nodes_correct = np.array([
        11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32,
        33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 51, 52, 53, 54,
        55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76,
        77, 78, 81, 82, 83, 84, 85, 86, 87, 88
    ])
    node_east_correct = np.array([
        1, 2, 3, 4, 5, 6, 7, 8, 9, -1, 11, 12, 13, 14, 15, 16, 17, 18, 19, -1,
        21, 22, 23, 24, 25, 26, 27, 28, 29, -1, 31, 32, 33, 34, 35, 36, 37, 38,
        39, -1, 41, 42, 43, 44, 45, 46, 47, 48, 49, -1, 51, 52, 53, 54, 55, 56,
        57, 58, 59, -1, 61, 62, 63, 64, 65, 66, 67, 68, 69, -1, 71, 72, 73, 74,
        75, 76, 77, 78, 79, -1, 81, 82, 83, 84, 85, 86, 87, 88, 89, -1, 91, 92,
        93, 94, 95, 96, 97, 98, 99, -1
    ])
    active_links_correct = np.array([
        10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29,
        30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49,
        50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69,
        70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89,
        90, 91, 92, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 105, 106, 107,
        108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122,
        124, 125, 126, 127, 128, 129, 130, 131, 133, 134, 135, 136, 137, 138,
        139, 140, 141, 143, 144, 145, 146, 147, 148, 149, 150, 152, 153, 154,
        155, 156, 157, 158, 159, 160, 162, 163, 164, 165, 166, 167, 168, 169
    ])
    link_west_correct = np.array([
        -1, 0, 1, 2, 3, 4, 5, 6, 7, -1, 9, 10, 11, 12, 13, 14, 15, 16, 17, -1,
        19, 20, 21, 22, 23, 24, 25, 26, -1, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        -1, 38, 39, 40, 41, 42, 43, 44, 45, -1, 47, 48, 49, 50, 51, 52, 53, 54,
        55, -1, 57, 58, 59, 60, 61, 62, 63, 64, -1, 66, 67, 68, 69, 70, 71, 72,
        73, 74, -1, 76, 77, 78, 79, 80, 81, 82, 83, -1, 85, 86, 87, 88, 89, 90,
        91, 92, 93, -1, 95, 96, 97, 98, 99, 100, 101, 102, -1, 104, 105, 106,
        107, 108, 109, 110, 111, 112, -1, 114, 115, 116, 117, 118, 119, 120,
        121, -1, 123, 124, 125, 126, 127, 128, 129, 130, 131, -1, 133, 134,
        135, 136, 137, 138, 139, 140, -1, 142, 143, 144, 145, 146, 147, 148,
        149, 150, -1, 152, 153, 154, 155, 156, 157, 158, 159, -1, 161, 162,
        163, 164, 165, 166, 167, 168, 169, -1, 171, 172, 173, 174, 175, 176,
        177, 178
    ])

    assert (dflow.core_nodes == core_nodes_correct).all()
    assert (dflow.node_east == node_east_correct).all()
    assert (dflow.active_links == active_links_correct).all()
    assert (dflow.link_west == link_west_correct).all()


def test_cip1d(dflow):
    dx = 0.1
    x = np.arange(0, 100, dx)
    x0 = 50
    sigma = 5.0
    f = np.zeros(x.shape)
    # f = 0.5 * np.exp(-(x-x0)**2/2/sigma**2)
    f[np.where((40 < x) & (x < 60))] = 1.0
    # dfdx = - (x - x0) * f / sigma ** 2

    dt = 0.02
    dfdx = np.append(np.append(0, (f[2:] - f[:-2]) / (dx * 2)), 0)
    u = 1.0 * np.ones(f.shape)
    u[np.where(x < x0)] = -1.0
    core = range(1, len(u) - 2)

    for i in range(np.round(20.0 / dt).astype(np.int64)):
        G = np.append(np.append(0, -f[1:-1] * (u[2:] - u[:-2]) / (dx * 2)), 0)
        up_id, down_id = dflow.find_up_down_index(f, u)
        f, dfdx = dflow.cip_1d(f, dfdx, G, u, core, up_id, down_id, dx, dt)

    exact_result = np.zeros(x.shape)
    exact_result[(x > 20) & (x < 30)] = 1.0
    exact_result[(x > 70) & (x < 80)] = 1.0
    comparison = np.sum((exact_result - f)**2) / len(f)
    # plt.plot(x, exact_result)
    # plt.plot(x, f)
    # plt.show()
    np.testing.assert_allclose(comparison, 0, atol=0.01)
