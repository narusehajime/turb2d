import numpy as np


def cip_2d_M_advection(f,
                       dfdx,
                       dfdy,
                       u,
                       v,
                       core,
                       h_up,
                       h_down,
                       v_up,
                       v_down,
                       dx,
                       dt,
                       out_f=None,
                       out_dfdx=None,
                       out_dfdy=None):
    """Calculate one time step using M-type 2D cip method
    """

    # First, the variables out and temp are allocated to
    # store the calculation results

    if out_f is None:
        out_f = np.empty(f.shape)
    if out_dfdx is None:
        out_dfdx = np.empty(dfdx.shape)
    if out_dfdy is None:
        out_dfdy = np.empty(dfdy.shape)

    # 1st step for horizontal advection
    D_x = -np.where(u > 0., 1.0, -1.0) * dx
    xi_x = -u * dt
    a = (dfdx[core] + dfdx[h_up]) / (D_x[core] ** 2)\
        + 2 * (f[core] - f[h_up]) / (D_x[core] ** 3)
    b = 3 * (f[h_up] - f[core]) / (D_x[core] ** 2)\
        - (2 * dfdx[core] + dfdx[h_up]) / D_x[core]
    out_f[core] = a * (xi_x[core] ** 3) + b * (xi_x[core] ** 2)\
        + dfdx[core] * xi_x[core] + f[core]
    out_dfdx[core] = 3 * a * (xi_x[core] ** 2) + 2 * b * xi_x[core]\
        + dfdx[core]
    out_dfdy[core] = dfdy[core] - xi_x[core] / \
        D_x[core] * (dfdy[core] - dfdy[h_up])

    # 2nd step for vertical advection
    D_y = -np.where(v > 0., 1.0, -1.0) * dx
    xi_y = -v * dt
    a = (out_dfdy[core] + out_dfdy[v_up]) / (D_y[core] ** 2)\
        + 2 * (out_f[core] - out_f[v_up]) / (D_y[core] ** 3)
    b = 3 * (out_f[v_up] - out_f[core]) / (D_y[core] ** 2)\
        - (2 * out_dfdy[core] + out_dfdy[v_up]) / D_y[core]
    out_f[core] = a * (xi_y[core] ** 3) + b * (xi_y[core] ** 2)\
        + out_dfdy[core] * xi_y[core] + out_f[core]
    out_dfdy[core] = 3 * a * (xi_y[core] ** 2) + 2 * b * xi_y[core]\
        + out_dfdy[core]
    out_dfdx[core] = out_dfdx[core] - xi_y[core] / \
        D_y[core] * (out_dfdx[core] - out_dfdx[v_up])

    return out_f, out_dfdx, out_dfdy


def cip_2d_nonadvection(f,
                        dfdx,
                        dfdy,
                        G,
                        u,
                        v,
                        core,
                        h_up,
                        h_down,
                        v_up,
                        v_down,
                        dx,
                        dt,
                        out_f=None,
                        out_dfdx=None,
                        out_dfdy=None):

    if out_f is None:
        out_f = np.zeros(f.shape)
    if out_dfdx is None:
        out_dfdx = np.zeros(dfdx.shape)
    if out_dfdy is None:
        out_dfdy = np.zeros(dfdy.shape)

    D_x = -np.where(u > 0., 1.0, -1.0) * dx
    xi_x = -u * dt
    D_y = -np.where(v > 0., 1.0, -1.0) * dx
    xi_y = -v * dt

    # non-advection term
    out_f[core] = f[core] + G[core] * dt
    out_dfdx[core] = dfdx[core] + ((out_f[h_down] - f[h_down])
                                   - (out_f[h_up] - f[h_up])) / \
        (-2 * D_x[core]) - dfdx[core] * \
        (xi_x[h_down] - xi_x[h_up]) / (2 * D_x[core])

    out_dfdy[core] = dfdy[core]
    +((out_f[v_down] - f[v_down]) -
      (out_f[v_up] - f[v_up])) / (-2 * D_y[core]) - dfdy[core] * (
          xi_y[v_down] - xi_y[v_up]) / (2 * D_y[core])

    return out_f, out_dfdx, out_dfdy
