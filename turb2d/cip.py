import numpy as np


def cip_2d_M_advection(f,
                       dfdx,
                       dfdy,
                       u,
                       v,
                       core,
                       h_up,
                       v_up,
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
    cip_1d_advection(f,
                     dfdx,
                     u,
                     core,
                     h_up,
                     dx,
                     dt,
                     out_f=out_f,
                     out_dfdx=out_dfdx)
    out_dfdy[core] = dfdy[core] - xi_x[core] / \
        D_x[core] * (dfdy[core] - dfdy[h_up[core]])

    # 2nd step for vertical advection
    D_y = -np.where(v > 0., 1.0, -1.0) * dx
    xi_y = -v * dt
    cip_1d_advection(out_f,
                     dfdy,
                     v,
                     core,
                     v_up,
                     dx,
                     dt,
                     out_f=out_f,
                     out_dfdx=out_dfdy)
    out_dfdx[core] = out_dfdx[core] - xi_y[core] / \
        D_y[core] * (out_dfdx[core] - out_dfdx[v_up[core]])

    return out_f, out_dfdx, out_dfdy


def cip_1d_advection(
        f,
        dfdx,
        u,
        core,
        up_id,
        dx,
        dt,
        out_f=None,
        out_dfdx=None,
):
    """Calculate one time step using M-type 2D cip method
    """

    # First, the variables out and temp are allocated to
    # store the calculation results

    if out_f is None:
        out_f = np.empty(f.shape)
    if out_dfdx is None:
        out_dfdx = np.empty(dfdx.shape)

    up = up_id[core]

    # 1st step for horizontal advection
    D_x = -np.where(u > 0., 1.0, -1.0) * dx
    xi_x = -u * dt
    a = (dfdx[core] + dfdx[up]) / (D_x[core] ** 2)\
        + 2.0 * (f[core] - f[up]) / (D_x[core] ** 3)
    b = 3.0 * (f[up] - f[core]) / (D_x[core] ** 2)\
        - (2.0 * dfdx[core] + dfdx[up]) / D_x[core]
    out_f[core] = a * (xi_x[core] ** 3) + b * (xi_x[core] ** 2)\
        + dfdx[core] * xi_x[core] + f[core]
    out_dfdx[core] = 3.0 * a * (xi_x[core] ** 2) + 2.0 * b * xi_x[core]\
        + dfdx[core]

    return out_f, out_dfdx


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


def cip_2d_diffusion(u,
                     v,
                     nu_t,
                     h_active,
                     v_active,
                     north,
                     south,
                     east,
                     west,
                     dx,
                     dt,
                     out_u=None,
                     out_v=None):
    """Caclulate horizontal and vertical diffusion of velocities u and v
    """
    if out_u is None:
        out_u = np.zeros(u.shape)
    if out_v is None:
        out_v = np.zeros(v.shape)

    out_u[h_active] = u[h_active] \
               + nu_t[h_active] * dt * (
               (u[east][h_active] - 2 * u[h_active] + u[west][h_active])
                + (u[north][h_active] - 2 * u[h_active] + u[south][h_active]))\
               / dx**2

    out_v[v_active] = v[v_active] \
               + nu_t[v_active] * dt * (
               (v[east][v_active] - 2 * v[v_active] + v[west][v_active])
                + (v[north][v_active] - 2 * v[v_active] + v[south][v_active]))\
                / dx**2

    return out_u, out_v


def rcip_1d_advection(f,
                      dfdx,
                      u,
                      core,
                      up_id,
                      dx,
                      dt,
                      out_f=None,
                      out_dfdx=None):
    """ calculate 1 step of advection phase by rational function
        CIP method.

        Parameters
        ----------------
        f : ndarray
            variable to be calculated

        dfdx : ndarray
            spatial gradient of the parameter f

        u : ndarray
            advection velocity of f

        core : ndarray
            indeces of core grids

        up : ndarray
            indeces of grids that locate upstream

        down : ndarray
            indeces of grids that locate downstream

        dx : float
            spatial grid spacing

        dt : float
            time step length

        out_f : ndarray
            resultant value of f

        out_dfdx : ndarray
            resultant value of dfdx

        Returns
        --------------------
        out_f : ndarray
            output value of f

        out_dfdx : ndarray
            output value of dfdx

    """

    if out_f is None:
        out_f = np.zeros(f.shape)
    if out_dfdx is None:
        out_dfdx = np.zeros(f.shape)

    up = up_id[core]

    # advection phase
    D = -np.where(u > 0., 1.0, -1.0) * dx
    xi = -u * dt
    BB = np.ones(D[core].shape)
    alpha = np.zeros(D[core].shape)
    S = (f[up] - f[core]) / D[core]
    dz_index = (dfdx[up] - S) == 0.0
    BB[dz_index] = -1.0 / D[core][dz_index]
    BB[~dz_index] = (np.abs((S[~dz_index] - dfdx[core][~dz_index]) /
                            (dfdx[up][~dz_index] - S[~dz_index] + 1.e-10)) -
                     1.0) / D[core][~dz_index]
    alpha[(S - dfdx[core]) / (dfdx[up] - S + 1.e-10) >= 0.0] = 1.0

    a = (dfdx[core] - S + (dfdx[up] - S) *
         (1.0 + alpha * BB * D[core])) / (D[core]**2)
    b = S * alpha * BB + (S - dfdx[core]) / D[core] - a * D[core]
    c = dfdx[core] + f[core] * alpha * BB

    out_f[core] = (((a * xi[core] + b) * xi[core] + c)
                   * xi[core] + f[core]) \
        / (1.0 + alpha * BB * xi[core])
    out_dfdx[core] = ((3. * a * xi[core] + 2. * b) * xi[core] + c) \
        / (1.0 + alpha * BB * xi[core]) \
        - out_f[core] * alpha * BB / (1.0 + alpha * BB * xi[core])

    return out_f, out_dfdx


def rcip_2d_M_advection(f,
                        dfdx,
                        dfdy,
                        u,
                        v,
                        core,
                        h_up,
                        v_up,
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
    # D_x = -np.where(u > 0., 1.0, -1.0) * dx
    # xi_x = -u * dt
    # BB_x = np.ones(D_x[core].shape)
    # alpha = np.zeros(D_x[core].shape)
    # S_x = (f[h_up] - f[core]) / D_x[core]
    # dz_index = (dfdx[h_up] - S_x) == 0.0
    # BB_x[dz_index] = -1.0 / D_x[core][dz_index]
    # BB_x[~dz_index] = (np.abs(
    #     (S_x[~dz_index] - dfdx[core][~dz_index]) /
    #     (dfdx[h_up][~dz_index] - S_x[~dz_index] + 1.e-10)) -
    #                    1.0) / D_x[core][~dz_index]
    # alpha[(S_x - dfdx[core]) / (dfdx[h_up] - S_x + 1.e-10) >= 0.0] = 1.0

    # a = (dfdx[core] - S_x +
    #          (dfdx[h_up] - S_x) * (1.0 + alpha * BB_x * D_x[core])) \
    #          / (D_x[core]**2)
    # b = S_x * alpha * BB_x + (S_x - dfdx[core]) / D_x[core] - a * D_x[core]
    # c = dfdx[core] + f[core] * alpha * BB_x

    # out_f[core] = (((a * xi_x[core] + b) * xi_x[core] + c)
    #                * xi_x[core] + f[core]) \
    #     / (1.0 + alpha * BB_x * xi_x[core])
    # out_dfdx[core] = ((3. * a * xi_x[core] + 2. * b) * xi_x[core] + c) \
    #         / (1.0 + alpha * BB_x * xi_x[core]) \
    #         - out_f[core] * alpha * BB_x / (1.0 + alpha * BB_x * xi_x[core])
    rcip_1d_advection(f,
                      dfdx,
                      u,
                      core,
                      h_up,
                      dx,
                      dt,
                      out_f=out_f,
                      out_dfdx=out_dfdx)
    D_x = -np.where(u > 0., 1.0, -1.0) * dx
    xi_x = -u * dt
    out_dfdy[core] = dfdy[core] - xi_x[core] / \
        D_x[core] * (dfdy[core] - dfdy[h_up[core]])

    # 2nd step for vertical advection
    D_y = -np.where(v > 0., 1.0, -1.0) * dx
    xi_y = -v * dt
    # BB_y = np.ones(D_y[core].shape)
    # alpha = np.zeros(D_y[core].shape)
    # S_y = (out_f[v_up] - out_f[core]) / D_y[core]
    # dz_index = (out_dfdy[v_up] - S_y) == 0.0
    # BB_y[dz_index] = -1.0 / D_y[core][dz_index]
    # BB_y[~dz_index] = (np.abs(
    #     (S_y[~dz_index] - out_dfdy[core][~dz_index]) /
    #     (out_dfdy[v_up][~dz_index] - S_y[~dz_index] + 1.e-10)) -
    #                    1.0) / D_y[core][~dz_index]
    # alpha[(S_y - out_dfdy[core]) /
    #       (out_dfdy[v_up] - S_y + 1.e-10) >= 0.0] = 1.0

    # a = (out_dfdy[core] - S_y +
    #          (out_dfdy[v_up] - S_y) * (1.0 + alpha * BB_y * D_y[core])) \
    #          / (D_y[core]**2)
    # b = S_y * alpha * BB_y + (S_y - out_dfdy[core]) / D_y[core] - a * D_y[core]
    # c = out_dfdy[core] + out_f[core] * alpha * BB_y

    # out_f[core] = (((a * xi_y[core] + b) * xi_y[core] + c)
    #                * xi_y[core] + out_f[core]) \
    #     / (1.0 + alpha * BB_y * xi_y[core])
    # out_dfdy[core] = ((3. * a * xi_y[core] + 2. * b) * xi_y[core] + c) \
    #         / (1.0 + alpha * BB_y * xi_y[core]) \
    #         - out_f[core] * alpha * BB_y / (1.0 + alpha * BB_y * xi_y[core])

    rcip_1d_advection(out_f,
                      dfdy,
                      v,
                      core,
                      v_up,
                      dx,
                      dt,
                      out_f=out_f,
                      out_dfdx=out_dfdy)
    out_dfdx[core] = out_dfdx[core] - xi_y[core] / \
        D_y[core] * (out_dfdx[core] - out_dfdx[v_up[core]])

    return out_f, out_dfdx, out_dfdy


def shock_dissipation(
        f,
        h,
        core,
        north_id,
        south_id,
        east_id,
        west_id,
        dt,
        kappa,
        out=None,
):
    """ adding artificial viscosity for numerical stability

        Parameters
        ------------------
        f : ndarray, float
            parameter for which the artificial viscosity is applied
        h : ndarray, float
            flow height
        core : ndarray, int
            indeces of core nodes or links
        north_id : ndarray, int
            indeces of nodes or links that locate north of core
        south_id : ndarray, int
            indeces of nodes or links that locate south of core
        east_id : ndarray, int
            indeces of nodes or links that locate east of core
        west_id : ndarray, int
            indeces of nodes or links that locate west of core
    """
    n = f.shape[0]

    if out is None:
        out = np.zeros(n)

    eps_i = np.zeros(n, dtype=np.float)
    eps_i_half = np.zeros(n, dtype=np.float)
    north = north_id[core]
    south = south_id[core]
    east = east_id[core]
    west = west_id[core]

    # First, artificial diffusion is applied to east-west direction
    eps_i[core] = kappa * np.abs(h[east] - 2 * h[core] + h[west]) / \
        (np.abs(h[east]) + 2 * np.abs(h[core]) + np.abs(h[west]) + 10**-20)
    eps_i_half[core] = np.max([eps_i[east], eps_i[core]], axis=0)
    out[core] = f[core] + eps_i_half[core] * (f[east] - f[core]) - \
        eps_i_half[west] * (f[core] - f[west])

    # Next, artificial diffusion is applied to north-south direction
    eps_i[core] = kappa * np.abs(h[north] - 2 * h[core] + h[south]) / (
        np.abs(h[north]) + 2 * np.abs(h[core]) + np.abs(h[south]) + 10**-20)
    eps_i_half[core] = np.max([eps_i[north], eps_i[core]], axis=0)
    out[core] = out[core] + (eps_i_half[core] *
                             (out[north] - out[core]) - eps_i_half[south] *
                             (out[core] - out[south]))

    return out


def update_gradient(f,
                    f_new,
                    dfdx,
                    dfdy,
                    core,
                    north,
                    south,
                    east,
                    west,
                    dx,
                    dt,
                    out_dfdx=None,
                    out_dfdy=None):
    """Update gradients when main variables are updated
    """

    if out_dfdx is None:
        out_dfdx = np.zeros(dfdx.shape[0], dtype=np.float)
    if out_dfdy is None:
        out_dfdx = np.zeros(dfdy.shape[0], dtype=np.float)

    # non-advection term
    out_dfdx[core] = dfdx[core] + ((f_new[east] - f[east]) -
                                   (f_new[west] - f[west])) / (2 * dx)

    out_dfdy[core] = dfdy[core] + ((f_new[north] - f[north]) -
                                   (f_new[south] - f[south])) / (2 * dx)


def update_gradient2(f,
                     dfdx,
                     dfdy,
                     u,
                     v,
                     core,
                     north,
                     south,
                     east,
                     west,
                     dx,
                     dt,
                     out_dfdx=None,
                     out_dfdy=None):
    """Update gradients when main variables are updated
    """

    if out_dfdx is None:
        out_dfdx = np.zeros(dfdx.shape[0], dtype=np.float)
    if out_dfdy is None:
        out_dfdx = np.zeros(dfdy.shape[0], dtype=np.float)

    # non-advection term
    out_dfdx[core] = dfdx[core] - ((f[east] - f[west]) * (u[east] - u[west]) /
                                   (2 * dx)**2 + (f[north] - f[south]) *
                                   (v[east] - v[west]) / (2 * dx)**2) * dt
    out_dfdy[core] = dfdy[core] - ((f[east] - f[west]) *
                                   (u[north] - u[south]) / (2 * dx)**2 +
                                   (f[north] - f[south]) *
                                   (v[north] - v[south]) / (2 * dx)**2) * dt
    # out_dfdx[core] = dfdx[core] - (dfdx[core] * (u[east] - u[west]) /
    #                                (2 * dx) + dfdy[core] *
    #                                (v[east] - v[west]) / (2 * dx)) * dt
    # out_dfdy[core] = dfdy[core] - (dfdx[core] * (u[north] - u[south]) /
    #                                (2 * dx) + dfdy[core] *
    #                                (v[north] - v[south]) / (2 * dx)) * dt


def tangentf(f, out_f=None):
    """Convert the value f to tangent transformed value
    """
    if out_f is None:
        out_f = np.zeros(f.shape[0])

    FAC = 0.95
    out_f = np.tan((f - 0.5) * FAC * np.pi)

    return out_f


def atangentf(f, out_f=None):
    """Reconstruct the value f from the tangent transformed value
    """
    if out_f is None:
        out_f = np.zeros(f.shape[0])

    FAC = 0.95
    out_f = np.arctan(f) / (FAC * np.pi) + 0.5

    return out_f


def cip_2d_advection(f,
                     dfdx,
                     dfdy,
                     u,
                     v,
                     core,
                     h_up,
                     v_up,
                     dx,
                     dt,
                     out_f=None,
                     out_dfdx=None,
                     out_dfdy=None):
    """Direct 2D calculation of advection by CIP method
    """
    if out_f is None:
        out_f = np.zeros(f.shape[0])
    if out_dfdx is None:
        out_dfdx = np.zeros(f.shape[0])
    if out_dfdy is None:
        out_dfdy = np.zeros(f.shape[0])

    XX = -u[core] * dt
    YY = -v[core] * dt
    Ddx = np.where(u[core] > 0., 1.0, -1.0) * dx
    Ddy = np.where(v[core] > 0., 1.0, -1.0) * dx
    xup = h_up[core]
    yup = v_up[core]
    xyup = (v_up[h_up])[core]

    a1 = ((dfdx[xup] + dfdx[core]) * Ddx - 2.0 *
          (f[core] - f[xup])) / (Ddx * Ddx * Ddx)
    e1 = (3.0 * (f[xup] - f[core]) +
          (dfdx[xup] + 2.0 * dfdx[core]) * Ddx) / (Ddx * Ddx)
    b1 = ((dfdy[yup] + dfdy[core]) * Ddy - 2.0 *
          (f[core] - f[yup])) / (Ddy * Ddy * Ddy)
    f1 = (3.0 * (f[yup] - f[core]) +
          (dfdy[yup] + 2.0 * dfdy[core]) * Ddy) / (Ddy * Ddy)
    tmp = f[core] - f[yup] - f[xup] + f[xyup]
    tmq = dfdy[xup] - dfdy[core]
    d1 = (-tmp - tmq * Ddy) / (Ddx * Ddy * Ddy)
    c1 = (-tmp - (dfdx[yup] - dfdx[core]) * Ddx) / (Ddx * Ddx * Ddy)
    g1 = (-tmq + c1 * Ddx * Ddx) / (Ddx)

    out_f[core] = (
        (a1 * XX + c1 * YY + e1) * XX + g1 * YY + dfdx[core]) * XX + (
            (b1 * YY + d1 * XX + f1) * YY + dfdy[core]) * YY + f[core]
    out_dfdx[core] = (3.0 * a1 * XX + 2.0 *
                      (c1 * YY + e1)) * XX + (d1 * YY + g1) * YY + dfdx[core]
    out_dfdy[core] = (3.0 * b1 * YY + 2.0 *
                      (d1 * XX + f1)) * YY + (c1 * XX + g1) * XX + dfdy[core]

    return out_f, out_dfdx, out_dfdy


def cubic_interp_1d(f, dfdx, core, iplus, iminus, dx, out=None):
    """interpolate values to links or nodes by cubic function
       
       Interplated values at the grid between "iplus" and "iminus" is returned.
       
       Parameters
       --------------------------
       f : ndarray, float
           values to be interpolated
       dfdx : ndarray, float
           spatial gradient of f
       iplus : ndarray, int
           grid id of (i + dx / 2)
       iminus : ndarray, int
           grid id of (i - dx / 2)
       dx : ndarray, float
           grid spacing
       out : ndarray, float
            interpolated values between grids of iplus and iminus
    """
    if out is None:
        out = np.empty(iplus.shape)

    # interplation by cubic function
    D_x = -dx
    xi_x = -dx / 2.
    a = (dfdx[iplus] + dfdx[iminus]) / (D_x ** 2)\
        + 2.0 * (f[iplus] - f[iminus]) / (D_x ** 3)
    b = 3.0 * (f[iminus] - f[iplus]) / (D_x ** 2)\
        - (2.0 * dfdx[iplus] + dfdx[iminus]) / D_x
    out[core] = a * (xi_x ** 3) + b * (xi_x ** 2)\
        + dfdx[iplus] * xi_x + f[iplus]

    return out


def rcubic_interp_1d(f, dfdx, core, iplus, iminus, dx, out=None):
    """interpolate values to links or nodes by cubic function
       
       Interplated values at the grid between "iplus" and "iminus" is returned.
       
       Parameters
       --------------------------
       f : ndarray, float
           values to be interpolated
       dfdx : ndarray, float
           spatial gradient of f
       iplus : ndarray, int
           grid id of (i + dx / 2)
       iminus : ndarray, int
           grid id of (i - dx / 2)
       dx : ndarray, float
           grid spacing
       out : ndarray, float
            interpolated values between grids of iplus and iminus
    """
    if out is None:
        out = np.zeros(f.shape)

    # advection phase
    D = -dx
    xi = -dx / 2.
    BB = np.ones(core.shape, dtype=float)
    alpha = np.zeros(core.shape, dtype=float)
    S = (f[iminus] - f[iplus]) / D
    dz_index = (dfdx[iminus] - S) == 0.0
    BB[dz_index] = -1.0 / D
    BB[~dz_index] = (np.abs(
        (S[~dz_index] - dfdx[iplus][~dz_index]) /
        (dfdx[iminus][~dz_index] - S[~dz_index] + 1.e-10)) - 1.0) / D
    alpha[(S - dfdx[iplus]) / (dfdx[iminus] - S + 1.e-10) >= 0.0] = 1.0

    a = (dfdx[iplus] - S + (dfdx[iminus] - S) *
         (1.0 + alpha * BB * D)) / (D**2)
    b = S * alpha * BB + (S - dfdx[iplus]) / D - a * D
    c = dfdx[iplus] + f[iplus] * alpha * BB

    out[core] = (((a * xi + b) * xi + c)
                   * xi + f[iplus]) \
        / (1.0 + alpha * BB * xi)
    negative_value = out[core] < 0
    out[core][negative_value] = (f[iplus][negative_value] +
                                 f[iminus][negative_value]) / 2.0

    # adjust negative values
    negative_id = np.where(out[core] < 0)[0]
    out[core[negative_id]] = (f[iplus[negative_id]] +
                              f[iminus[negative_id]]) / 2.0

    return out
