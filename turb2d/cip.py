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
    D = -np.where(u[core] > 0., 1.0, -1.0) * dx
    xi = -u * dt
    BB = np.zeros(D.shape)
    S = (f[up] - f[core]) / D
    dz_index = (S - dfdx[core]) * (dfdx[up] - S) > 0.0
    BB[dz_index] = (np.abs(
        (S[dz_index] - dfdx[core][dz_index]) /
        (dfdx[up][dz_index] - S[dz_index])) - 1.0) / D[dz_index]

    a = (dfdx[core] - S + (dfdx[up] - S) * (1.0 + BB * D)) / (D * D)
    b = S * BB + (S - dfdx[core]) / D - a * D
    c = dfdx[core] + f[core] * BB

    out_f[core] = (((a * xi[core] + b) * xi[core] + c)
                   * xi[core] + f[core]) \
        / (1.0 + BB * xi[core])
    out_dfdx[core] = ((3. * a * xi[core] + 2. * b) * xi[core] + c) \
        / (1.0 + BB * xi[core]) \
        - out_f[core] * BB / (1.0 + BB * xi[core])

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
    rcip_1d_advection(out_f,
                      dfdy,
                      v,
                      core,
                      v_up,
                      dx,
                      dt,
                      out_f=out_f,
                      out_dfdx=out_dfdy)
    D_y = -np.where(v > 0., 1.0, -1.0) * dx
    xi_y = -v * dt
    out_dfdx[core] = out_dfdx[core] - xi_y[core] / \
        D_y[core] * (out_dfdx[core] - out_dfdx[v_up[core]])

    return out_f, out_dfdx, out_dfdy


def shock_dissipation(
        f,
        p,
        core,
        north_id,
        south_id,
        east_id,
        west_id,
        dt,
        kappa2,
        kappa4,
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

    nu_i = np.zeros(n, dtype=np.float)
    eps_i_half2 = np.zeros(n, dtype=np.float)
    eps_i_half4 = np.zeros(n, dtype=np.float)
    d_i_half = np.zeros(n, dtype=np.float)
    north = north_id[core]
    south = south_id[core]
    east = east_id[core]
    west = west_id[core]
    easteast = east_id[east]
    northnorth = north_id[north]

    # First, artificial diffusion is applied to east-west direction
    nu_i[core] = np.abs(p[east] - 2 * p[core] + p[west]) / \
        (np.abs(p[east]) + 2 * np.abs(p[core]) + np.abs(p[west]) + 10**-20)
    eps_i_half2[core] = kappa2 * np.max([nu_i[east], nu_i[core]], axis=0)
    eps_i_half4[core] = np.max(
        [np.zeros_like(core), kappa4 - eps_i_half2[core]], axis=0)
    d_i_half[core] = eps_i_half2[core] * (
        f[east] - f[core]) - eps_i_half4[core] * (f[easteast] - 3.0 * f[east] +
                                                  3.0 * f[core] - f[west])
    out[core] = f[core] + d_i_half[core] - d_i_half[west]

    # Next, artificial diffusion is applied to north-south direction
    nu_i[core] = np.abs(p[north] - 2 * p[core] + p[south]) / (
        np.abs(p[north]) + 2 * np.abs(p[core]) + np.abs(p[south]) + 10**-20)
    eps_i_half2[core] = kappa2 * np.max([nu_i[north], nu_i[core]], axis=0)
    eps_i_half4[core] = np.max(
        [np.zeros_like(core), kappa4 - eps_i_half2[core]], axis=0)
    d_i_half[core] = eps_i_half2[core] * (f[north] - f[core]) - eps_i_half4[
        core] * (f[northnorth] - 3.0 * f[north] + 3.0 * f[core] - f[south])
    out[core] = f[core] + d_i_half[core] - d_i_half[south]

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

    C30 = ((dfdx[xup] + dfdx[core]) * Ddx - 2.0 *
           (f[core] - f[xup])) / (Ddx * Ddx * Ddx)
    C20 = (3.0 * (f[xup] - f[core]) +
           (dfdx[xup] + 2.0 * dfdx[core]) * Ddx) / (Ddx * Ddx)
    C03 = ((dfdy[yup] + dfdy[core]) * Ddy - 2.0 *
           (f[core] - f[yup])) / (Ddy * Ddy * Ddy)
    C02 = (3.0 * (f[yup] - f[core]) +
           (dfdy[yup] + 2.0 * dfdy[core]) * Ddy) / (Ddy * Ddy)
    tmp = f[core] - f[yup] - f[xup] + f[xyup]
    tmq = dfdy[xup] - dfdy[core]
    C12 = (-tmp - tmq * Ddy) / (Ddx * Ddy * Ddy)
    C21 = (-tmp - (dfdx[yup] - dfdx[core]) * Ddx) / (Ddx * Ddx * Ddy)
    C11 = (-tmq + C21 * Ddx * Ddx) / (Ddx)

    out_f[core] = (
        (C30 * XX + C21 * YY + C20) * XX + C11 * YY + dfdx[core]) * XX + (
            (C03 * YY + C12 * XX + C02) * YY + dfdy[core]) * YY + f[core]
    out_dfdx[core] = (3.0 * C30 * XX + 2.0 *
                      (C21 * YY + C20)) * XX + (C12 * YY +
                                                C11) * YY + dfdx[core]
    out_dfdy[core] = (3.0 * C03 * YY + 2.0 *
                      (C12 * XX + C02)) * YY + (C21 * XX +
                                                C11) * XX + dfdy[core]

    return out_f, out_dfdx, out_dfdy


def rcip_2d_advection(f,
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
    """Direct 2D calculation of advection by R-CIP method
    """
    if out_f is None:
        out_f = np.zeros(f.shape[0])
    if out_dfdx is None:
        out_dfdx = np.zeros(f.shape[0])
    if out_dfdy is None:
        out_dfdy = np.zeros(f.shape[0])

    XX = -u[core] * dt
    YY = -v[core] * dt
    Ddx = np.where(u[core] > 0., -1.0, 1.0) * dx
    Ddy = np.where(v[core] > 0., -1.0, 1.0) * dx
    xup = h_up[core]
    yup = v_up[core]
    xyup = (v_up[h_up])[core]

    a01 = np.zeros(core.shape[0])
    a10 = np.zeros(core.shape[0])
    b01 = np.zeros(core.shape[0])
    b10 = np.zeros(core.shape[0])

    Sx = (f[xup] - f[core]) / Ddx
    Sy = (f[yup] - f[core]) / Ddy

    a10 = np.where(dfdx[core] * dfdx[xup] < 0, 1.0, 0.0)
    a01 = np.where(dfdy[core] * dfdy[yup] < 0, 1.0, 0.0)

    b10 = (np.abs(
        (Sx - dfdx[core]) / (dfdx[xup] - Sx + 1.0 * 10**-10)) - 1) / Ddx
    b01 = (np.abs(
        (Sy - dfdy[core]) / (dfdy[yup] - Sy + 1.0 * 10**-10)) - 1) / Ddy

    C00 = f[core]
    C10 = dfdx[core] + a10 * b10 * C00
    C01 = dfdy[core] + a01 * b01 * C00

    # C30 = ((dfdx[xup] + dfdx[core]) * Ddx - 2.0 *
    #        (f[core] - f[xup])) / (Ddx * Ddx * Ddx)
    C30 = ((1 + a10 * b10 * Ddx) *
           (dfdx[xup] - Sx) + dfdx[core] - Sx) / (Ddx * Ddx)

    # C20 = (3.0 * (f[xup] - f[core]) +
    #        (dfdx[xup] + 2.0 * dfdx[core]) * Ddx) / (Ddx * Ddx)
    C20 = ((1 + a10 * b10 * Ddx) * f[xup] - C00 -
           C10 * Ddx) / (Ddx * Ddx) - C30 * Ddx

    # C03 = ((dfdy[yup] + dfdy[core]) * Ddy - 2.0 *
    #        (f[core] - f[yup])) / (Ddy * Ddy * Ddy)
    C03 = ((1 + a01 * b01 * Ddy) *
           (dfdy[yup] - Sy) + dfdy[core] - Sy) / (Ddy * Ddy)

    # C02 = (3.0 * (f[yup] - f[core]) +
    #        (dfdy[yup] + 2.0 * dfdy[core]) * Ddy) / (Ddy * Ddy)
    C02 = ((1 + a01 * b01 * Ddy) * f[yup] - C00 -
           C01 * Ddy) / (Ddy * Ddy) - C03 * Ddy

    # tmp = f[core] - f[yup] - f[xup] + f[xyup]
    # tmq = dfdy[xup] - dfdy[core]
    C11 = (a01 * b01 * f[xup] + (1 + a10 * b10 * Ddx) * dfdy[xup]) / Ddx \
          + (a10 * b10 * f[yup] + (1 + a01 * b01 * Ddy) * dfdx[yup]) / Ddy \
          + (C00 - (a10 * b10 * Ddx + a01 * b01 * Ddy) * f[xyup]) / Ddx / Ddy\
          + C30 * Ddx * Ddx / Ddy + C03 * Ddy * Ddy / Ddx \
          + C20 * Ddx / Ddy + C02 * Ddy / Ddx

    # C12 = (-tmp - tmq * Ddy) / (Ddx * Ddy * Ddy)
    C12 = (a10 * b10 * f[yup] +
           (1 + a01 * b01 * Ddy) * dfdx[yup] - C10) / (Ddy * Ddy) - C11 / Ddy

    # C21 = (-tmp - (dfdx[yup] - dfdx[core]) * Ddx) / (Ddx * Ddx * Ddy)
    C21 = (a01 * b01 * f[xup] +
           (1 + a10 * b10 * Ddx) * dfdy[xup] - C01) / (Ddx * Ddx) - C11 / Ddx

    # C11 = (-tmq + C21 * Ddx * Ddx) / (Ddx)

    out_f[core] = (
        (C30 * XX + C21 * YY + C20) * XX + C11 * YY + dfdx[core]) * XX + (
            (C03 * YY + C12 * XX + C02) * YY + dfdy[core]) * YY + f[core]
    out_dfdx[core] = (3.0 * C30 * XX + 2.0 *
                      (C21 * YY + C20)) * XX + (C12 * YY +
                                                C11) * YY + dfdx[core]
    out_dfdy[core] = (3.0 * C03 * YY + 2.0 *
                      (C12 * XX + C02)) * YY + (C21 * XX +
                                                C11) * XX + dfdy[core]

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
    BB = np.zeros(core.shape, dtype=float)
    S = (f[iminus] - f[iplus]) / D
    dz_index = (S - dfdx[iplus]) * (dfdx[iminus] - S) > 0.0
    BB[dz_index] = (np.abs((S[dz_index] - dfdx[iplus][dz_index]) /
                           (dfdx[iminus][dz_index] - S[dz_index])) - 1.0) / D

    a = (dfdx[iplus] - S + (dfdx[iminus] - S) * (1.0 + BB * D)) / (D**2)
    b = S * BB + (S - dfdx[iplus]) / D - a * D
    c = dfdx[iplus] + f[iplus] * BB

    out[core] = (((a * xi + b) * xi + c)
                   * xi + f[iplus]) \
        / (1.0 + BB * xi)

    # adjust negative values
    negative_value = out[core] < 0
    out[core][negative_value] = (f[iplus][negative_value] +
                                 f[iminus][negative_value]) / 2.0

    return out


def forester_filter(f,
                    core,
                    east_id,
                    west_id,
                    north_id,
                    south_id,
                    out_f=None,
                    loop=1000):
    """ Forester filter for removing negative values from Concentration and 
        Flow depth
    """

    if out_f is None:
        out_f = np.zeros(f.shape[0])

    out_f[:] = f[:]

    east = east_id[core]
    west = west_id[core]
    north = north_id[core]
    south = south_id[core]

    out_f[core] += (f[east] + f[west] + f[north] + f[south] -
                    4.0 * f[core]) / 4.0
    out_f[east] -= (f[east] - f[core]) / 4.0
    out_f[west] -= (f[west] - f[core]) / 4.0
    out_f[north] -= (f[north] - f[core]) / 4.0
    out_f[south] -= (f[south] - f[core]) / 4.0

    return out_f
