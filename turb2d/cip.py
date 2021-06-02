import numpy as np


def cip_2d_M_advection(
    f,
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
    out_dfdy=None,
):
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
    D_x = -np.where(u > 0.0, 1.0, -1.0) * dx
    xi_x = -u * dt
    cip_1d_advection(f, dfdx, u, core, h_up, dx, dt,
                     out_f=out_f, out_dfdx=out_dfdx)
    out_dfdy[core] = dfdy[core] - xi_x[core] / D_x[core] * (
        dfdy[core] - dfdy[h_up[core]]
    )

    # 2nd step for vertical advection
    D_y = -np.where(v > 0.0, 1.0, -1.0) * dx
    xi_y = -v * dt
    cip_1d_advection(out_f, dfdy, v, core, v_up, dx,
                     dt, out_f=out_f, out_dfdx=out_dfdy)
    out_dfdx[core] = out_dfdx[core] - xi_y[core] / D_y[core] * (
        out_dfdx[core] - out_dfdx[v_up[core]]
    )

    return out_f, out_dfdx, out_dfdy


def cip_1d_advection(
    f, dfdx, u, core, up_id, dx, dt, out_f=None, out_dfdx=None,
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
    D_x = -np.where(u > 0.0, 1.0, -1.0) * dx
    xi_x = -u * dt
    a = (dfdx[core] + dfdx[up]) / (D_x[core] ** 2) + 2.0 * (f[core] - f[up]) / (
        D_x[core] ** 3
    )
    b = (
        3.0 * (f[up] - f[core]) / (D_x[core] ** 2)
        - (2.0 * dfdx[core] + dfdx[up]) / D_x[core]
    )
    out_f[core] = (
        a * (xi_x[core] ** 3)
        + b * (xi_x[core] ** 2)
        + dfdx[core] * xi_x[core]
        + f[core]
    )
    out_dfdx[core] = 3.0 * a * (xi_x[core] ** 2) + \
        2.0 * b * xi_x[core] + dfdx[core]

    return out_f, out_dfdx


def cip_2d_nonadvection(
    f,
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
    out_dfdy=None,
):

    if out_f is None:
        out_f = np.zeros(f.shape)
    if out_dfdx is None:
        out_dfdx = np.zeros(dfdx.shape)
    if out_dfdy is None:
        out_dfdy = np.zeros(dfdy.shape)

    D_x = -np.where(u > 0.0, 1.0, -1.0) * dx
    xi_x = -u * dt
    D_y = -np.where(v > 0.0, 1.0, -1.0) * dx
    xi_y = -v * dt

    # non-advection term
    out_f[core] = f[core] + G[core] * dt
    out_dfdx[core] = (
        dfdx[core]
        + ((out_f[h_down] - f[h_down]) -
           (out_f[h_up] - f[h_up])) / (-2 * D_x[core])
        - dfdx[core] * (xi_x[h_down] - xi_x[h_up]) / (2 * D_x[core])
    )

    out_dfdy[core] = dfdy[core]
    +((out_f[v_down] - f[v_down]) - (out_f[v_up] - f[v_up])) / (-2 * D_y[core]) - dfdy[
        core
    ] * (xi_y[v_down] - xi_y[v_up]) / (2 * D_y[core])

    return out_f, out_dfdx, out_dfdy


def cip_2d_diffusion(
    u,
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
    out_v=None,
):
    """Caclulate horizontal and vertical diffusion of velocities u and v
    """
    if out_u is None:
        out_u = np.zeros(u.shape)
    if out_v is None:
        out_v = np.zeros(v.shape)

    out_u[h_active] = (
        u[h_active]
        + nu_t[h_active]
        * dt
        * (
            (u[east][h_active] - 2 * u[h_active] + u[west][h_active])
            + (u[north][h_active] - 2 * u[h_active] + u[south][h_active])
        )
        / dx ** 2
    )

    out_v[v_active] = (
        v[v_active]
        + nu_t[v_active]
        * dt
        * (
            (v[east][v_active] - 2 * v[v_active] + v[west][v_active])
            + (v[north][v_active] - 2 * v[v_active] + v[south][v_active])
        )
        / dx ** 2
    )

    return out_u, out_v


def rcip_1d_advection(f, dfdx, u, core, up_id, dx, dt, out_f=None, out_dfdx=None):
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
    D = -np.where(u[core] > 0.0, 1.0, -1.0) * dx
    xi = -u * dt
    BB = np.zeros(D.shape)
    S = (f[up] - f[core]) / D
    dz_index = (S - dfdx[core]) * (dfdx[up] - S) > 0.0
    BB[dz_index] = (
        np.abs(
            (S[dz_index] - dfdx[core][dz_index]) /
            (dfdx[up][dz_index] - S[dz_index])
        )
        - 1.0
    ) / D[dz_index]

    a = (dfdx[core] - S + (dfdx[up] - S) * (1.0 + BB * D)) / (D * D)
    b = S * BB + (S - dfdx[core]) / D - a * D
    c = dfdx[core] + f[core] * BB

    out_f[core] = (((a * xi[core] + b) * xi[core] + c) * xi[core] + f[core]) / (
        1.0 + BB * xi[core]
    )
    out_dfdx[core] = ((3.0 * a * xi[core] + 2.0 * b) * xi[core] + c) / (
        1.0 + BB * xi[core]
    ) - out_f[core] * BB / (1.0 + BB * xi[core])

    return out_f, out_dfdx


def rcip_2d_M_advection(
    f,
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
    out_dfdy=None,
):
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
    rcip_1d_advection(f, dfdx, u, core, h_up, dx, dt,
                      out_f=out_f, out_dfdx=out_dfdx)
    D_x = -np.where(u > 0.0, 1.0, -1.0) * dx
    xi_x = -u * dt
    out_dfdy[core] = dfdy[core] - xi_x[core] / D_x[core] * (
        dfdy[core] - dfdy[h_up[core]]
    )

    # 2nd step for vertical advection
    rcip_1d_advection(
        out_f, dfdy, v, core, v_up, dx, dt, out_f=out_f, out_dfdx=out_dfdy
    )
    D_y = -np.where(v > 0.0, 1.0, -1.0) * dx
    xi_y = -v * dt
    out_dfdx[core] = out_dfdx[core] - xi_y[core] / D_y[core] * (
        out_dfdx[core] - out_dfdx[v_up[core]]
    )

    return out_f, out_dfdx, out_dfdy


def shock_dissipation(
    f, p, core, north_id, south_id, east_id, west_id, dt, kappa2, kappa4, out=None,
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
    nu_j = np.zeros(n, dtype=np.float)
    eps_i_half2 = np.zeros(n, dtype=np.float)
    # eps_i_half4 = np.zeros(n, dtype=np.float)
    eps_j_half2 = np.zeros(n, dtype=np.float)
    # eps_j_half4 = np.zeros(n, dtype=np.float)
    d_i_half = np.zeros(n, dtype=np.float)
    d_j_half = np.zeros(n, dtype=np.float)
    north = north_id[core]
    south = south_id[core]
    east = east_id[core]
    west = west_id[core]
    # easteast = east_id[east]
    # northnorth = north_id[north]

    # First, artificial diffusion is applied to east-west direction
    nu_i[core] = np.abs(p[east] - 2 * p[core] + p[west]) / (
        np.abs(p[east]) + 2 * np.abs(p[core]) + np.abs(p[west]) + 10 ** -20
    )
    eps_i_half2[core] = kappa2 * np.max([nu_i[east], nu_i[core]], axis=0)
    # eps_i_half4[core] = np.max(
    #     [np.zeros_like(core), kappa4 - eps_i_half2[core]], axis=0)
    # d_i_half[core] = eps_i_half2[core] * (
    #     f[east] - f[core]) - eps_i_half4[core] * (f[easteast] - 3.0 * f[east] +
    #                                               3.0 * f[core] - f[west])
    d_i_half[core] = eps_i_half2[core] * (f[east] - f[core])

    # Next, artificial diffusion is applied to north-south direction
    nu_j[core] = np.abs(p[north] - 2 * p[core] + p[south]) / (
        np.abs(p[north]) + 2 * np.abs(p[core]) + np.abs(p[south]) + 10 ** -20
    )
    eps_j_half2[core] = kappa2 * np.max([nu_j[north], nu_j[core]], axis=0)
    # eps_j_half4[core] = np.max(
    #     [np.zeros_like(core), kappa4 - eps_j_half2[core]], axis=0)
    # d_j_half[core] = eps_j_half2[core] * (f[north] - f[core]) - eps_j_half4[
    #     core] * (f[northnorth] - 3.0 * f[north] + 3.0 * f[core] - f[south])
    d_j_half[core] = eps_j_half2[core] * (f[north] - f[core])

    # apply artificial diffusion
    out[core] = (
        f[core] + d_i_half[core] - d_i_half[west] +
        d_j_half[core] - d_j_half[south]
    )

    return out


def update_gradient(
    f,
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
    out_dfdy=None,
):
    """Update gradients when main variables are updated
    """

    if out_dfdx is None:
        out_dfdx = np.zeros(dfdx.shape[0], dtype=np.float)
    if out_dfdy is None:
        out_dfdx = np.zeros(dfdy.shape[0], dtype=np.float)

    # non-advection term
    out_dfdx[core] = dfdx[core] + (
        (f_new[east] - f[east]) - (f_new[west] - f[west])
    ) / (2 * dx)

    out_dfdy[core] = dfdy[core] + (
        (f_new[north] - f[north]) - (f_new[south] - f[south])
    ) / (2 * dx)


def update_gradient2(
    f,
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
    out_dfdy=None,
):
    """Update gradients when main variables are updated
    """

    if out_dfdx is None:
        out_dfdx = np.zeros(dfdx.shape[0], dtype=np.float)
    if out_dfdy is None:
        out_dfdx = np.zeros(dfdy.shape[0], dtype=np.float)

    # non-advection term
    out_dfdx[core] = (
        dfdx[core]
        - (
            (f[east] - f[west]) * (u[east] - u[west]) / (2 * dx) ** 2
            + (f[north] - f[south]) * (v[east] - v[west]) / (2 * dx) ** 2
        )
        * dt
    )
    out_dfdy[core] = (
        dfdy[core]
        - (
            (f[east] - f[west]) * (u[north] - u[south]) / (2 * dx) ** 2
            + (f[north] - f[south]) * (v[north] - v[south]) / (2 * dx) ** 2
        )
        * dt
    )


class CIP2D:
    """ CIP Direct 2D scheme

        parameters
        ---------------------
        max_number_of_grids : int
            maximum number of grids that may be used in this solver
    """

    def __init__(self, max_number_of_grids):

        self.XX = np.empty(max_number_of_grids, dtype=float)
        self.YY = np.empty(max_number_of_grids, dtype=float)
        self.Ddx = np.empty(max_number_of_grids, dtype=float)
        self.Ddy = np.empty(max_number_of_grids, dtype=float)
        self.xup = np.empty(max_number_of_grids, dtype=int)
        self.yup = np.empty(max_number_of_grids, dtype=int)
        self.xyup = np.empty(max_number_of_grids, dtype=int)

        self.C30 = np.empty(max_number_of_grids, dtype=float)
        self.C20 = np.empty(max_number_of_grids, dtype=float)
        self.C03 = np.empty(max_number_of_grids, dtype=float)
        self.C02 = np.empty(max_number_of_grids, dtype=float)
        self.tmp = np.empty(max_number_of_grids, dtype=float)
        self.tmq = np.empty(max_number_of_grids, dtype=float)
        self.C12 = np.empty(max_number_of_grids, dtype=float)
        self.C21 = np.empty(max_number_of_grids, dtype=float)
        self.C11 = np.empty(max_number_of_grids, dtype=float)

    def run(
        self,
        f,
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
        out_dfdy=None,
    ):
        """run this solver to calculate advection transport of a variable f

        parameters
        ----------------------
        f : ndarray, float
           A variable to calculate
        """

        if out_f is None:
            out_f = np.zeros_like(f)
        if out_dfdx is None:
            out_dfdx = np.zeros_like(f)
        if out_dfdy is None:
            out_dfdy = np.zeros_like(f)

        XX = self.XX
        YY = self.YY
        Ddx = self.Ddx
        Ddy = self.Ddy
        xup = self.xup
        yup = self.yup
        xyup = self.xyup

        XX[core] = -u[core] * dt
        YY[core] = -v[core] * dt
        Ddx[core] = np.where(u[core] > 0.0, 1.0, -1.0) * dx
        Ddy[core] = np.where(v[core] > 0.0, 1.0, -1.0) * dx
        xup[core] = h_up[core]
        yup[core] = v_up[core]
        xyup[core] = v_up[h_up[core]]

        tmp = self.tmp
        tmq = self.tmq

        self.C30[core] = (
            (dfdx[xup[core]] + dfdx[core]) *
            Ddx[core] - 2.0 * (f[core] - f[xup[core]])
        ) / (Ddx[core] * Ddx[core] * Ddx[core])
        self.C20[core] = (
            3.0 * (f[xup[core]] - f[core])
            + (dfdx[xup[core]] + 2.0 * dfdx[core]) * Ddx[core]
        ) / (Ddx[core] * Ddx[core])
        self.C03[core] = (
            (dfdy[yup[core]] + dfdy[core]) *
            Ddy[core] - 2.0 * (f[core] - f[yup[core]])
        ) / (Ddy[core] * Ddy[core] * Ddy[core])
        self.C02[core] = (
            3.0 * (f[yup[core]] - f[core])
            + (dfdy[yup[core]] + 2.0 * dfdy[core]) * Ddy[core]
        ) / (Ddy[core] * Ddy[core])
        self.tmp[core] = f[core] - f[yup[core]] - f[xup[core]] + f[xyup[core]]
        self.tmq[core] = dfdy[xup[core]] - dfdy[core]
        self.C12[core] = (-tmp[core] - tmq[core] * Ddy[core]) / (
            Ddx[core] * Ddy[core] * Ddy[core]
        )
        self.C21[core] = (-tmp[core] - (dfdx[yup[core]] - dfdx[core]) * Ddx[core]) / (
            Ddx[core] * Ddx[core] * Ddy[core]
        )
        self.C11[core] = (-tmq[core] + self.C21[core] * Ddx[core] * Ddx[core]) / (
            Ddx[core]
        )

        out_f[core] = (
            (
                (self.C30[core] * XX[core] + self.C21[core]
                 * YY[core] + self.C20[core])
                * XX[core]
                + self.C11[core] * YY[core]
                + dfdx[core]
            )
            * XX[core]
            + (
                (self.C03[core] * YY[core] + self.C12[core]
                 * XX[core] + self.C02[core])
                * YY[core]
                + dfdy[core]
            )
            * YY[core]
            + f[core]
        )
        out_dfdx[core] = (
            (
                3.0 * self.C30[core] * XX[core]
                + 2.0 * (self.C21[core] * YY[core] + self.C20[core])
            )
            * XX[core]
            + (self.C12[core] * YY[core] + self.C11[core]) * YY[core]
            + dfdx[core]
        )
        out_dfdy[core] = (
            (
                3.0 * self.C03[core] * YY[core]
                + 2.0 * (self.C12[core] * XX[core] + self.C02[core])
            )
            * YY[core]
            + (self.C21[core] * XX[core] + self.C11[core]) * XX[core]
            + dfdy[core]
        )

        return out_f, out_dfdx, out_dfdy


def rcip_2d_advection(
    f,
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
    out_dfdy=None,
):
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
    Ddx = np.where(u[core] > 0.0, -1.0, 1.0) * dx
    Ddy = np.where(v[core] > 0.0, -1.0, 1.0) * dx
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

    b10 = (np.abs((Sx - dfdx[core]) /
           (dfdx[xup] - Sx + 1.0 * 10 ** -10)) - 1) / Ddx
    b01 = (np.abs((Sy - dfdy[core]) /
           (dfdy[yup] - Sy + 1.0 * 10 ** -10)) - 1) / Ddy

    C00 = f[core]
    C10 = dfdx[core] + a10 * b10 * C00
    C01 = dfdy[core] + a01 * b01 * C00

    # C30 = ((dfdx[xup] + dfdx[core]) * Ddx - 2.0 *
    #        (f[core] - f[xup])) / (Ddx * Ddx * Ddx)
    C30 = ((1 + a10 * b10 * Ddx) *
           (dfdx[xup] - Sx) + dfdx[core] - Sx) / (Ddx * Ddx)

    # C20 = (3.0 * (f[xup] - f[core]) +
    #        (dfdx[xup] + 2.0 * dfdx[core]) * Ddx) / (Ddx * Ddx)
    C20 = ((1 + a10 * b10 * Ddx) * f[xup] -
           C00 - C10 * Ddx) / (Ddx * Ddx) - C30 * Ddx

    # C03 = ((dfdy[yup] + dfdy[core]) * Ddy - 2.0 *
    #        (f[core] - f[yup])) / (Ddy * Ddy * Ddy)
    C03 = ((1 + a01 * b01 * Ddy) *
           (dfdy[yup] - Sy) + dfdy[core] - Sy) / (Ddy * Ddy)

    # C02 = (3.0 * (f[yup] - f[core]) +
    #        (dfdy[yup] + 2.0 * dfdy[core]) * Ddy) / (Ddy * Ddy)
    C02 = ((1 + a01 * b01 * Ddy) * f[yup] -
           C00 - C01 * Ddy) / (Ddy * Ddy) - C03 * Ddy

    # tmp = f[core] - f[yup] - f[xup] + f[xyup]
    # tmq = dfdy[xup] - dfdy[core]
    C11 = (
        (a01 * b01 * f[xup] + (1.0 + a10 * b10 * Ddx) * dfdy[xup]) / Ddx
        + (a10 * b10 * f[yup] + (1.0 + a01 * b01 * Ddy) * dfdx[yup]) / Ddy
        + (C00 - (1.0 + a10 * b10 * Ddx + a01 * b01 * Ddy)
           * f[xyup]) / Ddx / Ddy
        + C30 * Ddx * Ddx / Ddy
        + C03 * Ddy * Ddy / Ddx
        + C20 * Ddx / Ddy
        + C02 * Ddy / Ddx
    )

    # C12 = (-tmp - tmq * Ddy) / (Ddx * Ddy * Ddy)
    C12 = (a10 * b10 * f[yup] + (1 + a01 * b01 * Ddy) * dfdx[yup] - C10) / (
        Ddy * Ddy
    ) - C11 / Ddy

    # C21 = (-tmp - (dfdx[yup] - dfdx[core]) * Ddx) / (Ddx * Ddx * Ddy)
    C21 = (a01 * b01 * f[xup] + (1 + a10 * b10 * Ddx) * dfdy[xup] - C01) / (
        Ddx * Ddx
    ) - C11 / Ddx

    # C11 = (-tmq + C21 * Ddx * Ddx) / (Ddx)

    out_f[core] = (
        ((C30 * XX + C21 * YY + C20) * XX + C11 * YY + C10) * XX
        + ((C03 * YY + C12 * XX + C02) * YY + C01) * YY
        + C00
    ) / (1 + a10 * b10 * XX + a01 * b01 * YY)
    out_dfdx[core] = (
        (3.0 * C30 * XX + 2.0 * (C21 * YY + C20)) * XX
        + (C12 * YY + C11) * YY
        + C10
        - a10 * b10 * out_f[core]
    ) / (1 + a10 * b10 * XX + a01 * b01 * YY)
    out_dfdy[core] = (
        (3.0 * C03 * YY + 2.0 * (C12 * XX + C02)) * YY
        + (C21 * XX + C11) * XX
        + C01
        - a01 * b01 * out_f[core]
    ) / (1 + a10 * b10 * XX + a01 * b01 * YY)

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
    xi_x = -dx / 2.0
    a = (dfdx[iplus] + dfdx[iminus]) / (D_x ** 2) + 2.0 * (f[iplus] - f[iminus]) / (
        D_x ** 3
    )
    b = (
        3.0 * (f[iminus] - f[iplus]) / (D_x ** 2)
        - (2.0 * dfdx[iplus] + dfdx[iminus]) / D_x
    )
    out[core] = a * (xi_x ** 3) + b * (xi_x ** 2) + \
        dfdx[iplus] * xi_x + f[iplus]

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
    xi = -dx / 2.0
    BB = np.zeros(core.shape, dtype=float)
    S = (f[iminus] - f[iplus]) / D
    dz_index = (S - dfdx[iplus]) * (dfdx[iminus] - S) > 0.0
    BB[dz_index] = (
        np.abs(
            (S[dz_index] - dfdx[iplus][dz_index])
            / (dfdx[iminus][dz_index] - S[dz_index])
        )
        - 1.0
    ) / D

    a = (dfdx[iplus] - S + (dfdx[iminus] - S) * (1.0 + BB * D)) / (D ** 2)
    b = S * BB + (S - dfdx[iplus]) / D - a * D
    c = dfdx[iplus] + f[iplus] * BB

    out[core] = (((a * xi + b) * xi + c) * xi + f[iplus]) / (1.0 + BB * xi)

    # adjust negative values
    negative_value = out[core] < 0
    out[core][negative_value] = (
        f[iplus][negative_value] + f[iminus][negative_value]
    ) / 2.0

    return out


def forester_filter(
    f, core, east_id, west_id, north_id, south_id, nu_f=0.1, out_f=None,
):
    """ Forester filter for removing negative values from Concentration and
        Flow depth
    """

    if out_f is None:
        out_f = np.zeros_like(f)

    out_f[:] = f[:]

    east = east_id[core]
    west = west_id[core]
    north = north_id[core]
    south = south_id[core]

    out_f[core] += (
        nu_f * (f[east] + f[west] + f[north] + f[south] - 4.0 * f[core]) / 4.0
    )
    out_f[east] -= nu_f * (f[east] - f[core]) / 4.0
    out_f[west] -= nu_f * (f[west] - f[core]) / 4.0
    out_f[north] -= nu_f * (f[north] - f[core]) / 4.0
    out_f[south] -= nu_f * (f[south] - f[core]) / 4.0

    return out_f


class Jameson:
    """ Jameson filter for smoothing the variables

        Parameters
        -------------------
        tc: TurbudityCurrent2D
          Instance of TurbidityCurrent2D to obtain necessary parameters
    """

    def __init__(self, tc):

        self.number_of_nodes = tc.grid.number_of_nodes
        self.number_of_links = tc.grid.number_of_links
        self.node_east = tc.node_east
        self.node_west = tc.node_west
        self.node_north = tc.node_north
        self.node_south = tc.node_south
        self.link_east = tc.link_east
        self.link_west = tc.link_west
        self.link_north = tc.link_north
        self.link_south = tc.link_south
        self.link_horiz = tc.grid.horizontal_links
        self.link_vert = tc.grid.vertical_links
        self.east_node_at_horizontal_link = tc.east_node_at_horizontal_link
        self.west_node_at_horizontal_link = tc.west_node_at_horizontal_link
        self.north_node_at_vertical_link = tc.north_node_at_vertical_link
        self.south_node_at_vertical_link = tc.south_node_at_vertical_link
        self.east_link_at_node = tc.east_link_at_node
        self.west_link_at_node = tc.west_link_at_node
        self.north_link_at_node = tc.north_link_at_node
        self.south_link_at_node = tc.south_link_at_node
        self.kappa = tc.kappa
        self.kappa = tc.kappa
        self.nu_x = np.zeros(self.number_of_nodes)
        self.nu_y = np.zeros(self.number_of_nodes)
        self.nu_x_link = np.zeros(self.number_of_links)
        self.nu_y_link = np.zeros(self.number_of_links)
        self.eps_link = np.zeros(self.number_of_links)
        self.eps_node_horiz = np.zeros(self.number_of_nodes)
        self.eps_node_vert = np.zeros(self.number_of_nodes)

    def update_artificial_viscosity(self, p, p_link):
        """ update artificial viscosity at nodes (nu) and links (eps)

            paramters
            -------------------
            p : ndarray, float
                pressure at nodes
            p_link : ndarray, float
                pressure at links
        """
        # artificial viscosity coefficient at links
        self.nu_x_link[self.link_horiz] = np.abs(
            p_link[self.link_east[self.link_horiz]]
            - 2 * p_link[self.link_horiz]
            + p_link[self.link_west[self.link_horiz]]
        ) / (
            p_link[self.link_east[self.link_horiz]]
            + 2 * p_link[self.link_horiz]
            + p_link[self.link_west[self.link_horiz]]
            + 10 ** -20
        )
        self.nu_y_link[self.link_vert] = np.abs(
            p_link[self.link_north[self.link_vert]]
            - 2 * p_link[self.link_vert]
            + p_link[self.link_south[self.link_vert]]
        ) / (
            p_link[self.link_north[self.link_vert]]
            + 2 * p_link[self.link_vert]
            + p_link[self.link_south[self.link_vert]]
            + 10 ** -20
        )

        # artificial viscosity coefficient at nodes
        self.nu_x[:] = np.abs(
            p[self.node_east]
            + p[self.node_north]
            + p[self.node_south]
            + p[self.node_west]
            - 4 * p
        ) / (
            p[self.node_east]
            + p[self.node_west]
            + p[self.node_north]
            + p[self.node_south]
            + 4 * p
            + 10 ** -20
        )
        self.nu_y[:] = self.nu_x[:]

        # maximum artificial viscosity coefficient at links
        self.eps_link[self.link_horiz] = self.kappa * np.max(
            [
                self.nu_x[self.east_node_at_horizontal_link[self.link_horiz]],
                self.nu_x[self.west_node_at_horizontal_link[self.link_horiz]],
            ],
            axis=0,
        )
        self.eps_link[self.link_vert] = self.kappa * np.max(
            [
                self.nu_y[self.north_node_at_vertical_link[self.link_vert]],
                self.nu_y[self.south_node_at_vertical_link[self.link_vert]],
            ],
            axis=0,
        )

        # maximum artificial viscosity coefficient at nodes
        self.eps_node_horiz[:] = (
            0.01
            * self.kappa
            * np.max(
                [
                    self.nu_x_link[self.east_link_at_node],
                    self.nu_x_link[self.west_link_at_node],
                ],
                axis=0,
            )
        )
        self.eps_node_vert[:] = (
            0.01
            * self.kappa
            * np.max(
                [
                    self.nu_y_link[self.north_link_at_node],
                    self.nu_y_link[self.south_link_at_node],
                ],
                axis=0,
            )
        )

    def run(self, f, core, at="node", out=None):
        """ run one step of the Jameson filter

            paramters
            --------------------
            f : ndarray, float
                variables to be filtered

            core : ndarray, int
                grid ids to apply the filter

            at : String, optional
                'node', 'hlink', or 'vlink' for values on nodes,
                horizontal links or vertical links

            out : ndarray, float
                output

            returns
            --------------------
            out : ndarray, float
                filtered variables
        """
        if out is None:
            out = np.zeros_like(f)

        if at == "node":
            out[core] = (
                f[core]
                + self.eps_link[self.east_link_at_node[core]]
                * (f[self.node_east[core]] - f[core])
                - self.eps_link[self.west_link_at_node[core]]
                * (f[core] - f[self.node_west[core]])
                + self.eps_link[self.north_link_at_node[core]]
                * (f[self.node_north[core]] - f[core])
                - self.eps_link[self.south_link_at_node[core]]
                * (f[core] - f[self.node_south[core]])
            )

        if at == "hlink":
            out[core] = (
                f[core]
                + self.eps_node_horiz[self.east_node_at_horizontal_link[core]]
                * (f[self.link_east[core]] - f[core])
                - self.eps_node_horiz[self.west_node_at_horizontal_link[core]]
                * (f[core] - f[self.west_node_at_horizontal_link[core]])
            )

        if at == "vlink":
            out[core] = (
                f[core]
                + self.eps_node_vert[self.north_node_at_vertical_link[core]]
                * (f[self.link_north[core]] - f[core])
                - self.eps_node_vert[self.south_node_at_vertical_link[core]]
                * (f[core] - f[self.south_node_at_vertical_link[core]])
            )

        return out


class SOR:
    """SOR method to solve inverse matrix
    """

    def __init__(
        self,
        number_of_nodes,
        node_east,
        node_west,
        node_north,
        node_south,
        implicit_threshold,
        max_loop,
        alpha,
        update_boundary_conditions,
    ):

        self.implicit_threshold = implicit_threshold
        self.max_loop = max_loop
        self.alpha = alpha
        self.update_boundary_conditions = update_boundary_conditions

        self.node_east = node_east
        self.node_west = node_west
        self.node_north = node_north
        self.node_south = node_south

        self.a = np.empty(number_of_nodes)
        self.b = np.empty(number_of_nodes)
        self.c = np.empty(number_of_nodes)
        self.d = np.empty(number_of_nodes)
        self.e = np.empty(number_of_nodes)
        self.g = np.empty(number_of_nodes)
        self.w = np.empty(number_of_nodes)

    def run(self, p, core, out=None):

        if out is None:
            out = np.zeros_like(p)

        out[:] = p[:]
        err = 100.0
        count = 0
        core_size = core.shape[0]

        while err > self.implicit_threshold:
            self.w[core] = (
                self.g[core]
                - self.b[core] * out[self.node_east[core]]
                - self.c[core] * out[self.node_west[core]]
                - self.d[core] * out[self.node_north[core]]
                - self.e[core] * out[self.node_south[core]]
            ) / self.a[core]
            err = np.linalg.norm(self.w[core] - out[core]) / (
                core_size + 1.0 * 10 ** -20
            )
            out[core] = out[core] * (1 - self.alpha) + \
                self.alpha * self.w[core]
            self.update_boundary_conditions(p=out)

            count += 1
            if count == self.max_loop:
                print("Implicit calculation did not converge")
                return out

        return out
