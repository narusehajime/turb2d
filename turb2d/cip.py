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
    out_dfdy[core] = dfdy[core] + xi_x[core] / \
        dx * (dfdy[core] - dfdy[h_up])

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
    out_dfdx[core] = out_dfdx[core] + xi_y[core] / \
        dx * (out_dfdx[core] - out_dfdx[v_up])

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


def rcip_1d_advection(self,
                      f,
                      dfdx,
                      u,
                      core,
                      up,
                      down,
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
    BB_x = np.ones(D_x[core].shape)
    alpha = np.zeros(D_x[core].shape)
    S_x = (f[h_up] - f[core]) / D_x[core]
    dz_index = (dfdx[h_up] - S_x) == 0.0
    BB_x[dz_index] = -1.0 / D_x[core][dz_index]
    BB_x[~dz_index] = (np.abs(
        (S_x[~dz_index] - dfdx[core][~dz_index]) /
        (dfdx[h_up][~dz_index] - S_x[~dz_index] + 1.e-10)) -
                       1.0) / D_x[core][~dz_index]
    alpha[(S_x - dfdx[core]) / (dfdx[h_up] - S_x + 1.e-10) >= 0.0] = 1.0

    a = (dfdx[core] - S_x +
             (dfdx[h_up] - S_x) * (1.0 + alpha * BB_x * D_x[core])) \
             / (D_x[core]**2)
    b = S_x * alpha * BB_x + (S_x - dfdx[core]) / D_x[core] - a * D_x[core]
    c = dfdx[core] + f[core] * alpha * BB_x

    out_f[core] = (((a * xi_x[core] + b) * xi_x[core] + c)
                   * xi_x[core] + f[core]) \
        / (1.0 + alpha * BB_x * xi_x[core])
    out_dfdx[core] = ((3. * a * xi_x[core] + 2. * b) * xi_x[core] + c) \
            / (1.0 + alpha * BB_x * xi_x[core]) \
            - out_f[core] * alpha * BB_x / (1.0 + alpha * BB_x * xi_x[core])
    out_dfdy[core] = dfdy[core] + xi_x[core] / \
        dx * (dfdy[core] - dfdy[h_up])

    # 2nd step for vertical advection
    D_y = -np.where(v > 0., 1.0, -1.0) * dx
    xi_y = -v * dt
    BB_y = np.ones(D_y[core].shape)
    alpha = np.zeros(D_y[core].shape)
    S_y = (out_f[v_up] - out_f[core]) / D_y[core]
    dz_index = (out_dfdy[v_up] - S_y) == 0.0
    BB_y[dz_index] = -1.0 / D_y[core][dz_index]
    BB_y[~dz_index] = (np.abs(
        (S_y[~dz_index] - out_dfdy[core][~dz_index]) /
        (out_dfdy[v_up][~dz_index] - S_y[~dz_index] + 1.e-10)) -
                       1.0) / D_y[core][~dz_index]
    alpha[(S_y - out_dfdy[core]) /
          (out_dfdy[v_up] - S_y + 1.e-10) >= 0.0] = 1.0

    a = (out_dfdy[core] - S_y +
             (out_dfdy[v_up] - S_y) * (1.0 + alpha * BB_y * D_y[core])) \
             / (D_y[core]**2)
    b = S_y * alpha * BB_y + (S_y - out_dfdy[core]) / D_y[core] - a * D_y[core]
    c = out_dfdy[core] + out_f[core] * alpha * BB_y

    out_f[core] = (((a * xi_y[core] + b) * xi_y[core] + c)
                   * xi_y[core] + out_f[core]) \
        / (1.0 + alpha * BB_y * xi_y[core])
    out_dfdy[core] = ((3. * a * xi_y[core] + 2. * b) * xi_y[core] + c) \
            / (1.0 + alpha * BB_y * xi_y[core]) \
            - out_f[core] * alpha * BB_y / (1.0 + alpha * BB_y * xi_y[core])
    out_dfdx[core] = out_dfdx[core] + xi_y[core] / \
        dx * (out_dfdx[core] - out_dfdx[v_up])

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


def update_gradient_terms(f,
                          f_new,
                          dfdx,
                          dfdy,
                          core,
                          north,
                          south,
                          east,
                          west,
                          dx,
                          out_dfdx=None,
                          out_dfdy=None):
    """update gradient terms that are used in cip scheme. When f is updated,
       this function must be executed.
       
       Parameters
       ---------------
       f : ndarray, float
         Main variable before update
       f_new : ndarray, float
         Main variable after update
       dfdx : ndarray, float
         f differentiated by x
       dfdy : ndarray, float
         f differentiated by y
       north : ndarray, int
         grid id located at north
       south : ndarray, int
         grid id located at south
       east : ndarray, int
         grid id located at east
       west : ndarray, int
         grid id located at west
       dx : float
         grid spacing
       out_dfdx : ndarray, float
         output for dfdx
       out_dfdy : ndarray, float
         output for dfdy
    """
    if out_dfdx is None:
        out_dfdx = np.zeros(dfdx.shape[0])
    if out_dfdy is None:
        out_dfdy = np.zeros(dfdy.shape[0])

    out_dfdx[core] = dfdx[core] + 1 / (2 * dx) * (f_new[east] - f[east] -
                                                  f_new[west] + f[west])
    out_dfdy[core] = dfdy[core] + 1 / (2 * dx) * (f_new[north] - f[north] -
                                                  f_new[south] + f[south])

    return out_dfdx, out_dfdy
