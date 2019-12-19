import numpy as np
cimport numpy as np

DOUBLE = np.float64
INT = np.int
BOOL = np.bool

ctypedef np.float64_t DOUBLE_T
ctypedef np.int_t INT_T
ctypedef np.npy_bool BOOL_T


def cip_2d_M_advection(np.ndarray[DOUBLE_T, ndim=1] f,
                       np.ndarray[DOUBLE_T, ndim=1] dfdx,
                       np.ndarray[DOUBLE_T, ndim=1] dfdy,
                       np.ndarray[DOUBLE_T, ndim=1] u,
                       np.ndarray[DOUBLE_T, ndim=1] v,
                       np.ndarray[INT_T, ndim=1] core,
                       np.ndarray[INT_T, ndim=1]  h_up,
                       np.ndarray[INT_T, ndim=1] h_down,
                       np.ndarray[INT_T, ndim=1] v_up,
                       np.ndarray[INT_T, ndim=1] v_down,
                       double dx,
                       double dt,
                       np.ndarray[DOUBLE_T, ndim=1] out_f=None,
                       np.ndarray[DOUBLE_T, ndim=1] out_dfdx=None,
                       np.ndarray[DOUBLE_T, ndim=1] out_dfdy=None):
    """Calculate one time step using M-type 2D cip method
    """
    cdef np.ndarray[DOUBLE_T, ndim= 1] D_x, D_y, xi_x, xi_y, a, b
    cdef int n = f.shape[0]

    # First, the variables out and temp are allocated to
    # store the calculation results

    if out_f is None:
        out_f = np.empty(n, dtype=DOUBLE)
    if out_dfdx is None:
        out_dfdx = np.empty(n, dtype=DOUBLE)
    if out_dfdy is None:
        out_dfdy = np.empty(n, dtype=DOUBLE)

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


def cip_2d_nonadvection(np.ndarray[DOUBLE_T, ndim=1] f,
                        np.ndarray[DOUBLE_T, ndim=1] dfdx,
                        np.ndarray[DOUBLE_T, ndim=1] dfdy,
                        np.ndarray[DOUBLE_T, ndim=1] G,
                        np.ndarray[DOUBLE_T, ndim=1] u,
                        np.ndarray[DOUBLE_T, ndim=1] v,
                        np.ndarray[INT_T, ndim=1] core,
                        np.ndarray[INT_T, ndim=1] h_up,
                        np.ndarray[INT_T, ndim=1] h_down,
                        np.ndarray[INT_T, ndim=1] v_up,
                        np.ndarray[INT_T, ndim=1] v_down,
                        double dx,
                        double dt,
                        np.ndarray[DOUBLE_T, ndim=1] out_f=None,
                        np.ndarray[DOUBLE_T, ndim=1] out_dfdx=None,
                        np.ndarray[DOUBLE_T, ndim=1] out_dfdy=None):

    cdef np.ndarray[DOUBLE_T, ndim= 1] D_x, D_y, xi_x, xi_y
    cdef int n = f.shape[0]

    if out_f is None:
        out_f = np.zeros(n, dtype=DOUBLE)
    if out_dfdx is None:
        out_dfdx = np.zeros(n, dtype=DOUBLE)
    if out_dfdy is None:
        out_dfdy = np.zeros(n, dtype=DOUBLE)

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


def cip_2d_diffusion(np.ndarray[DOUBLE_T, ndim=1] u,
                     np.ndarray[DOUBLE_T, ndim=1] v,
                     np.ndarray[DOUBLE_T, ndim=1] nu_t,
                     np.ndarray[INT_T, ndim=1] h_active,
                     np.ndarray[INT_T, ndim=1] v_active,
                     np.ndarray[INT_T, ndim=1] north,
                     np.ndarray[INT_T, ndim=1] south,
                     np.ndarray[INT_T, ndim=1] east,
                     np.ndarray[INT_T, ndim=1] west,
                     double dx,
                     double dt,
                     np.ndarray[DOUBLE_T, ndim=1] out_u=None,
                     np.ndarray[DOUBLE_T, ndim=1] out_v=None):
    """Caclulate horizontal and vertical diffusion of velocities u and v
    """
    cdef int n = u.shape[0]

    if out_u is None:
        out_u = np.zeros(n, dtype=DOUBLE)
    if out_v is None:
        out_v = np.zeros(n, dtype=DOUBLE)

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


def rcip_1d_advection(np.ndarray[DOUBLE_T, ndim=1] f,
                      np.ndarray[DOUBLE_T, ndim=1] dfdx,
                      np.ndarray[DOUBLE_T, ndim=1] u,
                      np.ndarray[INT_T, ndim=1] core,
                      np.ndarray[INT_T, ndim=1] up,
                      np.ndarray[INT_T, ndim=1] down,
                      double dx,
                      double dt,
                      np.ndarray[DOUBLE_T, ndim=1] out_f=None,
                      np.ndarray[DOUBLE_T, ndim=1] out_dfdx=None):
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
    cdef int n = f.shape[0]
    cdef np.ndarray[DOUBLE_T, ndim = 1] D, xi, BB, alpha, S
    cdef np.ndarray[BOOL_T, cast = True, ndim = 1] dz_index

    if out_f is None:
        out_f = np.zeros(n, dtype=DOUBLE)
    if out_dfdx is None:
        out_dfdx = np.zeros(n, dtype=DOUBLE)

    # advection phase
    D = -np.where(u > 0., 1.0, -1.0) * dx
    xi = -u * dt
    BB = np.ones(D[core].shape[0], dtype=DOUBLE)
    alpha = np.zeros(D[core].shape[0], dtype=DOUBLE)
    S = (f[up] - f[core]) / D[core]
    dz_index = (dfdx[up] - S) == 0.0
    BB[dz_index] = -1.0 / D[core][dz_index]
    BB[~dz_index] = (np.abs(
        (S[~dz_index] - dfdx[core][~dz_index]) /
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


def rcip_2d_M_advection(np.ndarray[DOUBLE_T, ndim=1] f,
                        np.ndarray[DOUBLE_T, ndim=1] dfdx,
                        np.ndarray[DOUBLE_T, ndim=1] dfdy,
                        np.ndarray[DOUBLE_T, ndim=1] u,
                        np.ndarray[DOUBLE_T, ndim=1] v,
                        np.ndarray[INT_T, ndim=1] core,
                        np.ndarray[INT_T, ndim=1] h_up,
                        np.ndarray[INT_T, ndim=1] h_down,
                        np.ndarray[INT_T, ndim=1] v_up,
                        np.ndarray[INT_T, ndim=1] v_down,
                        double dx,
                        double dt,
                        np.ndarray[DOUBLE_T, ndim=1] out_f=None,
                        np.ndarray[DOUBLE_T, ndim=1] out_dfdx=None,
                        np.ndarray[DOUBLE_T, ndim=1] out_dfdy=None):
    """Calculate one time step using M-type 2D cip method
    """

    # First, the variables out and temp are allocated to
    # store the calculation results
    cdef np.ndarray[DOUBLE_T, ndim= 1] D_x, D_y, xi_x, xi_y,
    cdef np.ndarray[DOUBLE_T, ndim = 1] alpha, BB_x, BB_y, S_x, S_y
    cdef np.ndarray[BOOL_T, cast= True, ndim = 1] dz_index
    cdef int n = f.shape[0]
    cdef int m = core.shape[0]

    if out_f is None:
        out_f = np.empty(n, dtype=DOUBLE)
    if out_dfdx is None:
        out_dfdx = np.empty(n, dtype=DOUBLE)
    if out_dfdy is None:
        out_dfdy = np.empty(n, dtype=DOUBLE)

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
    out_dfdy[core] = dfdy[core] - xi_x[core] / \
        D_x[core] * (dfdy[core] - dfdy[h_up])

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
    out_dfdx[core] = out_dfdx[core] - xi_y[core] / \
        D_y[core] * (out_dfdx[core] - out_dfdx[v_up])

    return out_f, out_dfdx, out_dfdy


def shock_dissipation(
    np.ndarray[DOUBLE_T, ndim=1] f,
    np.ndarray[DOUBLE_T, ndim=1] h,
    np.ndarray[INT_T, ndim=1] core,
    np.ndarray[INT_T, ndim=1] north_id,
    np.ndarray[INT_T, ndim=1] south_id,
    np.ndarray[INT_T, ndim=1] east_id,
    np.ndarray[INT_T, ndim=1] west_id,
    double dt,
    double kappa,
    np.ndarray[DOUBLE_T, ndim=1] out=None,
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
        kappa : double
            aritificial visocisty
    """
    cdef np.ndarray[DOUBLE_T, ndim = 1] eps_i, eps_i_half
    cdef np.ndarray[INT_T, ndim = 1] north, south, east, west
    cdef int n = f.shape[0]

    if out is None:
       out = np.zeros(n, dtype=DOUBLE)

    eps_i = np.zeros(n, dtype=DOUBLE)
    eps_i_half = np.zeros(n, dtype=DOUBLE)
    north = north_id[core]
    south = south_id[core]
    east = east_id[core]
    west = west_id[core]

    # First, artificial diffusion is applied to east-west direction
    eps_i[core] = kappa * np.abs(h[east] - 2 * h[core] + h[west]) /\
        (np.abs(h[east]) + 2 * np.abs(h[core]) + np.abs(h[west]) + 0.000000001)
    eps_i_half[core] = np.max([eps_i[east], eps_i[core]], axis=0)
    out[core] = f[core] + eps_i_half[core] * (f[east] - f[core]) - \
        eps_i_half[west] * (f[core] - f[west])

    # Next, artificial diffusion is applied to north-south direction
    eps_i[core] = kappa * np.abs(h[north] - 2 * h[core] + h[south]) /\
        (np.abs(h[north]) + 2 * np.abs(h[core]) + np.abs(h[south]) + 0.000000001)
    eps_i_half[core] = np.max([eps_i[north], eps_i[core]], axis=0)
    out[core] = out[core] + (eps_i_half[core] *
                             (out[north] - out[core]) - eps_i_half[south] *
                             (out[core] - out[south]))

    return out
