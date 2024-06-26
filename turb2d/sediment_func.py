import numpy as np

"""
Empirical functions for calculating models of sediment dynamics
"""


def get_ew(U, Ch, R, g, umin=0.01, out=None):
    """calculate entrainment coefficient of ambient water to a turbidity
    current layer

    Parameters
    ----------
    U : ndarray, float
       Flow velocities of a turbidity current.
    Ch : ndarray, float
       Flow height times sediment concentration of a turbidity current.
    R : float
       Submerged specific density of sediment
    g : float
       gravity acceleration
    umin: float
       minimum threshold value of velocity to calculate water entrainment

    out : ndarray
       Outputs

    Returns
    ---------
    e_w : ndarray, float
       Entrainment coefficient of ambient water

    """
    if out is None:
        out = np.zeros(U.shape)

    Ri = np.zeros(U.shape)
    flowing = np.where(U > umin)
    Ri[flowing] = R * g * Ch[flowing] / U[flowing] ** 2
    out = 0.075 / np.sqrt(1 + 718.0 * Ri**2.4)  # Parker et al. (1987)

    return out


def get_det_rate(ws, Ch_i, h, det_factor=1.0, out=None):
    """Calculate rate of detrainment caused by sediment
      settling

      The detrainment rate at the flow interface is assumed
      to be proportional to the sediment settling rate.
      The default value for the proportionality factor is 1.0,
      but Salinas et al. (2019) estimate that 3.05 is
      an appropriate value based on DNS calculations.

    Parameters
    ----------
    ws : ndarray, float
       sediment settling rate
    C_i: ndarray(2d), float
        Sediment concentration for the ith grain size class
    det_factor: float
        Factor for detrainment rate. The default value is
        1.0.

    out : ndarray
       Outputs

    Returns
    ---------
    e_d : ndarray, float
       Dentrainment coefficient of water (positive when
       fluid exits the flow)

    """

    if out is None:
        out = np.zeros(h.shape)

    eps = 1.0e-15
    C_i = Ch_i / h + eps

    out[:] = det_factor * np.sum((ws * C_i / np.sum(C_i, axis=0)), axis=0)

    return out


def get_ws(R, g, Ds, nu):
    """Calculate settling velocity of sediment particles
        on the basis of Ferguson and Church (1982)

    Return
    ------------------
    ws : settling velocity of sediment particles [m/s]

    """

    # Coefficients for natural sands
    C_1 = 18.0
    C_2 = 1.0

    ws = R * g * Ds**2 / (C_1 * nu + (0.75 * C_2 * R * g * Ds**3) ** 0.5)

    return ws


def get_es(R, g, Ds, nu, u_star, function="GP1991field", out=None):
    """Calculate entrainment rate of basal sediment to suspension using
    empirical functions proposed by Garcia and Parker (1991),
    van Rijn (1984), or Dorrell (2018)

    Parameters
    --------------
    R : float
        submerged specific density of sediment (~1.65 for quartz particle)
    g : float
        gravity acceleration
    Ds : float
        grain size
    nu : float
        kinematic viscosity of water
    u_star : ndarray
        flow shear velocity
    function : string, optional
        Name of emprical function to be used.

        'GP1991exp' is a function of Garcia and Parker (1991)
         in original form. This is suitable for experimental scale.

        'GP1991field' is Garcia and Parker (1991)'s function with
        a coefficient (0.1) to limit the entrainment rate. This is suitable
        for the natural scale.

    out : ndarray
        Outputs (entrainment rate of basal sediment)

    Returns
    ---------------
    out : ndarray
        dimensionless entrainment rate of basal sediment into
        suspension



    """
    if out is None:
        out = np.zeros([len(Ds), len(u_star)])

    if function == "GP1991field":
        _gp1991(R, g, Ds, nu, u_star, p=0.1, out=out)
    if function == "GP1991exp":
        _gp1991(R, g, Ds, nu, u_star, p=1.0, out=out)

    return out


def _gp1991(R, g, Ds, nu, u_star, p=1.0, out=None):
    """Calculate entrainment rate of basal sediment to suspension
    Based on Garcia and Parker (1991)

    Parameters
    --------------
    u_star : ndarray
        flow shear velocity
    out : ndarray
        Outputs (entrainment rate of basal sediment)

    Returns
    ---------------
    out : ndarray
        dimensionless entrainment rate of basal sediment into
        suspension
    """

    if out is None:
        out = np.zeros([len(Ds), u_star.shape])

    # basic parameters
    ws = get_ws(R, g, Ds, nu)

    # calculate subordinate parameters
    Rp = np.sqrt(R * g * Ds) * Ds / nu
    sus_index = u_star / ws

    # coefficients for calculation
    a = 7.8 * 10**-7
    alpha = 0.6

    # calculate entrainment rate
    Z = sus_index * Rp**alpha
    out[:, :] = p * a * Z**5 / (1 + (a / 0.3) * Z**5)

    return out
