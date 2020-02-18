import numpy as np
"""
Empirical functions for calculating models of sediment dynamics
"""


def get_ew(U, Ch, R, g, umin=0.01, out=None):
    """ calculate entrainemnt coefficient of ambient water to a turbidity
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
    Ri[flowing] = R * g * Ch[flowing] / U[flowing]**2
    out = 0.075 / np.sqrt(1 + 718. + Ri**2.4)  # Parker et al. (1987)

    return out


def get_ws(R, g, Ds, nu):
    """ Calculate settling velocity of sediment particles
        on the basis of Ferguson and Church (1982)

    Return
    ------------------
    ws : settling velocity of sediment particles [m/s]

    """

    # Coefficients for natural sands
    C_1 = 18.
    C_2 = 1.0

    ws = R * g * Ds**2 / (C_1 * nu + (0.75 * C_2 * R * g * Ds**3)**0.5)

    return ws


def get_es(R, g, Ds, nu, u_star, function='GP1991field', out=None):
    """ Calculate entrainment rate of basal sediment to suspension using
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
        out = np.zeros(u_star.shape)

    if function == 'GP1991field':
        _gp1991(R, g, Ds, nu, u_star, p=0.1, out=out)
    if function == 'GP1991exp':
        _gp1991(R, g, Ds, nu, u_star, p=1.0, out=out)

    return out


def _gp1991(R, g, Ds, nu, u_star, p=1.0, out=None):
    """ Calculate entrainment rate of basal sediment to suspension
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
        out = np.zeros(u_star.shape)

    # basic parameters
    ws = get_ws(R, g, Ds, nu)

    # calculate subordinate parameters
    Rp = np.sqrt(R * g * Ds) * Ds / nu
    sus_index = u_star / ws

    # coefficients for calculation
    a = 7.8 * 10**-7
    alpha = 0.6

    # calculate entrainemnt rate
    Z = sus_index * Rp**alpha
    out[:] = p * a * Z**5 / (1 + (a / 0.3) * Z**5)

    return out
