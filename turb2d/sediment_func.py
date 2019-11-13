def get_es(u_star, Ds, R, g, nu, ws, out=None):
    """ Calculate entrainment rate of basal sediment to suspension
        Based on Garcia and Parker (1991)
        Parameters

        --------------

        u_star : ndarray
            flow shear velocity
        Ds : float
            particle diameter
        R : float
            submerged specific density of sediment particle
        g : float
            gravity acceleration
        nu : float
            kinematic viscosity of water
        ws : float
            settling velocity of sediment particle

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

    # calculate subordinate parameters

    Rp = np.sqrt(R * g * Ds) * Ds / nu

    sus_index = u_star / ws

    # empirical coefficients for calculation

    a = 7.8 * 10**-7

    alpha = 0.6

    p = 0.1

    # calculate entrainemnt rate

    Z = sus_index * Rp**alpha

    out[:] = p * a * Z**5 / (1 + (a / 0.3) * Z**5)

    return out
