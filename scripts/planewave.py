import numpy as np
from numpy import sin, cos, tan, arctan2, sqrt, pi
from scipy.optimize import brute, fmin
import sys


c = 299792458.0 # / 1.0003  # speed of light in m/s - air has n ~ 1.0003; and cos(2 degrees) ~ 1 / 1.0003... needed?
twopi = 2 * pi
halfpi = 0.5 * pi
rad2deg = 360.0 / twopi
deg2rad = twopi / 360.0
nan = float('nan')


def directionForHorizontalArray(positions, times, ignoreZCoordinate=False):
    """
    Author: Arthur Corstanje
    Given N antenna positions, and (pulse) arrival times for each antenna,
    get a direction of arrival (az, el) assuming a source at infinity (plane wave).

    Here, we find the direction assuming all antennas are placed in the z=0 plane.
    If all antennas are co-planar, the best-fitting solution can be found using a 2D-linear fit.
    We find the best-fitting A and B in:

    .. math::

        t = A x + B y + C

    where t is the array of times; x and y are arrays of coordinates of the antennas.
    The C is the overall time offset in the data, that has to be subtracted out.
    The optimal value of C has to be determined in the fit process (it's not just the average time, nor the time at antenna 0).

    This is done using :mod:`numpy.linalg.lstsq`.

    The (az, el) follows from:

    .. math::

        A = \cos(\mathrm{el}) \cos(\mathrm{az})

        B = \cos(\mathrm{el}) \sin(\mathrm{az})

    Required arguments:

    =========== ==========================================================
    Parameter   Description
    =========== ==========================================================
    *positions* array ``(x1, y1, z1, x2, y2, z2, x3, y3, z3)``, in meters.
                NB: :math:`z_i` is simply ignored but still assumed to be
                there!
    *times*     array ``(t1, t2, t3)``, in seconds.
    =========== ==========================================================

    Input is assumed to be Numpy arrays.

    Output: ``(az, el)``, in radians, and seconds-squared.

    """

    # make x, y arrays out of the input position array
#    N = len(positions)
    x = positions[0:-1:3]
    y = positions[1:-1:3]

    # now a crude test for nonzero z-input, |z| > 0.5
    z = positions[2:-1:3]
    if not ignoreZCoordinate and max(abs(z)) > 0.5:
        raise ValueError("Input values of z are nonzero ( > 0.5) !")
        return (-1, -1)

    M = np.vstack([x, y, np.ones(len(x))]).T  # says the linalg.lstsq doc

    (A, B, C) = np.linalg.lstsq(M, c * times)[0]

    el = np.arccos(np.sqrt(A * A + B * B))
    az = halfpi - np.arctan2(-B, -A)  # note minus sign as we want the direction of the _incoming_ vector (from the sky, not towards it)
    # note: Changed to az = 90_deg - phi
    return (az, el)



def timeDelaysFromDirection(positions, direction):
    """
    Get time delays for antennas at given position for a given direction.
    Time delays come out as an np-array.

    Required arguments:

    =========== =================================================
    Parameter   Description
    =========== =================================================
    *positions* ``(np-array x1, y1, z1, x2, y2, z2, ...)``
    *direction* (az, el) in radians.
    =========== =================================================

    Example:

    .. doctest::

        >>> pos = np.array([0, 0, 0, 23, -21, 54, 11, 21, 33])
        >>> dir = (1.2, 0.7)
        >>> timeDelaysFromDirection(pos, dir)
        array([ -0.00000000e+00,  -8.73671261e-08,  -1.31017019e-07])

    """

    n = int(len(positions) / 3)
    phi = halfpi - direction[0]  # warning, 90 degree? -- Changed to az = 90_deg - phi
    theta = halfpi - direction[1]  # theta as in standard spherical coords, while el=90 means zenith...

    cartesianDirection = np.array([sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)])
    timeDelays = np.zeros(n)
    for i in range(n):
        thisPosition = np.array(positions[3 * i:3 * (i + 1)])
        timeDelays[i] = - (1 / c) * np.dot(cartesianDirection, thisPosition)  # note the minus sign! Signal vector points down from the sky.

    return timeDelays
