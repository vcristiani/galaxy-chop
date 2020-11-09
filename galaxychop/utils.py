import numpy as np
import dask
import dask.array as da
from astropy import units as u

G = 4.299e-6 * u.kpc * (u.km / u.s) ** 2 / u.M_sun


def _get_rot_matrix(m, pos, vel, r_corte=None):
    """
    Calculates the rotation matrix that aligns the TOTAL
    agular momentum of the particles with the z-axis. The
    positions, velocities and masses of the particles are
    used. Optionally, only particles within a cutting
    radius `(r_corte)` can be used.

    Parameters
    ----------
    m : `np.ndarray`, shape(n,1)
        Masses of particles.
    pos : `np.ndarray`, shape(n,3)
        Positions of particles.
    vel : `np.ndarray`, shape(n,3)
        Velocities of particles.
    r_corte : `float`, optional
        The default is ``None``; if provided, it must be
        positive and the rotation matrix `A` is calculated
        from the particles with radii smaller than
        r_corte.

    Returns
    -------
    A : `np.ndarray`, shape(3,3)
        Rotation matrix.
    """

    jx = m * (pos[:, 1] * vel[:, 2] - pos[:, 2] * vel[:, 1])
    jy = m * (pos[:, 2] * vel[:, 0] - pos[:, 0] * vel[:, 2])
    jz = m * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0])

    r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2)

    if r_corte is not None:
        mask = np.where(r < r_corte)
    else:
        mask = (np.repeat(True, len(r)),)

    rjx = np.sum(jx[mask])
    rjy = np.sum(jy[mask])
    rjz = np.sum(jz[mask])

    rjp = np.sqrt(rjx ** 2 + rjy ** 2)
    rj = np.sqrt(rjx ** 2 + rjy ** 2 + rjz ** 2)

    e1x = rjy / rjp
    e1y = -rjx / rjp
    e1z = 0.0

    e2x = rjx * rjz / (rjp * rj)
    e2y = rjy * rjz / (rjp * rj)
    e2z = -rjp / rj

    e3x = rjx / rj
    e3y = rjy / rj
    e3z = rjz / rj

    A = np.asarray(([e1x, e1y, e1z], [e2x, e2y, e2z], [e3x, e3y, e3z]))

    return A


def aling(m, pos, vel, r_corte):
    """This rotates the positions, speeds and angular
    moments of the particles so that the total angular
    moment coincides with the z-axis.

    Optionally, only particles within a cutting radius
    `(r_corte)` can be used to calculate the rotation
    matrix.

    Parameters
    ----------
    m : `np.ndarray`, shape(n,1)
        Masses of particles.
    pos : `np.ndarray`, shape(n,3)
        Positions of particles.
    vel : `np.ndarray`, shape(n,3)
        Velocities of particles.
    r_corte : `float`, optional
        The default is ``None``; if provided, it must be
        positive and the rotation matrix `A` is calculated
        from the particles with radii smaller than r_corte.

    Returns
    -------
    pos_rot : `np.ndarray`, shape(n,3)
        Rotated positions of particles
    vel_rot : `np.ndarray`, shape(n,3)
        Rotated velocities of particles
    """
    A = _get_rot_matrix(m, pos, vel, r_corte)

    pos_rot = np.dot(A, pos.T)
    vel_rot = np.dot(A, vel.T)

    return pos_rot.T, vel_rot.T


@dask.delayed
def _potential_dask(x, y, z, m, eps=0.1):
    """This calculates the specific gravitational potential energy of
    particles.

    Parameters
    ----------
    x, y, z: `np.ndarray`, shape(n,1)
        Positions of particles.
    m:  `np.ndarray`, shape(n,1)
        Masses of particles.
    eps: `float`, optional
        Softening parameter.

    Returns
    -------
    Specific potential energy of particles
    """

    dist = np.sqrt(np.square(x - x.reshape(-1, 1))
                   + np.square(y - y.reshape(-1, 1))
                   + np.square(z - z.reshape(-1, 1))
                   + np.square(eps))

    np.fill_diagonal(dist, 0.0)

    flt = dist != 0
    mdist = da.divide(m, dist.astype(np.float32), where=flt)

    return mdist.sum(axis=1) * G


def potential(x, y, z, m, eps=0.1):

    pot = _potential_dask(x, y, z, m, eps=0.1)

    return np.asarray(pot.compute())
