"""Utilities module."""

import numpy as np
import dask
import dask.array as da
import astropy.units as u

G = 4.299e-6 * u.kpc * (u.km / u.s) ** 2 / u.M_sun
G = G.to_value()


def _get_rot_matrix(m, pos, vel, r_corte=None):
    """
    Rotation matrix calculation.

    Calculates the rotation matrix that aligns the TOTAL
    agular momentum of the particles with the z-axis.
    The positions, velocities and masses of the particles are used.
    Optionally, only particles within a cutting radius `(r_corte)` can be used.

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


def aling(m_s, x_s, y_s, z_s, vx_s, vy_s, vz_s,
          m_dm, x_dm, y_dm, z_dm, vx_dm, vy_dm, vz_dm,
          m_g, x_g, y_g, z_g, vx_g, vy_g, vz_g, r_corte):
    """
    Aling the galaxy.

    Rotate the positions, speeds and angular moments of the
    particles so that the total angular moment coincides with the z-axis.
    Optionally, only particles within a cutting radius
    `(r_corte)` can be used to calculate the rotation matrix.

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
    pos = np.vstack((x_s, y_s, z_s)).T
    vel = np.vstack((vx_s, vy_s, vz_s)).T

    A = _get_rot_matrix(m_s, pos, vel, r_corte)

    pos_rot_s = np.dot(A, pos.T)
    vel_rot_s = np.dot(A, vel.T)

    pos = np.vstack((x_dm, y_dm, z_dm)).T
    vel = np.vstack((vx_dm, vy_dm, vz_dm)).T

    pos_rot_dm = np.dot(A, pos.T)
    vel_rot_dm = np.dot(A, vel.T)

    pos = np.vstack((x_g, y_g, z_g)).T
    vel = np.vstack((vx_g, vy_g, vz_g)).T

    pos_rot_g = np.dot(A, pos.T)
    vel_rot_g = np.dot(A, vel.T)

    return (
        pos_rot_s.T[:, 0], pos_rot_s.T[:, 1], pos_rot_s.T[:, 2],
        vel_rot_s.T[:, 0], vel_rot_s.T[:, 1], vel_rot_s.T[:, 2],
        pos_rot_dm.T[:, 0], pos_rot_dm.T[:, 1], pos_rot_dm.T[:, 2],
        vel_rot_dm.T[:, 0], vel_rot_dm.T[:, 1], vel_rot_dm.T[:, 2],
        pos_rot_g.T[:, 0], pos_rot_g.T[:, 1], pos_rot_g.T[:, 2],
        vel_rot_g.T[:, 0], vel_rot_g.T[:, 1], vel_rot_g.T[:, 2]
    )


@dask.delayed
def _potential_dask(x, y, z, m, eps):
    """
    Calculate the specific gravitational potential energy of particles.

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
    dist = np.sqrt(
        np.square(x - x.reshape(-1, 1))
        + np.square(y - y.reshape(-1, 1))
        + np.square(z - z.reshape(-1, 1))
        + np.square(eps)
    )

    np.fill_diagonal(dist, 0.0)

    flt = dist != 0
    mdist = da.divide(m, dist.astype(np.float32), where=flt)

    return mdist.sum(axis=1) * G


def potential(x, y, z, m, eps=0.0):
    """Compute de potential energy."""
    pot = _potential_dask(x, y, z, m, eps)
    return np.asarray(pot.compute())
