# This file is part of
# the galxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Utilities module."""

# #####################################################
# IMPORTS
# #####################################################

import astropy.units as u

import dask
import dask.array as da

import numpy as np

G = (4.299e-6 * u.kpc * (u.km / u.s) ** 2 / u.M_sun).to_value()

# #####################################################
# Utility functions
# #####################################################


def _get_rot_matrix(m, pos, vel, r_cut=None):
    """
    Rotation matrix calculation.

    Calculates the rotation matrix that aligns the TOTAL
    agular momentum of the particles with the z-axis.
    The positions, velocities and masses of the particles are used.
    Optionally, only particles within a cutting radius `(r_cut)` can be used.

    Parameters
    ----------
    m : `np.ndarray`, shape(n,1)
        Masses of particles.
    pos : `np.ndarray`, shape(n,3)
        Positions of particles.
    vel : `np.ndarray`, shape(n,3)
        Velocities of particles.
    r_cut : `float`, optional
        The default is ``None``; if provided, it must be
        positive and the rotation matrix `A` is calculated
        from the particles with radii smaller than
        r_cut.

    Returns
    -------
    A : `np.ndarray`, shape(3,3)
        Rotation matrix.
    """
    jx = m * (pos[:, 1] * vel[:, 2] - pos[:, 2] * vel[:, 1])
    jy = m * (pos[:, 2] * vel[:, 0] - pos[:, 0] * vel[:, 2])
    jz = m * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0])

    r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2)

    if r_cut is not None:
        (mask,) = np.where(r < r_cut)
    else:
        mask = np.repeat(True, len(r))

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

    A = np.array(([e1x, e1y, e1z], [e2x, e2y, e2z], [e3x, e3y, e3z]))

    return A


def align(
    m_s,
    x_s,
    y_s,
    z_s,
    vx_s,
    vy_s,
    vz_s,
    x_dm,
    y_dm,
    z_dm,
    vx_dm,
    vy_dm,
    vz_dm,
    x_g,
    y_g,
    z_g,
    vx_g,
    vy_g,
    vz_g,
    r_cut,
):
    """
    Align the galaxy.

    Rotates the positions, velocities and angular momentum of the
    particles so that the total angular moment coincides with the z-axis.
    Optionally, only particles within a cutting radius
    `(r_cut)` can be used to calculate the rotation matrix.

    Parameters
    ----------
    m_s : `np.ndarray(n_s,1)`
        Star masses.
    x_s, y_s, z_s : `np.ndarray(n_s,1)`
        Star positions.
    vx_s, vy_s, vz_s : `np.ndarray(n_s,1)`
        Star velocities.
    x_dm, y_dm, z_dm : `np.ndarray(n_dm,1)`
        Dark matter positions.
    vx_dm, vy_dm, vz_dm : `np.ndarray(n_dm,1)`
        Dark matter velocities.
    x_g, y_g, z_g : `np.ndarray(n_g,1)`
        Gas positions.
    vx_g, vy_g, vz_g : `np.ndarray(n_g,1)`
        Gas velocities.
    r_cut : `float`, optional
        The default is ``None``; if provided, it must be
        positive and the rotation matrix `A` is calculated
        from the particles with smaller radii than r_cut.

    Returns
    -------
    tuple : `np.ndarray`
        x_s : `np.ndarray(n_s,1)`
            Rotated positions of the star particles.
        y_s : `np.ndarray(n_s,1)`
            Rotated positions of the star particles.
        z_s : `np.ndarray(n_s,1)`
            Rotated positions of the star particles.
        vx_s : `np.ndarray(n_s,1)`
            Rotated velocities of the star particles.
        vy_s : `np.ndarray(n_s,1)`
            Rotated velocities of the star particles.
        vz_s : `np.ndarray(n_s,1)`
            Rotated velocities of the star particles.
        x_dm : `np.ndarray(n_dm,1)`
            Rotated positions of the dark matter particles.
        y_dm : `np.ndarray(n_dm,1)`
            Rotated positions of the dark matter particles.
        z_dm : `np.ndarray(n_dm,1)`
            Rotated positions of the dark matter particles.
        vx_dm : `np.ndarray(n_dm,1)`
            Rotated velocities of the dark matter particles.
        vy_dm : `np.ndarray(n_dm,1)`
            Rotated velocities of the dark matter particles.
        vz_dm : `np.ndarray(n_dm,1)`
            Rotated velocities of the dark matter particles.
        x_g : `np.ndarray(n_g,1)`
            Rotated positions of the gas particles.
        y_g : `np.ndarray(n_g,1)`
            Rotated positions of the gas particles.
        z_g : `np.ndarray(n_g,1)`
            Rotated positions of the gas particles.
        vx_g : `np.ndarray(n_g,1)`
            Rotated velocities of the gas particles.
        vy_g : `np.ndarray(n_g,1)`
            Rotated velocities of the gas particles.
        vz_g : `np.ndarray(n_g,1)`
            Rotated velocities of the gas particles.

    """
    if (r_cut is not None) and (r_cut <= 0.0):
        raise ValueError("r_cut must not be lower than 0.")

    pos = np.vstack((x_s, y_s, z_s)).T
    vel = np.vstack((vx_s, vy_s, vz_s)).T

    A = _get_rot_matrix(m_s, pos, vel, r_cut)

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
        pos_rot_s.T[:, 0],
        pos_rot_s.T[:, 1],
        pos_rot_s.T[:, 2],
        vel_rot_s.T[:, 0],
        vel_rot_s.T[:, 1],
        vel_rot_s.T[:, 2],
        pos_rot_dm.T[:, 0],
        pos_rot_dm.T[:, 1],
        pos_rot_dm.T[:, 2],
        vel_rot_dm.T[:, 0],
        vel_rot_dm.T[:, 1],
        vel_rot_dm.T[:, 2],
        pos_rot_g.T[:, 0],
        pos_rot_g.T[:, 1],
        pos_rot_g.T[:, 2],
        vel_rot_g.T[:, 0],
        vel_rot_g.T[:, 1],
        vel_rot_g.T[:, 2],
    )


@dask.delayed
def _potential_dask(x, y, z, m, eps):
    """
    Specific gravitational potential energy of particles calculation.

    Parameters
    ----------
    x, y, z : `np.ndarray`, shape(n,1)
        Positions of particles.
    m : `np.ndarray`, shape(n,1)
        Masses of particles.
    eps : `float`, optional
        Softening parameter.

    Returns
    -------
    np.ndarray : `float`
    Specific potential energy of particles.

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
    """
    Potential energy calculation.

    Given the positions and masses of particles, calculate
    their specific gravitational potential energy.

    Parameters
    ----------
    x, y, z : `np.ndarray`, shape(n,1)
        Positions of particles.
    m : `np.ndarray`, shape(n,1)
        Masses of particles.
    eps : `float`, shape(1,), optional
        Softening parameter.

    Returns
    -------
    potential : `np.ndarray`
        Specific potential energy of particles.
    """
    pot = _potential_dask(x, y, z, m, eps)
    return np.asarray(pot.compute())


def center(
    m_s,
    x_s,
    y_s,
    z_s,
    m_dm,
    x_dm,
    y_dm,
    z_dm,
    m_g,
    x_g,
    y_g,
    z_g,
    pot_s=0,
    pot_dm=0,
    pot_g=0,
    eps_dm=0,
    eps_s=0,
    eps_g=0,
):
    """
    Centers the particles.

    Centers the position of all particles in the galaxy respect
    to the position of the lowest potential dark matter particle.

    Parameters
    ----------
    x_s, y_s, z_s : `np.ndarray(n_s,1)`
        Star positions.
    x_dm, y_dm, z_dm : `np.ndarray(n_dm,1)`
        Dark matter positions.
    x_g, y_g, z_g : `np.ndarray(n_g,1)`
        Gas positions.

    Returns
    -------
    tuple : `np.ndarray`
        x_s : `np.ndarray(n_s,1)`
            Centered star positions.
        y_s : `np.ndarray(n_s,1)`
            Centered star positions.
        z_s : `np.ndarray(n_s,1)`
            Centered star positions.
        x_dm : `np.ndarray(n_dm,1)`
            Centered dark matter positions.
        y_dm : `np.ndarray(n_dm,1)`
            Centered dark matter positions.
        z_dm : `np.ndarray(n_dm,1)`
            Centered dark matter positions.
        x_g : `np.ndarray(n_g,1)`
            Centered gas positions.
        y_g : `np.ndarray(n_g,1)`
            Centered gas positions.
        z_g : `np.ndarray(n_g,1)`
            Centered gas positions.

    """
    x = np.hstack((x_s, x_dm, x_g))
    y = np.hstack((y_s, y_dm, y_g))
    z = np.hstack((z_s, z_dm, z_g))
    m = np.hstack((m_s, m_dm, m_g))
    eps = np.max([eps_dm, eps_s, eps_g])

    total_potential = pot_dm

    if np.all(total_potential == 0.0):
        pot = potential(
            da.asarray(x, chunks=100),
            da.asarray(y, chunks=100),
            da.asarray(z, chunks=100),
            da.asarray(m, chunks=100),
            da.asarray(eps),
        )

        num_s = len(m_s)
        num = len(m_s) + len(m_dm)
        pot_dark = pot[num_s:num]
    else:
        pot_dark = pot_dm

    x_s = x_s - x_dm[pot_dark.argmax()]
    y_s = y_s - y_dm[pot_dark.argmax()]
    z_s = z_s - z_dm[pot_dark.argmax()]

    x_dm = x_dm - x_dm[pot_dark.argmax()]
    y_dm = y_dm - y_dm[pot_dark.argmax()]
    z_dm = z_dm - z_dm[pot_dark.argmax()]

    x_g = x_g - x_dm[pot_dark.argmax()]
    y_g = y_g - y_dm[pot_dark.argmax()]
    z_g = z_g - z_dm[pot_dark.argmax()]

    return x_s, y_s, z_s, x_dm, y_dm, z_dm, x_g, y_g, z_g
