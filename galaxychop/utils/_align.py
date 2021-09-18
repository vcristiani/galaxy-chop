# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Utilities for align the galaxies."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

# =============================================================================
# API
# =============================================================================


def _get_rot_matrix(m, pos, vel, r_cut=None):
    """
    Rotation matrix calculation.

    Calculates the rotation matrix that aligns the TOTAL
    angular momentum of the particles with the z-axis.
    The positions, velocities and masses of the particles are used.
    Optionally, only particles within a cutting radius `(r_cut)` can be used.

    Parameters
    ----------
    m : `np.ndarray`
        Masses of particles. Shape(n,1)
    pos : `np.ndarray`
        Positions of particles. Shape(n,3)
    vel : `np.ndarray`
        Velocities of particles. Shape(n,3)
    r_cut : `float`, optional
        The default is ``None``; if provided, it must be
        positive and the rotation matrix `A` is calculated
        from the particles with radii smaller than
        r_cut.

    Returns
    -------
    A : `np.ndarray`
        Rotation matrix. Shape(3,3)
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
    m,
    x_s,
    y_s,
    z_s,
    x_dm,
    y_dm,
    z_dm,
    x_g,
    y_g,
    z_g,
    vx,
    vy,
    vz,
    r_cut,
):
    """Align the galaxy.

    Rotates the positions, velocities and angular momentum of the
    particles so that the total angular moment coincides with the z-axis.
    Optionally, only particles within a cutting radius
    `(r_cut)` can be used to calculate the rotation matrix.



    """
    if r_cut is not None and r_cut <= 0.0:
        raise ValueError("r_cut must not be lower than 0.")

    pos = np.vstack((x_s, y_s, z_s)).T
    vel = np.vstack((vx[0], vy[0], vz[0])).T

    A = _get_rot_matrix(m[0], pos, vel, r_cut)

    pos_rot_s = np.dot(A, pos.T)
    vel_rot_s = np.dot(A, vel.T)

    pos = np.vstack((x_dm, y_dm, z_dm)).T
    vel = np.vstack((vx[1], vy[1], vz[1])).T

    pos_rot_dm = np.dot(A, pos.T)
    vel_rot_dm = np.dot(A, vel.T)

    pos = np.vstack((x_g, y_g, z_g)).T
    vel = np.vstack((vx[2], vy[2], vz[2])).T

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
