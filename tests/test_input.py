# -*- coding: utf-8 -*-
# This file is part of the Galaxy-Chop Project
# License: MIT

"""Test imput data."""

# =============================================================================
# IMPORTS
# =============================================================================

import pytest
import numpy as np
from galaxychop import utils
from galaxychop import galaxychop
import astropy.units as u

# =============================================================================
# Random state
# =============================================================================

random = np.random.RandomState(seed=42)

# =============================================================================
# Defining utility functions for mocking data
# =============================================================================


def rot_matrix_xaxis(theta=0):
    """
    Rotation matrix of a transformation around X axis.

    Parameters
    ----------
    theta : `float`
        Rotation angle in radians

    Returns
    -------
    A : `np.ndarray`
        Rotation matrix, with shape (3, 3)
    """
    A = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -1 * np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    return A


def rot_matrix_yaxis(theta=0):
    """
    Rotation matrix of a transformation around Y axis.

    Parameters
    ----------
    theta : `float`
        Rotation angle in radians
    Returns
    -------
    A : `np.ndarray`
        Rotation matrix, with shape (3, 3)
    """
    A = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-1 * np.sin(theta), 0, np.cos(theta)],
        ]
    )
    return A


def rot_matrix_zaxis(theta=0):
    """
    Rotation matrix of a transformation around Z axis.

    Parameters
    ----------
    theta : `float`
        Rotation angle in radians
    Returns
    -------
    A : `np.ndarray`
        Rotation matrix, with shape (3, 3)
    """
    A = np.array(
        [
            [np.cos(theta), -1 * np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return A


def rotate(pos, vel, matrix):
    """
    Rotate.

    Apply the rotation `matrix` to a set of particles positions `pos` and
    velocities `vel`

    Parameters
    ----------
    pos : `np.ndarray`, shape = N_part, 3
        Positions of particles
    vel : `np.ndarray`, shape = N_part, 3
        Velocities of particles
    matrix : `np.ndarray`
        Rotation matrix, with shape (3, 3)

    Returns
    -------
    pos_rot : `np.ndarray`, shape = N_part, 3
        Rotated, positions of particles
    vel_rot : `np.ndarray`, shape = N_part, 3
        Rotated, velocities of particles
    """
    pos_rot = pos @ matrix
    vel_rot = vel @ matrix

    return pos_rot, vel_rot


def save_data(N_part=100):
    """
    Save data.

    This function saves a file with mock particles in a solid disk created with
    `solid_disk` function to run potentials with `potential_test.f90`
    to validate the potential function with dask

    Parameters
    ----------
    N_part : `int`
        The total number of particles to obtain

    Returns
    -------
    File named  `mock_particles.dat` on the folder tests/test_data
    with 4 columns and N_part rows. From left to right:
    x, y, z : Positions
    mass : Masses
    """
    mass, pos, vel = solid_disk(N_part)
    data = np.ndarray([len(mass), 4])
    data[:, 0] = pos[:, 0]
    data[:, 1] = pos[:, 1]
    data[:, 2] = pos[:, 2]
    data[:, 3] = mass

    np.savetxt("test_data/mock_particles.dat", data, fmt="%12.6f")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def solid_disk():
    """Mock solid disk.

    Creates a mock solid disc of particles with masses
    and velocities.
    """

    def make(N_part=100, rmax=30, rmin=5, omega=10, seed=42):

        random = np.random.RandomState(seed=seed)

        r = (rmax - rmin) * random.random_sample(size=N_part) + rmin
        phi0 = 2 * np.pi * random.random_sample(size=N_part)
        mass = 1.0 * np.ones_like(r)

        x = r * np.cos(phi0)
        y = r * np.sin(phi0)
        z = 1 * random.random_sample(size=N_part) - 0.5

        xdot = -1 * omega * r * np.sin(phi0)
        ydot = omega * r * np.cos(phi0)
        zdot = np.zeros_like(xdot)

        pos = np.array([x, y, z]).T
        vel = np.array([xdot, ydot, zdot]).T

        return mass, pos, vel

    return make


@pytest.fixture(scope="session")
def mock_dm_halo():
    """Mock Dark Matter Halo.

    Creates a mock DM Halo of particles with masses
    and velocities.
    """

    def make(N_part=100, rmax=30, rmin=5, omega=10, seed=55):

        random = np.random.RandomState(seed=seed)

        r = (rmax - rmin) * random.random_sample(size=N_part) + rmin
        cos_t = random.random_sample(size=N_part) * 2.0 - 1
        phi0 = 2 * np.pi * random.random_sample(size=N_part)
        sin_t = np.sqrt(1 - cos_t ** 2)
        mass = 1.0 * np.ones_like(r)

        x = r * sin_t * np.cos(phi0)
        y = r * sin_t * np.sin(phi0)
        z = r * cos_t

        pos = np.array([x, y, z]).T

        return mass, pos

    return make


@pytest.fixture(scope="session")
def disc_zero_angle(solid_disk):
    """Disc with no angle of inclination."""
    mass, pos, vel = solid_disk(N_part=1000)
    return mass, pos, vel


@pytest.fixture(scope="session")
def disc_xrotation(solid_disk):
    """Disc rotated over x axis."""
    mass, pos, vel = solid_disk(N_part=1000)
    a = rot_matrix_xaxis(theta=0.3 * np.pi * random.random())

    return mass, pos @ a, vel @ a, a


@pytest.fixture(scope="session")
def disc_yrotation(solid_disk):
    """Disc rotated over y axis."""
    mass, pos, vel = solid_disk(N_part=1000)
    a = rot_matrix_yaxis(theta=0.3 * np.pi * random.random())

    return mass, pos @ a, vel @ a, a


@pytest.fixture(scope="session")
def disc_zrotation(solid_disk):
    """Disc rotated over z axis."""
    mass, pos, vel = solid_disk(N_part=1000)
    a = rot_matrix_zaxis(theta=0.3 * np.pi * random.random())

    return mass, pos @ a, vel @ a, a


@pytest.fixture(scope="session")
def disc_particles(solid_disk):
    """Solid disc without velocities."""
    mass, pos, vel = solid_disk(N_part=100)
    return pos[:, 0], pos[:, 1], pos[:, 2], mass


@pytest.fixture(scope="session")
def disc_particles_all(solid_disk):
    """Solid disc wit velocities."""
    mass_s, pos_s, vel_s = solid_disk(N_part=100)
    mass_g, pos_g, vel_g = solid_disk(N_part=100)

    return mass_s, pos_s, vel_s, mass_g, pos_g, vel_g


@pytest.fixture(scope="session")
def halo_particles(mock_dm_halo):
    """Spherical mock halo."""
    mass_dm, pos_dm = mock_dm_halo(N_part=100)
    vel_dm = random.random_sample(size=(100, 3))

    return mass_dm, pos_dm, vel_dm


@pytest.fixture(scope="session")
def mock_galaxy(disc_particles_all, halo_particles):
    """Mock galaxy."""
    (mass_s, pos_s, vel_s,
     mass_g, pos_g, vel_g) = disc_particles_all

    mass_dm, pos_dm, vel_dm = halo_particles

    g = galaxychop.Galaxy(
        x_s=pos_s[:, 0] * u.kpc,
        y_s=pos_s[:, 1] * u.kpc,
        z_s=pos_s[:, 2] * u.kpc,
        vx_s=vel_s[:, 0] * (u.km / u.s),
        vy_s=vel_s[:, 1] * (u.km / u.s),
        vz_s=vel_s[:, 2] * (u.km / u.s),
        m_s=mass_s * u.M_sun,
        x_dm=pos_dm[:, 0] * u.kpc,
        y_dm=pos_dm[:, 1] * u.kpc,
        z_dm=pos_dm[:, 2] * u.kpc,
        vx_dm=vel_dm[:, 0] * (u.km / u.s),
        vy_dm=vel_dm[:, 1] * (u.km / u.s),
        vz_dm=vel_dm[:, 2] * (u.km / u.s),
        m_dm=mass_dm * u.M_sun,
        x_g=pos_g[:, 0] * u.kpc,
        y_g=pos_g[:, 1] * u.kpc,
        z_g=pos_g[:, 2] * u.kpc,
        vx_g=vel_g[:, 0] * (u.km / u.s),
        vy_g=vel_g[:, 1] * (u.km / u.s),
        vz_g=vel_g[:, 2] * (u.km / u.s),
        m_g=mass_g * u.M_sun,
    )

    return g

# =============================================================================
# TESTS
# =============================================================================


def test_getrotmat0(disc_zero_angle):
    """Test rotation matrix 1."""
    gxchA = utils._get_rot_matrix(*disc_zero_angle)

    np.testing.assert_allclose(1.0, gxchA[2, 2], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[2, 1], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[2, 0], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[0, 2], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[1, 2], rtol=1e-4, atol=1e-3)


def test_invert_xaxis(disc_xrotation):
    """Test rotation matrix 2."""
    m, pos, vel, a = disc_xrotation
    gxchA = utils._get_rot_matrix(m, pos, vel)

    np.testing.assert_allclose(1.0, gxchA[0, 0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[0, 1], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[0, 2], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[1, 0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[2, 0], rtol=1e-3, atol=1e-3)


def test_invert_yaxis(disc_yrotation):
    """Test rotation matrix 3."""
    m, pos, vel, a = disc_yrotation
    gxchA = utils._get_rot_matrix(m, pos, vel)

    np.testing.assert_allclose(0.0, gxchA[0, 0], rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(1.0, gxchA[0, 1], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[0, 2], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[1, 1], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[2, 1], rtol=1e-3, atol=1e-3)


def test_invert_zaxis(disc_zrotation):
    """Test rotation matrix 4."""
    m, pos, vel, a = disc_zrotation
    gxchA = utils._get_rot_matrix(m, pos, vel)

    np.testing.assert_allclose(1.0, gxchA[2, 2], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[2, 1], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[2, 0], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[0, 2], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(0.0, gxchA[1, 2], rtol=1e-4, atol=1e-3)


@pytest.mark.xfail
def test_daskpotential(disc_particles):
    """Test potential function."""
    dpotential = utils.potential(*disc_particles)
    fpotential = np.loadtxt("tests/test_data/fpotential_test.dat")
    np.testing.assert_allclose(dpotential, fpotential, rtol=1e-4, atol=1e-3)


@pytest.mark.xfail
def test_energy_method(disc_particles_all, halo_particles):
    """Test energy method."""
    (mass_s, pos_s, vel_s,
     mass_g, pos_g, vel_g) = disc_particles_all

    mass_dm, pos_dm, vel_dm = halo_particles

    g = galaxychop.Galaxy(
        x_s=pos_s[:, 0] * u.kpc,
        y_s=pos_s[:, 1] * u.kpc,
        z_s=pos_s[:, 2] * u.kpc,
        vx_s=vel_s[:, 0] * (u.km / u.s),
        vy_s=vel_s[:, 1] * (u.km / u.s),
        vz_s=vel_s[:, 2] * (u.km / u.s),
        m_s=mass_s * u.M_sun,
        x_dm=pos_dm[:, 0] * u.kpc,
        y_dm=pos_dm[:, 1] * u.kpc,
        z_dm=pos_dm[:, 2] * u.kpc,
        vx_dm=vel_dm[:, 0] * (u.km / u.s),
        vy_dm=vel_dm[:, 1] * (u.km / u.s),
        vz_dm=vel_dm[:, 2] * (u.km / u.s),
        m_dm=mass_dm * u.M_sun,
        x_g=pos_g[:, 0] * u.kpc,
        y_g=pos_g[:, 1] * u.kpc,
        z_g=pos_g[:, 2] * u.kpc,
        vx_g=vel_g[:, 0] * (u.km / u.s),
        vy_g=vel_g[:, 1] * (u.km / u.s),
        vz_g=vel_g[:, 2] * (u.km / u.s),
        m_g=mass_g * u.M_sun,
    )

    E_tot_dm, E_tot_s, E_tot_g = g.energy

    k_s = 0.5 * (vel_s[:, 0] ** 2 + vel_s[:, 1] ** 2 + vel_s[:, 2] ** 2)
    k_dm = 0.5 * (vel_dm[:, 0] ** 2 + vel_dm[:, 1] ** 2 + vel_dm[:, 2] ** 2)
    k_g = 0.5 * (vel_g[:, 0] ** 2 + vel_g[:, 1] ** 2 + vel_g[:, 2] ** 2)

    pot_s = utils.potential(x=pos_s[:, 0], y=pos_s[:, 1],
                            z=pos_s[:, 2], m=mass_s)

    pot_dm = utils.potential(x=pos_dm[:, 0], y=pos_dm[:, 1],
                             z=pos_dm[:, 2], m=mass_dm)

    pot_g = utils.potential(x=pos_g[:, 0], y=pos_g[:, 1],
                            z=pos_g[:, 2], m=mass_g)

    np.testing.assert_allclose(E_tot_s.value, k_s - pot_s,
                               rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(E_tot_dm.value, k_dm - pot_dm,
                               rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(E_tot_g.value, k_g - pot_g,
                               rtol=1e-3, atol=1e-3)


def test_k_energy(disc_particles_all, halo_particles):
    """Test kinetic energy."""
    (mass_s, pos_s, vel_s,
     mass_g, pos_g, vel_g) = disc_particles_all

    mass_dm, pos_dm, vel_dm = halo_particles

    k_s = 0.5 * (vel_s[:, 0] ** 2 + vel_s[:, 1] ** 2 + vel_s[:, 2] ** 2)
    k_dm = 0.5 * (vel_dm[:, 0] ** 2 + vel_dm[:, 1] ** 2 + vel_dm[:, 2] ** 2)
    k_g = 0.5 * (vel_g[:, 0] ** 2 + vel_g[:, 1] ** 2 + vel_g[:, 2] ** 2)

    assert (k_s >= 0).all()
    assert (k_dm >= 0).all()
    assert (k_g >= 0).all()


def test_dm_pot_energy(halo_particles):
    """Test potential energy DM."""
    mass_dm, pos_dm, vel_dm = halo_particles

    p_s = utils.potential(x=pos_dm[:, 0], y=pos_dm[:, 1], z=pos_dm[:, 2],
                          m=mass_dm)
    assert (p_s > 0).all()


def test_stars_and_gas_pot_energy(disc_particles_all):
    """Test potential energy STAR and GAS."""
    (mass_s, pos_s, vel_s,
     mass_g, pos_g, vel_g) = disc_particles_all

    p_g = utils.potential(x=pos_g[:, 0], y=pos_g[:, 1], z=pos_g[:, 2],
                          m=mass_g)
    p_s = utils.potential(x=pos_s[:, 0], y=pos_s[:, 1], z=pos_s[:, 2],
                          m=mass_s)
    assert (p_g > 0).all()
    assert (p_s > 0).all()


@pytest.mark.xfail
def test_total_enrgy(mock_galaxy):
    """Test total energy."""
    g = mock_galaxy

    E_tot_dark, E_tot_star, E_tot_gas = g.energy

    ii, = np.where(E_tot_star.value < 0)
    perc = len(ii) / len(E_tot_star.value)

    assert perc > 0.9
    assert (E_tot_star.value < 0).any()


def test_type_energy(disc_particles_all, halo_particles):
    """Checks the object."""
    (mass_s, pos_s, vel_s,
     mass_g, pos_g, vel_g) = disc_particles_all

    mass_dm, pos_dm, vel_dm = halo_particles

    k_s = 0.5 * (vel_s[:, 0] ** 2 + vel_s[:, 1] ** 2 + vel_s[:, 2] ** 2)
    k_dm = 0.5 * (vel_dm[:, 0] ** 2 + vel_dm[:, 1] ** 2 + vel_dm[:, 2] ** 2)
    k_g = 0.5 * (vel_g[:, 0] ** 2 + vel_g[:, 1] ** 2 + vel_g[:, 2] ** 2)

    p_s = utils.potential(x=pos_s[:, 0], y=pos_s[:, 1], z=pos_s[:, 2],
                          m=mass_s)
    p_dm = utils.potential(x=pos_dm[:, 0], y=pos_dm[:, 1], z=pos_dm[:, 2],
                           m=mass_dm)
    p_g = utils.potential(x=pos_g[:, 0], y=pos_g[:, 1], z=pos_g[:, 2],
                          m=mass_g)

    assert isinstance(p_s, (float, np.float, np.ndarray))
    assert isinstance(p_dm, (float, np.float, np.ndarray))
    assert isinstance(p_g, (float, np.float, np.ndarray))
    assert isinstance(k_s, (float, np.float, np.ndarray))
    assert isinstance(k_dm, (float, np.float, np.ndarray))
    assert isinstance(k_g, (float, np.float, np.ndarray))


def test_jcirc_E_tot_len(mock_galaxy):
    """Check the E_tot array len."""
    g = mock_galaxy

    E_tot_dm, E_tot_s, E_tot_g = g.energy

    E_tot = np.hstack((E_tot_s.value, E_tot_dm.value, E_tot_g.value))
    tot_len = len(E_tot_dm) + len(E_tot_s) + len(E_tot_g)

    assert (len(E_tot) == tot_len)


@pytest.mark.xfail
def test_jcirc_x_y_len(disc_particles_all):
    """Check the x and y array len."""
    (
        mass_s,
        pos_s,
        vel_s,
        mass_g,
        pos_g,
        vel_g,
        mass_d,
        pos_d,
        vel_d,
    ) = disc_particles_all
    g = galaxychop.Galaxy(
        pos_s[:, 0] * u.kpc,
        pos_s[:, 1] * u.kpc,
        pos_s[:, 2] * u.kpc,
        vel_s[:, 0] * u.km / u.s,
        vel_s[:, 1] * u.km / u.s,
        vel_s[:, 2] * u.km / u.s,
        mass_s * u.M_sun,
        pos_d[:, 0] * u.kpc,
        pos_d[:, 1] * u.kpc,
        pos_d[:, 2] * u.kpc,
        vel_d[:, 0] * u.km / u.s,
        vel_d[:, 1] * u.km / u.s,
        vel_d[:, 2] * u.km / u.s,
        mass_d * u.M_sun,
        pos_g[:, 0] * u.kpc,
        pos_g[:, 1] * u.kpc,
        pos_g[:, 2] * u.kpc,
        vel_g[:, 0] * u.km / u.s,
        vel_g[:, 1] * u.km / u.s,
        vel_g[:, 2] * u.km / u.s,
        mass_g * u.M_sun,
    )
    g.Etot_dm = (random.random(50) * (-100))
    g.Etot_s = (random.random(50) * (-100))
    g.Etot_g = (random.random(50) * (-100))

    Jx = random.random(150)
    Jy = random.random(150)
    Jz = random.random(150)
    g.J_part = (np.vstack((Jx, Jy, Jz))) * 100

    x, y = g.jcirc()
    assert (len(x) == len(y))


@pytest.mark.xfail
def test_param_circ_eps_one_minus_one(disc_particles_all):
    """Check is the eps range."""
    (
        mass_s,
        pos_s,
        vel_s,
        mass_g,
        pos_g,
        vel_g,
        mass_d,
        pos_d,
        vel_d,
    ) = disc_particles_all
    g = galaxychop.Galaxy(
        pos_s[:, 0] * u.kpc,
        pos_s[:, 1] * u.kpc,
        pos_s[:, 2] * u.kpc,
        vel_s[:, 0] * u.km / u.s,
        vel_s[:, 1] * u.km / u.s,
        vel_s[:, 2] * u.km / u.s,
        mass_s * u.M_sun,
        pos_d[:, 0] * u.kpc,
        pos_d[:, 1] * u.kpc,
        pos_d[:, 2] * u.kpc,
        vel_d[:, 0] * u.km / u.s,
        vel_d[:, 1] * u.km / u.s,
        vel_d[:, 2] * u.km / u.s,
        mass_d * u.M_sun,
        pos_g[:, 0] * u.kpc,
        pos_g[:, 1] * u.kpc,
        pos_g[:, 2] * u.kpc,
        vel_g[:, 0] * u.km / u.s,
        vel_g[:, 1] * u.km / u.s,
        vel_g[:, 2] * u.km / u.s,
        mass_g * u.M_sun,
    )
    g.Etot_dm = (random.random(300) * (-100))
    g.Etot_s = (random.random(300) * (-100))
    g.Etot_g = (random.random(300) * (-100))

    Jx = random.random(900)
    Jy = random.random(900)
    Jz = random.random(900)

    g.J_star = np.vstack((Jx, Jy, Jz))
    g.J_part = np.vstack((Jx, Jy, Jz))
    g.Jr_star = random.random(600)
    g.Jr = random.random(900)
    g.jcirc()
    E_star, eps, eps_r = g.paramcirc()
    assert (eps <= 1).all()
    assert (eps >= -1).all()
