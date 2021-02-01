"""Save data for fortran potential test."""

import numpy as np


def mock_dm_halo():
    """
    Mock dark matter Halo.

    Creates a mock DM halo of particles with masses
    and velocities.
    """

    def make(N_part=1000, rmax=100, seed=55):

        random = np.random.RandomState(seed=seed)

        r = random.random_sample(size=N_part) * rmax

        cos_t = random.random_sample(size=N_part) * 2.0 - 1
        phi0 = 2 * np.pi * random.random_sample(size=N_part)
        sin_t = np.sqrt(1 - cos_t ** 2)
        mass = 1.0e10 * np.ones_like(r)

        x = r * sin_t * np.cos(phi0)
        y = r * sin_t * np.sin(phi0)
        z = r * cos_t

        pos = np.array([x, y, z]).T

        return mass, pos

    return make


def halo_particles(mock_dm_halo):
    """Spherical mock halo."""

    def make(N_part=100, seed=None):
        random = np.random.RandomState(seed=seed)
        mass_dm, pos_dm = mock_dm_halo(N_part=N_part)
        vel_dm = random.random_sample(size=(N_part, 3))

        return mass_dm, pos_dm, vel_dm

    return make


def save_data(halo_particles):
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
    mass, pos, vel = halo_particles(N_part=100, seed=42)
    data = np.ndarray([len(mass), 4])
    data[:, 0] = pos[:, 0]
    data[:, 1] = pos[:, 1]
    data[:, 2] = pos[:, 2]
    data[:, 3] = mass

    np.savetxt("mock_particles.dat", data, fmt="%12.6f")
