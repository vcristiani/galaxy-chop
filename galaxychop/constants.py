# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt


# =============================================================================
# DOCS
# =============================================================================

"""Constants for use inside galxychop"""

# =============================================================================
# IMPORTS
# =============================================================================

import astropy.constants as c
import astropy.units as u

# =============================================================================
# STELLAR DYNAMICS
# =============================================================================

SD_DEFAULT_CBIN = (0.05, 0.005)
"""Default binning of circularity for stellar dynamics calculation.

Please check the documentation of ``galaxychop.circ.stellar_dynamics()``.

"""
SD_DEFAULT_REASSIGN = False
"""Default value to reassign the values of the particle stellar dynamics.

Please check the documentation of ``galaxychop.circ.stellar_dynamics()``.

"""

SD_RUNTIME_WARNING_ACTION = "ignore"
"""Default of "what-to-do" about the RuntimeWarning in stellar_dynamics \
calculation.

Please check the documentation of ``galaxychop.circ.stellar_dynamics()``.

"""

# =============================================================================
# Gravity
# =============================================================================

#: GalaxyChop Gravitational unit
G_UNIT = (u.km**2 * u.kpc) / (u.s**2 * u.solMass)

#: Gravitational constant as float in G_UNIT
G = c.G.to(G_UNIT).to_value()
