# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2021, Valeria Cristiani
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

"""Test utilities provided by galaxychop."""

# =============================================================================
# IMPORTS
# =============================================================================


from galaxychop.utils import decorators


# =============================================================================
#   POTENTIAL ENERGY
# =============================================================================


def test_doc_inherit(galaxy):
    def with_doc():
        "something"

    @decorators.doc_inherit(with_doc)
    def foo():
        """
        Parameters
        ----------
        a: int
            Something

        """
    expected = 'something\n\nParameters\n----------\na: int\nSomething'

    assert foo.__doc__ == expected