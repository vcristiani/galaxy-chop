# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""test for galaxychop.methods

"""


# =============================================================================
# IMPORTS
# =============================================================================

from galaxychop.core import methods

import pytest


# =============================================================================
# TESTS
# =============================================================================
# _gchop_decomp_type
# _gchop_parameters
# _gchop_abstract_class
# GchopMethodABC


def test_GchopMethodABC_no__gchop_decomp_type():
    with pytest.raises(TypeError):

        class Foo(methods.GchopMethodABC):
            pass


def test_GchopMethodABC_no__gchop_parameters():
    with pytest.raises(TypeError):

        class Foo(methods.GchopMethodABC):
            _gchop_decomp_type = "foo"

            def __init__(self, **kwargs):
                pass


def test_GchopMethodABC_repr():
    class Foo(methods.GchopMethodABC):
        _gchop_decomp_type = "foo"
        _gchop_parameters = ["foo", "faa"]

        def __init__(self, foo, faa):
            self.foo = foo
            self.faa = faa

    foo = Foo(foo=2, faa=1)

    assert repr(foo) == "<Foo [faa=1, foo=2]>"


def test_GchopMethodABC_repr_no_params():
    class Foo(methods.GchopMethodABC):
        _gchop_decomp_type = "foo"
        _gchop_parameters = []

    foo = Foo()

    assert repr(foo) == "<Foo []>"


def test_GchopMethodABC_no_params():
    class Foo(methods.GchopMethodABC):
        _gchop_decomp_type = "foo"
        _gchop_parameters = []

    assert Foo._gchop_parameters == frozenset()


def test_GchopMethodABC_already_defined__gchop_parameters():
    class Base(methods.GchopMethodABC):
        _gchop_decomp_type = "foo"
        _gchop_parameters = ["x"]

        def __init__(self, x):
            pass

    class Foo(Base):
        def __init__(self, x):
            pass

    assert Foo._gchop_parameters == {"x"}


def test_GchopMethodABC_params_in_init():
    class Base(methods.GchopMethodABC):
        _gchop_decomp_type = "foo"
        _gchop_parameters = ["x"]

        def __init__(self, **kwargs):
            pass

    with pytest.raises(TypeError):

        class Foo(Base):
            def __init__(self):
                pass


def test_GchopMethodABC_get_parameters():
    class Foo(methods.GchopMethodABC):
        _gchop_decomp_type = "foo"
        _gchop_parameters = ["foo", "faa"]

        def __init__(self, foo, faa):
            self.foo = foo
            self.faa = faa

    foo = Foo(foo=2, faa=1)

    assert foo.get_parameters() == {"foo": 2, "faa": 1}


def test_GchopMethodABC_copy():
    class Foo(methods.GchopMethodABC):
        _gchop_decomp_type = "foo"
        _gchop_parameters = ["foo", "faa"]

        def __init__(self, foo, faa):
            self.foo = foo
            self.faa = faa

    foo = Foo(foo=2, faa=1)
    copy = foo.copy()

    assert foo.get_parameters() == copy.get_parameters()
