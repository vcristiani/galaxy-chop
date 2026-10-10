# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt
#
# The uttr module is adapted from uttrs (https://github.com/quatrope/uttrs):
# Copyright (c) 2020, Juan B Cabral and QuatroPe.
# License: BSD-3-Clause
# Full Text: licenses/uttrs-LICENSE.txt


# =============================================================================
# DOCS
# =============================================================================

"""test for uttr

"""

# =============================================================================
# IMPORTS
# =============================================================================

import unittest.mock as mock

import astropy.units as u

import attr

from galaxychop.utils import uttr

import numpy as np

import pytest

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def make_ucav():
    def make(unit):
        return uttr.UnitConverterAndValidator(unit=unit)

    return make


@pytest.fixture
def attribute():
    class Attribute:
        pass

    attribute = Attribute()
    attribute.name = "foo"
    return attribute


@pytest.fixture(scope="session")
def make_cls():
    def make(name="Foo", **kwargs):
        cls = type(name, (object,), kwargs)
        return attr.s(cls)

    return make


# =============================================================================
# TESTS
# =============================================================================


class TestUnitConverterAndValidator:
    def test_is_dimensionless(self, make_ucav):
        ucav = make_ucav(u.kg)
        assert ucav.is_dimensionless([1, 2, 3])
        assert ucav.is_dimensionless(1)
        assert not ucav.is_dimensionless([1, 2, 3] * u.kg)
        assert not ucav.is_dimensionless(1 * u.kg)

    def test_to_array(self, make_ucav):
        ucav = make_ucav(u.kg)
        assert ucav.to_array(1 * u.kg) == 1
        assert ucav.to_array(1 * u.g) == 0.001

        arr = [1, 2, 3] * u.g
        np.testing.assert_array_equal(
            ucav.to_array(arr), [0.001, 0.002, 0.003]
        )
        with pytest.raises(u.UnitConversionError):
            ucav.to_array(1 * u.m)

        with pytest.raises(AttributeError):
            ucav.to_array(1)

        with pytest.raises(AttributeError):
            ucav.to_array([1])

        with pytest.raises(AttributeError):
            ucav.to_array(np.array([1]))

    def test_convert_if_dimensionless(self, make_ucav):
        ucav = make_ucav(u.kg)
        assert ucav.convert_if_dimensionless(1).unit == u.kg
        assert ucav.convert_if_dimensionless(1 * u.kg).unit == u.kg
        assert ucav.convert_if_dimensionless(1 * u.m).unit == u.m

        arr = [1, 2, 3] * u.kg
        assert ucav.convert_if_dimensionless(arr) is arr

    def test_validate_is_equivalent_unit(self, make_ucav, attribute):
        ucav = make_ucav(u.kg)
        assert ucav.validate_is_equivalent_unit(None, attribute, 1) is None
        assert ucav.validate_is_equivalent_unit(None, attribute, [1]) is None

        assert (
            ucav.validate_is_equivalent_unit(None, attribute, 1 * u.g) is None
        )
        assert (
            ucav.validate_is_equivalent_unit(None, attribute, 1 * u.kg) is None
        )
        assert (
            ucav.validate_is_equivalent_unit(None, attribute, 1 * u.M_sun)
            is None
        )

        with pytest.raises(ValueError):
            ucav.validate_is_equivalent_unit(None, attribute, 1 * u.m)


class TestAttributeFunction:
    def test_converter(self, make_cls):
        with mock.patch.object(
            uttr.UnitConverterAndValidator, "convert_if_dimensionless"
        ) as asunit:
            Cls = make_cls(foo=uttr.ib(unit=u.kg))

            # check the converter
            asunit.assert_not_called()
            Cls(foo="foo")
            asunit.assert_called_once_with("foo")

    def test_converter_with_another(self, make_cls):
        def fake_converter(v):
            return v

        with mock.patch.object(
            uttr.UnitConverterAndValidator, "convert_if_dimensionless"
        ) as asunit:
            Cls = make_cls(foo=uttr.ib(unit=u.kg, converter=fake_converter))

            # check the converter
            asunit.assert_not_called()
            Cls(foo="foo")
            asunit.assert_called_once_with("foo")

    def test_validator(self, make_cls):
        with mock.patch.object(
            uttr.UnitConverterAndValidator, "validate_is_equivalent_unit"
        ) as validator:
            Cls = make_cls(foo=uttr.ib(unit=u.kg))
            attribute = attr.fields(Cls).foo

            validator.assert_not_called()
            instance = Cls(foo=1)
            validator.assert_called_once_with(instance, attribute, 1 * u.kg)

    def test_validator_with_another(self, make_cls):
        def fake_validator(*args, **kwargs):
            pass

        with mock.patch.object(
            uttr.UnitConverterAndValidator, "validate_is_equivalent_unit"
        ) as validator:
            Cls = make_cls(foo=uttr.ib(unit=u.kg, validator=fake_validator))
            attribute = attr.fields(Cls).foo

            validator.assert_not_called()
            instance = Cls(foo=1)
            validator.assert_called_once_with(instance, attribute, 1 * u.kg)

    def test_metadata(self, make_cls):
        Cls = make_cls(foo=uttr.ib(unit=u.kg))
        attribute = attr.fields(Cls).foo

        assert uttr.UTTR_METADATA in attribute.metadata

        ucav = attribute.metadata[uttr.UTTR_METADATA]
        assert isinstance(ucav, uttr.UnitConverterAndValidator)
        assert ucav.unit is u.kg

    def test_auto_conversion(self, make_cls):
        Cls = make_cls(foo=uttr.ib(unit=u.kg))
        instance = Cls(foo=1)
        assert instance.foo == 1 * u.kg

    def test_same_unit(self, make_cls):
        Cls = make_cls(foo=uttr.ib(unit=u.kg))
        instance = Cls(foo=1 * u.kg)
        assert instance.foo == 1 * u.kg

    def test_equivalent_unit(self, make_cls):
        Cls = make_cls(foo=uttr.ib(unit=u.kg))
        instance = Cls(foo=1 * u.g)
        assert instance.foo == 1 * u.g

    def test_invalid_unit(self, make_cls):
        Cls = make_cls(foo=uttr.ib(unit=u.kg))
        with pytest.raises(ValueError):
            Cls(foo=1 * u.m)

    def test_unit_None(self, make_cls):
        Cls = make_cls(foo=uttr.ib(unit=None), faa=uttr.ib(unit=u.kg))
        fields = attr.fields_dict(Cls)

        assert uttr.UTTR_METADATA not in fields["foo"].metadata
        assert uttr.UTTR_METADATA in fields["faa"].metadata


class TestArrayAccessor:
    def test_dimensionless_access(self, make_cls):
        Cls = make_cls(foo=uttr.ib(unit=u.kg))
        instance = Cls(foo=1)
        arr_ = uttr.ArrayAccessor(instance)

        assert instance.foo == 1 * u.kg
        assert arr_.foo == 1
        assert arr_["foo"] == 1

    def test_equivalent_access(self, make_cls):
        Cls = make_cls(foo=uttr.ib(unit=u.kg))
        instance = Cls(foo=1 * u.g)
        arr_ = uttr.ArrayAccessor(instance)

        assert instance.foo == 1 * u.g
        assert arr_.foo == 0.001
        assert arr_["foo"] == 0.001

    def test_same_unit_access(self, make_cls):
        Cls = make_cls(foo=uttr.ib(unit=u.kg))
        instance = Cls(foo=1 * u.kg)
        arr_ = uttr.ArrayAccessor(instance)

        assert instance.foo == 1 * u.kg
        assert arr_.foo == 1
        assert arr_["foo"] == 1

    def test_no_uttr_access(self, make_cls):
        Cls = make_cls(foo=attr.ib())
        instance = Cls(foo="foo")
        arr_ = uttr.ArrayAccessor(instance)

        assert instance.foo == "foo"
        with pytest.raises(AttributeError):
            arr_.foo
        with pytest.raises(KeyError):
            arr_["foo"]

    def test_no_uttr_quantity(self, make_cls):
        Cls = make_cls(foo=uttr.ib(unit=u.kg))
        instance = Cls(foo=1 * u.kg)
        instance.faa = 1 * u.kpc

        arr_ = uttr.ArrayAccessor(instance)

        assert instance.faa == 1 * u.kpc
        with pytest.raises(AttributeError):
            assert arr_.faa
        with pytest.raises(KeyError):
            assert arr_["faa"]

    def test_repr(self, make_cls):
        Cls = make_cls(foo=uttr.ib(unit=u.kg))
        instance = Cls(foo=1 * u.kg)
        arr_ = uttr.ArrayAccessor(instance)

        assert repr(arr_) == f"ArrayAccessor({repr(instance)})"

    def test_dir(self, make_cls):
        Cls = make_cls(foo=uttr.ib(unit=u.kg))
        instance = Cls(foo=1 * u.kg)
        arr_ = uttr.ArrayAccessor(instance)
        dir_arr = dir(arr_)

        expected = super(uttr.ArrayAccessor, arr_).__dir__() + ["foo"]
        assert set(dir_arr) == set(expected)

    def test_attrs_quantity(self, make_cls):
        Cls = make_cls(foo=uttr.ib(unit=u.kg), faa=attr.ib())
        instance = Cls(foo=1 * u.kg, faa=1 * u.kg)
        arr_ = uttr.ArrayAccessor(instance)

        assert instance.foo == 1 * u.kg
        assert arr_.foo == 1

        assert instance.faa == 1 * u.kg
        with pytest.raises(AttributeError):
            arr_.faa


class TestArrayAccessorFunction:
    def test_dimensionless_access(self, make_cls):
        Cls = make_cls(foo=uttr.ib(unit=u.kg), arr_=uttr.array_accessor())
        instance = Cls(foo=1)

        assert instance.foo == 1 * u.kg
        assert instance.arr_.foo == 1
        assert instance.arr_["foo"] == 1

    def test_equivalent_access(self, make_cls):
        Cls = make_cls(foo=uttr.ib(unit=u.kg), arr_=uttr.array_accessor())
        instance = Cls(foo=1 * u.g)

        assert instance.foo == 1 * u.g
        assert instance.arr_.foo == 0.001
        assert instance.arr_["foo"] == 0.001

    def test_same_unit_access(self, make_cls):
        Cls = make_cls(foo=uttr.ib(unit=u.kg), arr_=uttr.array_accessor())
        instance = Cls(foo=1 * u.kg)

        assert instance.foo == 1 * u.kg
        assert instance.arr_["foo"] == 1

    def test_no_uttr_access(self, make_cls):
        Cls = make_cls(foo=attr.ib(), arr_=uttr.array_accessor())
        instance = Cls(foo="foo")

        assert instance.foo == "foo"

        with pytest.raises(AttributeError):
            instance.arr_.foo
        with pytest.raises(KeyError):
            instance.arr_["foo"]

    def test_no_uttr_quantity(self, make_cls):
        Cls = make_cls(foo=attr.ib(), arr_=uttr.array_accessor())
        instance = Cls(foo=1 * u.kg)

        assert instance.foo == 1 * u.kg

        with pytest.raises(AttributeError):
            instance.arr_.foo
        with pytest.raises(KeyError):
            instance.arr_["foo"]


# =============================================================================
# uttr.s
# =============================================================================


class TestClassDecorator:
    def test_simple_decoration_maybe_cls(self):
        @uttr.s
        class Foo:
            fatt = uttr.ib(unit=u.K)

        assert "arr_" in vars(Foo)

        foo = Foo(fatt=1)

        assert isinstance(foo.arr_, uttr.ArrayAccessor)
        assert np.all(foo.arr_.fatt == foo.fatt.to_value(u.K))

    def test_simple_decoration_maybe_cls_None(self):
        @uttr.s()
        class Foo:
            fatt = uttr.ib(unit=u.K)

        assert "arr_" in vars(Foo)

        foo = Foo(fatt=1)

        assert isinstance(foo.arr_, uttr.ArrayAccessor)
        assert np.all(foo.arr_.fatt == foo.fatt.to_value(u.K))

    def test_simple_decoration_repeated_accesor_name(self):
        with pytest.raises(ValueError):

            @uttr.s
            class Foo:
                fatt = uttr.ib(unit=u.K)
                arr_ = "foo"

    def test_simple_decoration_other_name(self):
        @uttr.s(aaccessor="cosoro_")
        class Foo:
            fatt = uttr.ib(unit=u.K)

        assert "cosoro_" in vars(Foo)

        foo = Foo(fatt=1)

        assert isinstance(foo.cosoro_, uttr.ArrayAccessor)
        assert np.all(foo.cosoro_.fatt == foo.fatt.to_value(u.K))

    def test_simple_decoration_no_accessor(self):
        @uttr.s(aaccessor=None)
        class Foo:
            fatt = uttr.ib(unit=u.K)

        assert "arr_" not in vars(Foo)

        foo = Foo(fatt=1)
        with pytest.raises(AttributeError):
            foo.arr_


# =============================================================================
# uttr.s
# =============================================================================


def test_attribute_default_None():
    """https://github.com/quatrope/uttrs/issues/1"""

    @attr.s()
    class Foo:
        a = uttr.ib(default=None, unit=u.Msun)
        arr_ = uttr.array_accessor()

    foo = Foo()
    assert foo.a is None
    assert foo.arr_.a is None


def test_asdict_recursive():
    """https://github.com/quatrope/uttrs/issues/2"""

    @attr.s(frozen=True)
    class Foo:
        x = uttr.ib(unit=u.kpc)
        y = uttr.ib(unit=u.kpc)
        z = attr.ib()

        arr_ = uttr.array_accessor()

    foo = Foo(x=1, y=2, z=3)
    result = attr.asdict(foo)

    assert result == {"x": 1 * u.kpc, "y": 2 * u.kpc, "z": 3}


def test_array_accessor_order():
    """https://github.com/quatrope/uttrs/issues/7"""

    @attr.s
    class ArrAtFirst:

        x = uttr.ib(unit=u.kpc)
        y = uttr.ib(unit=u.kpc)

        arr_ = uttr.array_accessor()
        z = uttr.ib(init=False, unit=u.kpc)

        @z.default
        def _z_default(self):
            return self.arr_.x * self.arr_.y

    arrfirst = ArrAtFirst(x=1, y=2)
    assert np.all(arrfirst.arr_.z == 2)

    @attr.s
    class ArrAtLast:
        x = uttr.ib(unit=u.kpc)
        y = uttr.ib(unit=u.kpc)

        z = uttr.ib(init=False, unit=u.kpc)
        arr_ = uttr.array_accessor()

        @z.default
        def _z_default(self):
            return self.arr_.x * self.arr_.y

    arrlast = ArrAtLast(x=1, y=2)
    assert np.all(arrlast.arr_.z == 2)
