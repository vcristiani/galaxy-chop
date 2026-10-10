# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt
#
# The uttr module is adapted from uttrs (https://github.com/quatrope/uttrs),
# by Juan B Cabral and QuatroPe, relicensed under MIT by its authors.


# =============================================================================
# DOCS
# =============================================================================

"""uttrs bridge between attrs and Astropy units [1]_.

uttrs seeks to interoperate Classes defined using attrs and Astropy units
in a simple manner with two main functionalities:

- ``uttr.ib`` which generates attributes sensitive to units.
- ``uttr.array_accessor`` which allows access to attributes linked to units,
  and transform them into numpy arrays.
- ``uttr.s`` a class decorator to automatically add the `array_accessor`.


References
----------
.. [1] Price-Whelan, Adrian M., et al. "The Astropy project:
   Building an open-science project and status of the v2. 0 core
   package." The Astronomical Journal 156.3 (2018): 123.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import astropy.units as u

import attr


# =============================================================================
# METADATA
# =============================================================================

__all__ = ["s", "ib", "attribute", "array_accessor"]


__version__ = "0.5"


# =============================================================================
# CONSTANTS AND METADATA
# =============================================================================

UTTR_METADATA = "_uttr_ucav"


# =============================================================================
# ATTRIBUTE IMPLEMENTATION
# =============================================================================


@attr.s(frozen=True)
class UnitConverterAndValidator:
    """Converter and validator of astropy.units for attrs.

    Parameters
    ----------
    unit : astropy.units.UnitBase
        The base units for attribute default unit assignation and validation
        of inputs.

    """

    unit: u.UnitBase = attr.ib(
        validator=attr.validators.instance_of(u.UnitBase)
    )

    def is_dimensionless(self, v):
        """Return true if v is dimensionless."""
        return (
            not isinstance(v, u.Quantity) or v.unit == u.dimensionless_unscaled
        )

    def to_array(self, v):
        """Convert the quantity to an array of the given unit."""
        return v.to_value(self.unit)

    def convert_if_dimensionless(self, value):
        """Assign a unit to a dimensionless object.

        If the object already has a dimension it returns it without change

        Examples
        --------
        >>> uc = UnitConverter(u.km)

        >>> uc.convert_if_dimensionless(1)  # dimensionless then convert
        '<Quantity 1. km>'

        >>> # the same object is returned
        >>> uc.convert_if_dimensionless(1 * u.kpc)
        '<Quantity 1. kpc>'

        """
        if self.is_dimensionless(value) and value is not None:
            return value * self.unit
        return value

    def validate_is_equivalent_unit(self, instance, attribute, value):
        """Validate that the unit equivalence with. the configured unit.

        This method follows the suggested signature by attrs validators.

        - the instance that’s being validated (aka self),
        - the attribute that it’s validating, and finally
        - the value that is passed for it.

        Raises
        ------
        ValueError:
            If the value has a non-equivalent dimesion to unit.

        """
        if self.is_dimensionless(value):
            return

        # creamos una solo escalar con la unidad del quantity
        # y tratamos de convertir eso. Eso evita hacer la prueba
        # con un array muy grande
        unity = 1 * value.unit
        try:
            unity.to(self.unit)
        except u.UnitConversionError:
            unit, aname, ufound = self.unit, attribute.name, value.unit
            raise ValueError(
                f"Unit of attribute '{aname}' must be equivalent to '{unit}'."
                f" Found '{ufound}'."
            )


def attribute(unit: u.UnitBase = None, **kwargs):
    """Create a new attribute with converters and validators for a given unit.

    Parameters
    ----------
    unit : u.UnitBase or None
        The unit to use in the converters and the attribute validator.
        If it's None, the call is equivalent to ``attr.ib(**kwargs)``
    kwargs :
        Extra parameter of attr.ib()


    Example
    -------
    >>> @attr.s()
    ... class Foo:
    ...     p = unit_attribute(unit=(u.km / u.s))
    >>>> Foo(p=[1, 2, 3])
    Foo(p=<Quantity [1., 2., 3.] km / s>)

    @attr.s()
    >>> class Foo:
    ...     p = unit_attribute(unit=(u.km / u.h))

    >>> Foo(p=[1, 2, 3])
    Foo(p=<Quantity [1., 2., 3.] km / h>)

    >>> @attr.s()
    ... class Foo:
    ...     p = unit_attribute(unit=(u.km / u.s))

    >>> Foo(p=[1, 2, 3] * u.km / u.h)
    Foo(p=<Quantity [1., 2., 3.] km / h>)

    >>> Foo(p=[1, 2, 3] * u.kpc)
    ValueError: Unit of attribute 'p' must be equivalent to 'km / s'.
    Found 'kpc'.

    """
    if unit is None:  # si unit es None entonces uso attrib a la antigua
        return attr.ib(**kwargs)

    ucav = UnitConverterAndValidator(unit=unit)

    # si ya habia validadores los saco o creo una nueva lista
    # (puede ser solo una funcion asi que lo meto a una lista)
    # para asignar mi validador de unidades
    validator = kwargs.pop("validator", [])
    if callable(validator):
        validator = [validator]
    validator.append(ucav.validate_is_equivalent_unit)

    # si habia converters lo saco
    # (puede ser solo una funcion asi que lo meto a una lista)
    # y meto ahi mi conversor al final de todo
    converter = kwargs.pop("converter", [])
    if callable(converter):
        converter = [converter]
    converter.append(ucav.convert_if_dimensionless)

    metadata = kwargs.pop("metadata", {})
    metadata[UTTR_METADATA] = ucav

    return attr.ib(
        validator=validator, converter=converter, metadata=metadata, **kwargs
    )


#: Equivalent to ``uttr.attribute`` to use like *attr.ib*.
ib = attribute


# =============================================================================
# ARRAY ACCESORS
# =============================================================================


class ArrayAccessor:
    """Automatic converter of the ``uttrs`` attributes in ``numpy.ndarray``.

    Instances of ArrayAccessor (``arr_``) access to the attributes
    (defined with uttrs) of the provided instance, and if they are of
    ``atropy.units.Quantity`` type it converts them into ``numpy.ndarray``.

    If you try to access a attribute no defined by uttrs, an
    ``AttributeErrror`` is raised.

    Examples
    --------
    >>> @attr.s()
    ... class Foo:
    ...     quantity = uttr.ib(unit=u.km)
    ...     array = attr.ib()

    >>> foo = Foo(
    ...     quantity=u.Quantity([1, 2]),
    ...     array=np.array([1, 2]),
    ... )

    >>> arr_ = ArrayAccessor(foo)

    >>> arr_
    ArrayAccessor(
        Foo(quantity=<Quantity [1., 2.]>, array=array([1, 2]))

    >>> arr_.quantity
    array([1., 2., 3.])

    >>> arr_.array
    AttributeError("No uttr.Attribute 'array'")

    """

    def __init__(self, instance):
        self._instance = instance

        # filter only the uttrs attributes
        # the ones with the UTTR_METADATA
        ifields_dict = attr.fields_dict(type(instance))
        self._fields_dict = {
            aname: attrib
            for aname, attrib in ifields_dict.items()
            if UTTR_METADATA in attrib.metadata
        }

    def __repr__(self):
        """repr(x) <==> x.__repr__()."""
        return f"ArrayAccessor({repr(self._instance)})"

    def __dir__(self):
        """dir(x) <==> x.__dir__()."""
        return super().__dir__() + list(self._fields_dict)

    def __getitem__(self, k):
        """x[k] <==> x.__getitem__(k)."""
        try:
            return self.__getattr__(k)
        except AttributeError:
            raise KeyError(k)

    def __getattr__(self, a):
        """getattr(x, y) <==> x.__getattr__(y) <==> getattr(x, y)."""
        fd = self._fields_dict
        if a in fd:
            v = getattr(self._instance, a)
            if v is None:
                return
            ucav = fd[a].metadata[UTTR_METADATA]
            arr = ucav.to_array(v)
            return arr

        raise AttributeError(f"No uttr.Attribute '{a}'")


def array_accessor():
    """Provide an ArrayAccessor attribute to an attrs based class.

    This new attribute allows access to any other uttrs defined attribute or of
    the class. It converts it to the default unit of the attribute
    and afterward to a `numpy.ndarray`.

    If you try to access an attribute no defined by uttrs, an
    ``AttributeErrror`` is raised.

    Example
    -------
    >>> @attr.s()
    ... class Foo:
    ...     q = uttr.ib(unit=u.kg)
    ...     a = attr.ib()
    ...     arr_ = array_accessor()

    >>> foo = Foo(q=[1, 2, 3] * u.kg, a=np.array([1, 2, 3]))
    >>> foo
    Foo(q=<Quantity [1., 2., 3.] kg>, a=array([1, 2, 3]))

    >>> foo.q
    <Quantity [1., 2., 3.] kg>

    >>> foo.arr_.q
    array([1., 2., 3.])

    >>> foo.a
    array([1, 2, 3])

    >>> foo.arr_.a
    array([1, 2, 3])

    """

    @property
    def array_accessor_property(self):
        return ArrayAccessor(instance=self)

    return array_accessor_property


# =============================================================================
# CLASS DECORATOR
# =============================================================================


def s(maybe_cls=None, *, aaccessor="arr_", **kwargs):
    r"""Class decorator to automatically add an array accessor to the class.

    The behaviour is the same as
    `attr.s <https://www.attrs.org/en/stable/api.html#attr.s>`_ function
    but also automatically creates an ``uttrs.ArrayAccessor`` property with
    defined by `aaccessor`.

    Parameters
    ----------
    aaccessor: str or None, default 'arr_`
        Name of the array accessor property. If is None, no property is added.
    maybe_cls: class or None, default None.
        Same behavior of ``attr.s()`` maybe_cls parameter.
    kwargs:
        Same parameter as ``attr.s()``.

    Examples
    --------
    The next two codes are equivalent

    .. code-block:: python

        import astropy.units as u
        import uttr

        @attr.s
        class Foo:
            attribute = uttr.ib(unit=u.K)
            arr_ = uttr.array_accessor()

    .. code-block:: python

        import astropy.units as u
        import uttr

        @uttr.s
        class Foo:
            attribute = uttr.ib(unit=u.K)

    """

    def wrap(cls):

        if aaccessor is not None:
            if hasattr(cls, aaccessor):
                raise ValueError(
                    f"{cls} already has an attribute with name {aaccessor}"
                )
            setattr(cls, aaccessor, array_accessor())

        attr_s = attr.s(**kwargs)
        return attr_s(cls)

    if maybe_cls is None:
        return wrap
    return wrap(maybe_cls)
