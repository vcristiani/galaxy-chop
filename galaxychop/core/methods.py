# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt


# =============================================================================
# DOCS
# =============================================================================

"""Core functionalities of GchopPipeline."""

# =============================================================================
# IMPORTS
# =============================================================================ç

import abc
import copy
import inspect


# =============================================================================
# BASE TRANSFORMER AND DECOMPOSER PIPELINE CLASS
# =============================================================================


class GchopMethodABC(metaclass=abc.ABCMeta):
    """
    Base class for the pipeline of GalaxyChop.

    Notes
    -----
    All subclasses should specify:

    - ``_gchop_decomp_type``: The type of the decomposer.
    - ``_gchop_parameters``: Availebe parameters.
    - ``_gchop_abstract_class``: If the class is abstract.

    If the class is *abstract* all validations are turned off.

    """

    _gchop_abstract_class = True

    def __init_subclass__(cls):
        """Validate if the subclass are well formed."""
        is_abstract = vars(cls).get("_gchop_abstract_class ", False)
        if is_abstract:
            return

        decomp_type = getattr(cls, "_gchop_decomp_type", None)
        if decomp_type is None:
            raise TypeError(f"{cls} must redefine '_gchop_decomp_type'")
        cls._gchop_decomp_type = str(decomp_type)

        params = getattr(cls, "_gchop_parameters", None)
        if params is None:
            raise TypeError(f"{cls} must redefine '_gchop_parameters'")

        params = frozenset(params)

        signature = inspect.signature(cls.__init__)
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )

        params_not_in_signature = params.difference(signature.parameters)
        if params_not_in_signature and not has_kwargs:
            raise TypeError(
                f"{cls} defines the parameters {params_not_in_signature} "
                "which is not found as a parameter in the __init__ method."
            )

        cls._gchop_parameters = params

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        cls_name = type(self).__name__

        parameters = []
        if self._gchop_parameters:
            for pname in sorted(self._gchop_parameters):
                pvalue = getattr(self, pname)
                parameters.append(f"{pname}={repr(pvalue)}")

        str_parameters = ", ".join(parameters)
        return f"<{cls_name} [{str_parameters}]>"

    def get_parameters(self):
        """Return the parameters of the method as dictionary."""
        the_parameters = {}
        for parameter_name in self._gchop_parameters:
            parameter_value = getattr(self, parameter_name)
            the_parameters[parameter_name] = copy.deepcopy(parameter_value)
        return the_parameters

    def copy(self, **kwargs):
        """Return a custom copy of the current decomposer.

        This method is also useful for manually modifying the values of the
        object.

        Parameters
        ----------
        kwargs :
            The same parameters supported by object constructor. The values
            provided replace the existing ones in the object to be copied.

        Returns
        -------
        A new object.

        """
        asdict = self.get_parameters()

        asdict.update(kwargs)

        cls = type(self)
        return cls(**asdict)
