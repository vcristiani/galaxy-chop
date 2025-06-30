# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt


# =============================================================================
# DOCS
# =============================================================================

"""Common functionalities for galaxy preprocessing."""

# =============================================================================
# IMPORTS
# =============================================================================

import abc

import attr

from .. import core

# =============================================================================
# CONSTANTS
# =============================================================================

_PTYPES_ORDER = tuple(p.name.lower() for p in core.ParticleSetType)

# =============================================================================
# FUNCTIONS
# =============================================================================


def hparam(default, **kwargs):
    """
    Create a hyper parameter for decomposers.

    By design decision, hyper-parameter is required to have a sensitive default
    value.

    Parameters
    ----------
    default :
        Sensitive default value of the hyper-parameter.
    **kwargs :
        Additional keyword arguments are passed and are documented in
        ``attr.ib()``.

    Return
    ------
    Hyper parameter with a default value.

    Notes
    -----
    This function is a thin-wrapper over the attrs function ``attr.ib()``.
    """
    metadata = kwargs.pop("metadata", {})
    metadata["__gchop_model_hparam__"] = True
    return attr.ib(default=default, metadata=metadata, kw_only=True, **kwargs)


# =============================================================================
# ABC
# =============================================================================


@attr.s(frozen=False, repr=False)  # True, repr=False)
class GalaxyTransformerABC(metaclass=abc.ABCMeta):
    """Abstract class.

    to facilitate the creation of preprocessors.
    (a.k.a. Transformers).

    This class requests the redefinition of one method: transform.
    """

    __gchop_method_cls_config__ = {"repr": False, "frozen": False}  # True}

    # block meta checks =======================================================
    def __init_subclass__(cls):
        """
        Initiate of subclasses.

        It ensures that every inherited class is decorated by ``attr.s()`` and
        assigns as class configuration the parameters defined in the class
        variable `__gchop_method_cls_config__`.

        In other words it is slightly equivalent to:

        .. code-block:: python

            @attr.s(**GalaxyTransformerABC.__gchop_method_cls_config__)
            class Transformer(GalaxyTransformerABC):
                pass

        """

    # block  to implement in every method =====================================

    @abc.abstractmethod
    def transform(self, galaxy):
        """
        Preprocess method.

        Transform the particles attributes values (position,
        potential energy, etc).
        Validation of the input galaxy instance.

        Parameters
        ----------
        galaxy : ``Galaxy class`` object
            Instance of Galaxy class.

        Return
        ------
        galaxy : ``Galaxy class`` object
            Instance of Galaxy class, with the result of the
            preprocessing manipulation.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def checker(self, galaxy):
        """Checker method.

        Check if the galaxy has already been preprocessed.
        Validation of the input galaxy instance.

        Parameters
        ----------
        galaxy : ``Galaxy class`` object
            Instance of Galaxy class.

        Return
        ------
        bool
            True if galaxy had already been preprocessed,
            False otherwise.

        """
        raise NotImplementedError()

    # internal ================================================================

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        clsname = type(self).__name__

        selfd = attr.asdict(
            self,
            recurse=False,
            filter=lambda attr, _: attr.repr,
        )
        attrs_str = ", ".join([f"{k}={repr(v)}" for k, v in selfd.items()])
        return f"{clsname}({attrs_str})"
