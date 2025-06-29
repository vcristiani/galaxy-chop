# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Module to create pipelines for galaxy decomposition and transformation."""

# =============================================================================
# IMPORTS
# =============================================================================

from .core import GchopMethodABC
from .utils import Bunch, unique_names


# =============================================================================
# CLASS
# =============================================================================
class GchopPipeline(GchopMethodABC):
    """Pipeline of transforms with a final galaxy decomposition.

    Sequentially apply a list of transforms and a final decomposition.
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement `transform` method.

    The final decomposer only needs to implement `decompose`.

    The purpose of the pipeline is to assemble several steps that can be
    applied together while setting different parameters.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing decompose/transform)
        that are chained, in the order in which they are chained, with the last
        object an decomposer.

    See Also
    --------
    gchop.pipeline.mkpipe : Convenience function for simplified
        pipeline construction.

    """

    _gchop_decomp_type = "pipeline"
    _gchop_parameters = ["steps"]

    def __init__(self, steps):
        steps = list(steps)
        self._validate_steps(steps)
        self._steps = steps

    # INTERNALS ===============================================================

    def _validate_steps(self, steps):
        for name, step in steps[:-1]:
            if not isinstance(name, str):
                raise TypeError("step names must be instance of str")
            if not (hasattr(step, "transform") and callable(step.transform)):
                raise TypeError(
                    f"step '{name}' must implement 'transform()' method"
                )

        name, decomp = steps[-1]
        if not isinstance(name, str):
            raise TypeError("step names must be instance of str")
        if not (hasattr(decomp, "decompose") and callable(decomp.decompose)):
            raise TypeError(
                f"step '{name}' must implement 'decompose()' method"
            )

    # PROPERTIES ==============================================================

    @property
    def steps(self):
        """List of steps of the pipeline."""
        return list(self._steps)

    @property
    def named_steps(self):
        """Dictionary-like object, with the following attributes.

        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

        """
        return Bunch("steps", dict(self.steps))

    # DUNDERS =================================================================

    def __len__(self):
        """Return the length of the Pipeline."""
        return len(self._steps)

    def __getitem__(self, ind):
        """Return a sub-pipeline or a single step in the pipeline.

        Indexing with an integer will return an step; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying steps in the sub-pipeline
        will affect the larger pipeline and vice-versa.
        However, replacing a value in `step` will not affect a copy.

        """
        if isinstance(ind, slice):
            if ind.step not in (1, None):
                cname = type(self).__name__
                raise ValueError(f"{cname} slicing only supports a step of 1")
            return self.__class__(self.steps[ind])
        elif isinstance(ind, int):
            return self.steps[ind][-1]
        elif isinstance(ind, str):
            return self.named_steps[ind]
        raise KeyError(ind)

    # API =====================================================================

    def decompose(self, galaxy):
        """Run the all the transformers and the decomposer.

        Parameters
        ----------
        galaxy: ``Galaxy class`` object

        Returns
        -------
        r : Result
            Whatever the last step (decomposer) returns from their decompose
            method.


        """
        galaxy = self.transform(galaxy)
        _, decomp = self.steps[-1]
        result = decomp.decompose(galaxy)

        return result

    def transform(self, galaxy):
        """Run the all the transformers.

        Parameters
        ----------
        galaxy: ``Galaxy class`` object

        Returns
        -------
        galaxy: ``Galaxy class`` object.
            Transformed galaxy.

        """
        for _, step in self.steps[:-1]:
            galaxy = step.transform(galaxy)
        return galaxy


# =============================================================================
# FACTORY
# ==========================z===================================================


def mkpipe(*steps):
    """Construct a Pipeline from the given transformers and decomposer.

    This is a shorthand for the GchopPipeline constructor; it does not require,
    and does not permit, naming the estimators. Instead, their names will
    be set to the lowercase of their types automatically.

    Parameters
    ----------
    *steps: list of transformers and decomposer object
        List of the galaxy-chop transformers and decomposer
        that are chained together.

    Returns
    -------
    p : GchopPipeline
        Returns a galaxy-chop :class:`GchopPipeline` object.

    """
    names = [type(step).__name__.lower() for step in steps]
    named_steps = unique_names(names=names, elements=steps)
    return GchopPipeline(named_steps)
