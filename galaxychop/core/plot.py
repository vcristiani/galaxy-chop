# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Plot helper for the galaxy object."""

# =============================================================================
# IMPORTS
# =============================================================================

from collections import OrderedDict

import attr

import numpy as np

import pandas as pd

import seaborn as sns

from .. import constants as const, models, preproc

# =============================================================================
# ACCESSOR
# =============================================================================


@attr.s(frozen=True, order=False)
class GalaxyPlotter:
    """Make plots of a Galaxy."""

    _P_KIND_FORBIDEN_METHODS = ("get_df_and_hue", "get_circ_df_and_hue")
    _DEFAULT_HUE_COLUMN = "Labels"
    _DEFAULT_HUE_COUNT_COLUMN = "LabelsCnt"

    _galaxy = attr.ib()

    # INTERNAL ================================================================

    def __call__(self, plot_kind="pairplot", **kwargs):
        """Make plots of the galaxy.

        Parameters
        ----------
        kind : str
            The kind of plot to produce:
                - 'pairplot' : pairplot matrix of any coordinates (default)

        **kwargs
            Options to pass to subjacent plotting method.

        Returns
        -------
        :class:`matplotlib.axes.Axes` or numpy.ndarray of them
           The ax used by the plot

        """
        if (
            plot_kind.startswith("_")
            or plot_kind in self._P_KIND_FORBIDEN_METHODS
        ):
            raise ValueError(f"invalid 'plot_kind' name '{plot_kind}'")
        method = getattr(self, plot_kind, None)
        if not callable(method):
            raise ValueError(f"invalid 'plot_kind' name '{plot_kind}'")
        return method(**kwargs)

    # COMMON PLOTS ============================================================

    def get_df_and_hue(self, ptypes, attributes, labels, lmap):
        """
        Dataframe and Hue constructor for plot implementations.

        Parameters
        ----------
        ptypes : keys of ``ParticleSet class`` parameters.
            Particle type.
        attributes : keys of ``ParticleSet class`` parameters.
            Names of ``ParticleSet class`` parameters.
        labels : keys of ``ParticleSet class`` parameters.
            Variable to map plot aspects to different colors.
        lmap :  dicts
            Name assignment to the label.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame of galaxy properties with labels added.
        hue : keys of ``ParticleSet class`` parameters.
            Labels of all galaxy particles.
        """
        # if we use the components as labels we need to extract the labels
        # and the lmap if lmap is None
        if isinstance(labels, models.Components):
            lmap = labels.lmap if lmap is None else lmap
            labels = labels.labels

        attributes = ["x", "y", "z"] if attributes is None else attributes

        hue = None  # by default labels is None

        # labels: column used to map plot aspects to different colors (hue).
        # if is a str and it was not in the attributes I have to take it out
        # of the dataframe.
        if isinstance(labels, str):
            hue = labels
            attributes = np.unique(list(attributes) + [labels])

        # put all attributes in a df
        df = self._galaxy.to_dataframe(ptypes=ptypes, attributes=attributes)

        # labels can be an np array and must be added as a column to the
        # dataframe and assign hue to the name of this new column.
        if hue is None and labels is not None:
            hue = (
                self._DEFAULT_HUE_COLUMN
            )  # Hue is not in ParticleSet, so it is useful
            df.insert(0, hue, labels)  # I place it as the first column

        if hue and lmap is not None:
            lmap_func = (
                (lambda label: lmap.get(label, label))
                if isinstance(lmap, dict)
                else lmap
            )
            df[hue] = df[hue].apply(lmap_func)

        # for consitency if we have a hue, we use the natural order
        if hue is not None:
            df[hue] = df[hue].astype("category")

            hue_count = df[hue].value_counts()
            df[self._DEFAULT_HUE_COUNT_COLUMN] = df[hue].map(hue_count)

            df = df.sort_values(
                by=self._DEFAULT_HUE_COUNT_COLUMN, ascending=True
            )

            del df[self._DEFAULT_HUE_COUNT_COLUMN]

        return df, hue

    def pairplot(
        self,
        *,
        ptypes=None,
        attributes=None,
        labels="ptype",
        lmap=None,
        **kwargs,
    ):
        """
        Draw a pairplot of the galaxy properties.

        By default, this function will create a grid of Axes such that each
        numeric variable in data will by shared across the y-axes across a
        single row and the x-axes across a single column. The diagonal
        plots drow a univariate distribution to show the marginal distribution
        of the data in each column.
        This function groups the values of all galaxy particles according to
        some ``ParticleSet class`` parameter.

        Parameters
        ----------
        ptypes : keys of ``ParticleSet class`` parameters.
            Particle type. Default value = None
        attributes : keys of ``ParticleSet class`` parameters.
            Names of ``ParticleSet class`` parameters. Default value = None
        labels : keys of ``ParticleSet class`` parameters.
            Variable to map plot aspects to different colors.
            Default value = None
        lmap :  dicts
            Name assignment to the label.
            Default value = None
        **kwargs :
            Additional keyword arguments are passed and are documented in
            ``seaborn.pairplot``.

        Returns
        -------
        seaborn.axisgrid.PairGrid
        """
        df, hue = self.get_df_and_hue(
            ptypes=ptypes,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )

        kwargs.setdefault("kind", "hist")
        kwargs.setdefault("diag_kind", "kde")

        ax = sns.pairplot(data=df, hue=hue, **kwargs)
        return ax

    def scatter(self, x, y, *, ptypes=None, labels=None, lmap=None, **kwargs):
        """Draw a scatter plot of galaxy properties.

        Shows the relationship between x and y. This function groups the values
        of all galaxy particles according to some ``ParticleSet class``
        parameter.

        Parameters
        ----------
        x, y : keys of ``ParticleSet class`` parameters.
            Variables that specify positions on the x and y axes.
            Default value y = None.
        ptypes : keys of ``ParticleSet class`` parameters.
            Particle type. Default value = None
        labels : keys of ``ParticleSet class`` parameters.
            Variable to map plot aspects to different colors.
            Default value = None
        lmap :  dict
            Name assignment to the label.
            Default value = None
        **kwargs
            Additional keyword arguments are passed and are documented
            in ``seaborn.scatterplot``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        attributes = [x, y]
        df, hue = self.get_df_and_hue(
            ptypes=ptypes,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        kwargs.setdefault("marker", ".")
        ax = sns.scatterplot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax

    def hist(
        self, x, *, y=None, ptypes=None, labels=None, lmap=None, **kwargs
    ):
        """Draw a histogram of galaxy properties.

        Plot univariate or bivariate histograms to show distributions of
        datasets. This function groups the values of all galaxy particles
        according to some ``ParticleSet class`` parameter.

        Parameters
        ----------
        x, y : keys of ``ParticleSet class`` parameters.
            Variables that specify positions on the x and y axes.
            Default value y = None.
        ptypes : keys of ``ParticleSet class`` parameters.
            Particle type. Default value = None
        labels : keys of ``ParticleSet class`` parameters.
            Variable to map plot aspects to different colors.
            Default value = None
        lmap :  dicts
            Name assignment to the label.
            Default value = None
        **kwargs
            Additional keyword arguments are passed and are documented
            in ``seaborn.histplot``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        attributes = [x] if y is None else [x, y]
        df, hue = self.get_df_and_hue(
            ptypes=ptypes,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        ax = sns.histplot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax

    def kde(self, x, *, y=None, ptypes=None, labels=None, lmap=None, **kwargs):
        """Draw a Kernel Density plot of galaxy properties.

        Plot univariate or bivariate distributions using kernel density
        estimation (KDE). This plot represents the galaxy properties using
        a continuous probability density curve in one or more dimensions.
        This function groups the values of all galaxy particles according
        to some ``ParticleSet class`` parameter.

        Parameters
        ----------
        x, y : keys of ``ParticleSet class`` parameters.
            Variables that specify positions on the x and y axes.
            Default value y = None.
        ptypes : keys of ``ParticleSet class`` parameters.
            Particle type. Default value = None
        labels : keys of ``ParticleSet class`` parameters.
            Variable to map plot aspects to different colors.
            Default value = None
        lmap :  dicts
            Name assignment to the label. Default value = None
        **kwargs
            Additional keyword arguments are passed and are documented
            in ``seaborn.kdeplot``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        attributes = [x] if y is None else [x, y]
        df, hue = self.get_df_and_hue(
            ptypes=ptypes,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        ax = sns.kdeplot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax

    # CICULARITY ==============================================================

    def get_sdyn_df_and_hue(self, sdyn_kws, attributes, labels, lmap):
        """
        Dataframe and Hue constructor for plot implementations.

        Parameters
        ----------
        sdyn_kws: dict or None
            Extra parameters for galaxy.stellar_dynamics() method.
        attributes : keys of ``GalaxyStellarDynamics`` dataframe.
            Keys of the normalized specific energy, the circularity parameter
            (J_z/J_circ) and/or the projected circularity parameter
            (J_p/J_circ) of the stellar particles.
        labels : keys of ``GalaxyStellarDynamics`` dataframe.
            Variable to map plot aspects to different colors.
        lmap :  dict
            Name assignment to the label.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame of the normalized specific energy, the circularity
            parameter (J_z/J_circ) and/or the projected circularity parameter
            (J_p/J_circ) of the stellar particles with labels added.
        hue : keys of ``GalaxyStellarDynamics`` dataframe.
            Labels of stellar particles.
        """
        # if we use the components as laberls we need to extract the labels
        # and the lmap if lmap is None
        if isinstance(labels, models.Components):
            lmap = labels.lmap if lmap is None else lmap
            labels = labels.labels

        # first we extract the circularity parameters from the galaxy
        # as a dictionary
        sdyn_kws = {} if sdyn_kws is None else sdyn_kws
        sdyn = self._galaxy.stellar_dynamics(**sdyn_kws)
        mask = sdyn.isfinite()

        sdyn_dict = sdyn.as_dict()

        # determine the correct number of attributes
        attributes = (
            list(sdyn_dict.keys()) if attributes is None else attributes
        )
        hue = None

        # labels: column used to map plot aspects to different colors (hue).
        # if is a str and it was not in the attributes but we can retrieve from
        # circ, we add as an attribute
        if isinstance(labels, str):
            hue = labels
            attributes = np.unique(list(attributes) + [labels])

        columns = OrderedDict()
        for aname in attributes:
            columns[aname] = sdyn_dict[aname][mask]

        df = pd.DataFrame(columns)  # here we create the dataframe

        # At this point if "hue" is still "None" we can assume:
        # is an array simply paste it into the dataframe.
        if hue is None and labels is not None:
            # if the labels are passed to me as an array,
            # I only delete the nans and inf.
            labels = np.asarray(labels)
            labels = labels[np.isfinite(labels)]
            hue = self._DEFAULT_HUE_COLUMN

            # I place it as the first column
            df.insert(0, hue, labels)

        if hue and lmap is not None:
            lmap_func = (
                (lambda label: lmap.get(label, label))
                if isinstance(lmap, dict)
                else lmap
            )
            df[hue] = df[hue].apply(lmap_func)

        # for consitency if we have a hue, we use the natural order
        if hue is not None:
            df[hue] = df[hue].astype("category")

            hue_count = df[hue].value_counts()
            df[self._DEFAULT_HUE_COUNT_COLUMN] = df[hue].map(hue_count)

            df = df.sort_values(
                by=self._DEFAULT_HUE_COUNT_COLUMN, ascending=True
            )

        return df, hue

    def sdyn_pairplot(
        self,
        *,
        attributes=None,
        labels=None,
        lmap=None,
        sdyn_kws=None,
        **kwargs,
    ):
        """
        Draw a pairplot of stellar dynamics.

        By default, this function will create a grid of Axes such that each
        numeric variable in data will by shared across the y-axes across a
        single row and the x-axes across a single column. The diagonal
        plots drow a univariate distribution to show the marginal distribution
        of the data in each column.
        This function groups the values of stellar particles according to some
        keys of ``JCirc`` tuple.

        Parameters
        ----------
        attributes : keys of ``GalaxyStarsDynamics class`` parameters.
            Names of ``GalaxyStarsDynamics class`` parameters.
            Default value = None
        labels : keys of ``JCirc`` tuple.
            Variable to map plot aspects to different colors.
            Default value = None
        lmap :  dicts
            Name assignment to the label. Default value = None
        sdyn_kws: dict
            Extra parameters for galaxy.stellar_dynamics() method.
        **kwargs :
            Additional keyword arguments are passed and are documented in
            ``seaborn.pairplot``.

        Returns
        -------
        seaborn.axisgrid.PairGrid
        """
        df, hue = self.get_sdyn_df_and_hue(
            attributes=attributes,
            labels=labels,
            sdyn_kws=sdyn_kws,
            lmap=lmap,
        )

        kwargs.setdefault("kind", "hist")
        kwargs.setdefault("diag_kind", "kde")

        ax = sns.pairplot(data=df, hue=hue, **kwargs)
        return ax

    def sdyn_scatter(
        self,
        x,
        y,
        *,
        labels=None,
        lmap=None,
        sdyn_kws=None,
        **kwargs,
    ):
        """Draw a scatter plot of stellar dynamics.

        Shows the relationship between x and y. This function groups the values
        of stellar particles according to some keys of ``JCirc`` tuple.

        Parameters
        ----------
        x, y : keys of ``JCirc`` tuple.
            Variables that specify positions on the x and y axes.
            Default value y = None.
        labels : keys of ``JCirc`` tuple.
            Variable to map plot aspects to different colors.
            Default value = None
        lmap :  dicts
            Name assignment to the label. Default value = None
        sdyn_kws: dict
            Extra parameters for galaxy.stellar_dynamics() method.
        **kwargs
            Additional keyword arguments are passed and are documented
            in ``seaborn.scatterplot``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        attributes = [x, y]
        df, hue = self.get_sdyn_df_and_hue(
            sdyn_kws=sdyn_kws,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        kwargs.setdefault("marker", ".")
        ax = sns.scatterplot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax

    def sdyn_hist(
        self,
        x,
        *,
        y=None,
        labels=None,
        lmap=None,
        sdyn_kws=None,
        **kwargs,
    ):
        """Draw a histogram of stellar dynamics.

        Plot univariate or bivariate histograms to show distributions of
        datasets. This function groups the values of stellar particles
        according to some keys of ``JCirc`` tuple.

        Parameters
        ----------
        x, y : keys of ``JCirc`` tuple.
            Variables that specify positions on the x and y axes.
            Default value y = None.
        labels : keys of ``JCirc`` tuple.
            Variable to map plot aspects to different colors.
            Default value = None
        lmap :  dicts
            Name assignment to the label. Default value = None
        sdyn_kws: dict
            Extra parameters for galaxy.stellar_dynamics() method.
        **kwargs
            Additional keyword arguments are passed and are documented
            in ``seaborn.histplot``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        attributes = [x] if y is None else [x, y]
        df, hue = self.get_sdyn_df_and_hue(
            sdyn_kws=sdyn_kws,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        ax = sns.histplot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax

    def sdyn_kde(
        self,
        x,
        *,
        y=None,
        labels=None,
        lmap=None,
        sdyn_kws=None,
        **kwargs,
    ):
        """Draw a Kernel Density plot of stellar dynamics.

        Plot univariate or bivariate distributions using kernel density
        estimation (KDE). This plot represents normalized specific energy, the
        circularity parameter (J_z/J_circ) and/or the projected circularity
        parameter (J_p/J_circ)  of the stellar particles using a continuous
        probability density curve in one or more dimensions.
        This function groups the values of stellar particles according
        to some keys of ``JCirc`` tuple.

        Parameters
        ----------
        x, y : keys of ``JCirc`` tuple.
            Variables that specify positions on the x and y axes.
            Default value y = None.
        labels : keys of ``JCirc`` tuple.
            Variable to map plot aspects to different colors.
            Default value = None
        lmap :  dicts
            Name assignment to the label. Default value = None
        sdyn_kws: dict
            Extra parameters for galaxy.stellar_dynamics() method.
        **kwargs
            Additional keyword arguments are passed and are documented
            in ``seaborn.kdeplot``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        attributes = [x] if y is None else [x, y]
        df, hue = self.get_sdyn_df_and_hue(
            sdyn_kws=sdyn_kws,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        ax = sns.kdeplot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax
