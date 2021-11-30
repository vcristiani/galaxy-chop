# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) 2020, Valeria Cristiani
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

from . import utils

# =============================================================================
# ACCESSOR
# =============================================================================


@attr.s(frozen=True, order=False)
class GalaxyPlotter:
    """Make plots of DecisionMatrix."""

    _P_KIND_FORBIDEN_METHODS = ("get_df_and_hue", "get_circ_df_and_hue")

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
        attributes = ["x", "y", "z"] if attributes is None else attributes

        hue = None  # by default not hue is selected

        # labels: column used to map plot aspects to different colors (hue).
        # if is a str and it was not in the attributes I have to take it out
        # of the dataframe.
        if isinstance(labels, str) and labels not in attributes:
            hue = labels
            attributes = np.concatenate((attributes, [labels]))

        # put all attributes in a df
        df = self._galaxy.to_dataframe(ptypes=ptypes, attributes=attributes)

        # labels can be an np array and must be added as a column to the
        # dataframe and assign hue to the name of this new column.
        if hue is None:
            hue = "Labels"  # Hue is not in ParticleSet, so it is useful
            df.insert(0, hue, labels)  # I place it as the first column

        if lmap is not None:
            df[hue] = df[hue].apply(lambda l: lmap.get(l, l))

        return df, hue

    def pairplot(
        self, ptypes=None, attributes=None, labels="ptype", lmap=None, **kwargs
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

    def dis(self, x, y=None, ptypes=None, labels=None, lmap=None, **kwargs):
        """Draw a distribution plots onto a FacetGrid.

        Plot univariate or bivariate distributions of datasets using
        different approachs for visualizing the galaxy parameters.
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
            Name assignment to the label.
            Default value = None
        **kwargs
            Additional keyword arguments are passed and are documented in
            ``seaborn.displot``.

        Returns
        -------
        seaborn.axisgrid.PairGrid
        """
        attributes = [x] if y is None else [x, y]
        df, hue = self.get_df_and_hue(
            ptypes=ptypes,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        ax = sns.displot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax

    def scatter(self, x, y, ptypes=None, labels=None, lmap=None, **kwargs):
        """Draw a scatter plot of galaxy properties.

        Shows the relationship between x and y.
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
        ax = sns.scatterplot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax

    def hist(self, x, y=None, ptypes=None, labels=None, lmap=None, **kwargs):
        """Draw a histogram of galaxy properties.

        Plot univariate or bivariate histograms to show distributions
        of datasets. This function groups the values of all galaxy
        particles according to some ``ParticleSet class`` parameter.

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

    def kde(self, x, y=None, ptypes=None, labels=None, lmap=None, **kwargs):
        """Draw a Kernel Density plot of galaxy properties.

        Plot univariate or bivariate distributions using kernel density
        estimation (KDE). This plot represents the galay properties using
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
            Name assignment to the label.
            Default value = None
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

    def get_circ_df_and_hue(self, cbins, attributes, labels, lmap):
        # first we extract the circularity parameters from the galaxy
        # as a dictionary
        circ = utils.jcirc(self._galaxy, *cbins)._asdict()

        mask = (
            np.isfinite(circ["normalized_star_energy"])
            & np.isfinite(circ["eps"])
            & np.isfinite(circ["eps_r"])
        )

        # determine the correct number of attributes
        attributes = (
            [k for k in circ.keys() if k not in ("x", "y")]
            if attributes is None
            else attributes
        )
        hue = None  # by default no hue is selected

        # labels: column used to map plot aspects to different colors (hue).
        # if is a str and it was not in the attributes but we can retrieve from
        # circ, we add as an attribute
        if (
            isinstance(labels, str)  # must be an string
            and labels not in attributes  # is not in attributes
            and labels in circ  # but is in circ
        ):
            attributes = np.concatenate((attributes, [labels]))
            hue = labels

        columns = OrderedDict()
        for aname in attributes:
            columns[aname] = circ[aname][mask]

        df = pd.DataFrame(columns)  # here we create the dataframe

        # At this point if "hue" is still "None" we can assume:
        # 1. if labels is a str then labels is a column of space.
        # real galaxy column
        # 2. Or if it is an array simply paste it into the dataframe.
        if hue is None and labels is not None:

            if isinstance(labels, str):
                hue = labels
                sdf = self._galaxy.stars.to_dataframe(attributes=[labels])
                labels = sdf[labels].values[mask]
            else:
                hue = "Labels"

                # si me pasaron los labels como "array"
                # solo borro los nans e inf
                labels = labels[np.isfinite(labels)]

            # I place it as the first column
            df.insert(0, hue, labels)

        if lmap is not None:
            df[hue] = df[hue].apply(lambda l: lmap.get(l, l))

        return df, hue

    def circ_pairplot(
        self,
        cbins=utils.DEFAULT_CBIN,
        attributes=None,
        labels=None,
        lmap=None,
        **kwargs,
    ):

        df, hue = self.get_circ_df_and_hue(
            cbins=cbins, attributes=attributes, labels=labels, lmap=lmap
        )

        kwargs.setdefault("kind", "hist")
        kwargs.setdefault("diag_kind", "kde")

        ax = sns.pairplot(data=df, hue=hue, **kwargs)
        return ax

    def circ_dis(
        self,
        x,
        y=None,
        cbins=utils.DEFAULT_CBIN,
        labels=None,
        lmap=None,
        **kwargs,
    ):
        attributes = [x] if y is None else [x, y]
        df, hue = self.get_circ_df_and_hue(
            cbins=cbins,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        ax = sns.displot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax

    def circ_scatter(
        self,
        x,
        y,
        cbins=utils.DEFAULT_CBIN,
        labels=None,
        lmap=None,
        **kwargs,
    ):
        attributes = [x, y]
        df, hue = self.get_circ_df_and_hue(
            cbins=cbins,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        ax = sns.scatterplot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax

    def circ_hist(
        self,
        x,
        y=None,
        cbins=utils.DEFAULT_CBIN,
        labels=None,
        lmap=None,
        **kwargs,
    ):
        attributes = [x] if y is None else [x, y]
        df, hue = self.get_circ_df_and_hue(
            cbins=cbins,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        ax = sns.histplot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax

    def circ_kde(
        self,
        x,
        y=None,
        cbins=utils.DEFAULT_CBIN,
        labels=None,
        lmap=None,
        **kwargs,
    ):
        attributes = [x] if y is None else [x, y]
        df, hue = self.get_circ_df_and_hue(
            cbins=cbins,
            attributes=attributes,
            labels=labels,
            lmap=lmap,
        )
        ax = sns.kdeplot(x=x, y=y, data=df, hue=hue, **kwargs)
        return ax
