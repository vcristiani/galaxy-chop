# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

# WIP

import galaxychop as gchop

import numpy as np

import pytest


# =============================================================================
# TRANSFORMER ABC
# =============================================================================


@pytest.mark.model
def GalaxyTransformerABC_not_implemethed():
    class Transformer(gchop.preproc._base.GalaxyTransformerABC):
        def transform(self, galaxy):
            return super().transform(galaxy)

        def checker(self, galaxy):
            return super().checker(galaxy)

    transformer = Transformer()

    with pytest.raises(NotImplementedError):
        transformer.transform(None)

    with pytest.raises(NotImplementedError):
        transformer.checker(None)


@pytest.mark.model
def test_GalaxyTransformerABC_repr():
    class Transformer(gchop.preproc._base.GalaxyTransformerABC):
        def transform(self, galaxy):
            ...

        def checker(self, galaxy):
            ...

    transformer = Transformer()
    result = repr(transformer)
    expected = "Transformer()"

    assert result == expected


@pytest.mark.model
def test_GalaxyTransformerABC_transform(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    # Bruno: Claro, no puedo centrarla/alinearla porque
    # eso es lo que quiero. Así que CREAMOS un
    # transformador que, por ejemplo, "aplaste" la glx en
    # el plano z=0 (!)
    center = gchop.preproc.pcenter.Centralizer()
    gal = center.transform(gal)

    # Bruno: El test consiste en que un transf (no importa
    # cuál) devuelva una galaxia con la misma estructura que
    # la que se come (misma cant de parts, length, etc...).
    class Transformer(gchop.preproc._base.GalaxyTransformerABC):
        def transform(self, galaxy):
            df = galaxy.to_dataframe(attributes=["ptypev", "z"])
            df.loc[:, "z"] = np.zeros(len(df))
            # Bruno: Copio...
            stars = df[
                df.ptypev == gchop.core.data.ParticleSetType.STARS.value
            ]
            dark_matter = df[
                df.ptypev == gchop.core.data.ParticleSetType.DARK_MATTER.value
            ]
            gas = df[df.ptypev == gchop.core.data.ParticleSetType.GAS.value]
            new = galaxy.disassemble()
            new.update(
                z_s=stars.z.to_numpy(),
                z_dm=dark_matter.z.to_numpy(),
                z_g=gas.z.to_numpy(),
            )
            return gchop.core.data.mkgalaxy(**new)

        def checker(self, galaxy):
            # Tiene que revisar si todas las partículas tienen z=0
            gal_transf = self.transform(galaxy)
            df = gal_transf.to_dataframe(attributes=["ptypev", "z"])

            return np.allclose(df.loc[:, "z"], 0, rtol=1e-05, atol=1e-08)

    transformer = Transformer()
    gal_transf = transformer.transform(gal)
    is_transf = transformer.checker(gal_transf)

    assert len(gal_transf.stars) == len(gal.stars)
    assert len(gal_transf.dark_matter) == len(gal.dark_matter)
    assert len(gal_transf.gas) == len(gal.gas)
    # Bruno: Ojo con esto. igual está bien que el "checker" debe dar True
    # para cuando al galaxia sea tranformada...
    assert is_transf
