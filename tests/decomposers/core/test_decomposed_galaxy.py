# This file is part of
# the galaxy-chop project (https://github.com/vcristiani/galaxy-chop)
# Copyright (c) Cristiani, et al. 2021, 2022, 2023, 2026
# License: MIT
# Full Text: https://github.com/vcristiani/galaxy-chop/blob/master/LICENSE.txt

import galaxychop as gchop
from galaxychop.core.galaxy import ParticleSetType
from galaxychop.decomposers.core.decomposed_galaxy import (
    DecomposedGalaxy,
    DecomposedParticleSet,
)

import numpy as np

import pytest


# =============================================================================
# DECOMPOSEDGALAXY
# =============================================================================


@pytest.mark.model
def test_Decomposedgalaxy(read_hdf5_galaxy):
    gal = read_hdf5_galaxy("gal394242.h5")
    gal = gchop.preproc.salign.star_align(gchop.preproc.pcenter.center(gal))

    class Decomposer(gchop.decomposers.GalaxyDecomposerABC):
        def get_attributes(self):
            return ["eps"]

        def split(self, X, y, attributes):
            return np.full(len(X), 100), None

    decomposer = Decomposer()

    gal_decomp = decomposer.decompose(gal)

    assert len(gal_decomp) == len(gal)
    assert isinstance(gal_decomp, gchop.decomposers.DecomposedGalaxy)


@pytest.mark.model
def test_DecomposedGalaxy_post_init_validations():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

    softening_value = 0.1

    stars = DecomposedParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3, 4]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=softening_value,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    dark_matter = DecomposedParticleSet(
        ptype=ParticleSetType.DARK_MATTER,
        m=stars.m,
        x=stars.x,
        y=stars.y,
        z=stars.z,
        vx=stars.vx,
        vy=stars.vy,
        vz=stars.vz,
        potential=stars.potential,
        softening=softening_value,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    gas = DecomposedParticleSet(
        ptype=ParticleSetType.GAS,
        m=stars.m,
        x=stars.x,
        y=stars.y,
        z=stars.z,
        vx=stars.vx,
        vy=stars.vy,
        vz=stars.vz,
        potential=stars.potential,
        softening=softening_value,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    # Método vacío
    with pytest.raises(ValueError, match="method cannot be empty"):
        DecomposedGalaxy(
            method="",
            component_name_mapping={0: "disk", 1: "halo"},
            stars=stars,
            dark_matter=dark_matter,
            gas=gas,
        )


@pytest.mark.model
def test_DecomposedGalaxy_has_probabilities():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

    softening_value = 0.1

    stars = DecomposedParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3, 4]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=softening_value,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    dark_matter = DecomposedParticleSet(
        ptype=ParticleSetType.DARK_MATTER,  # Tipo correcto
        m=stars.m,
        x=stars.x,
        y=stars.y,
        z=stars.z,
        vx=stars.vx,
        vy=stars.vy,
        vz=stars.vz,
        potential=stars.potential,
        softening=softening_value,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    gas = DecomposedParticleSet(
        ptype=ParticleSetType.GAS,  # Tipo correcto
        m=stars.m,
        x=stars.x,
        y=stars.y,
        z=stars.z,
        vx=stars.vx,
        vy=stars.vy,
        vz=stars.vz,
        potential=stars.potential,
        softening=softening_value,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    galaxy = DecomposedGalaxy(
        method="clustering",
        component_name_mapping={0: "disk", 1: "halo"},
        stars=stars,
        dark_matter=dark_matter,
        gas=gas,
    )

    assert galaxy.has_probabilities is True


@pytest.mark.model
def test_DecomposedGalaxy_unique_properties():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

    softening_value = 0.1

    stars = DecomposedParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3, 4]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=softening_value,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    dark_matter = DecomposedParticleSet(
        ptype=ParticleSetType.DARK_MATTER,
        m=stars.m,
        x=stars.x,
        y=stars.y,
        z=stars.z,
        vx=stars.vx,
        vy=stars.vy,
        vz=stars.vz,
        potential=stars.potential,
        softening=softening_value,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    gas = DecomposedParticleSet(
        ptype=ParticleSetType.GAS,
        m=stars.m,
        x=stars.x,
        y=stars.y,
        z=stars.z,
        vx=stars.vx,
        vy=stars.vy,
        vz=stars.vz,
        potential=stars.potential,
        softening=softening_value,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    galaxy = DecomposedGalaxy(
        method="clustering",
        component_name_mapping={0: "disk", 1: "halo"},
        stars=stars,
        dark_matter=dark_matter,
        gas=gas,
    )

    assert galaxy.unique_components == {0, 1}
    assert galaxy.unique_components_labels == {"disk", "halo"}


@pytest.mark.model
def test_DecomposedGalaxy_copy():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

    softening_value = 0.1

    stars = DecomposedParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3, 4]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=softening_value,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    dark_matter = DecomposedParticleSet(
        ptype=ParticleSetType.DARK_MATTER,
        m=stars.m,
        x=stars.x,
        y=stars.y,
        z=stars.z,
        vx=stars.vx,
        vy=stars.vy,
        vz=stars.vz,
        potential=stars.potential,
        softening=softening_value,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    gas = DecomposedParticleSet(
        ptype=ParticleSetType.GAS,
        m=stars.m,
        x=stars.x,
        y=stars.y,
        z=stars.z,
        vx=stars.vx,
        vy=stars.vy,
        vz=stars.vz,
        potential=stars.potential,
        softening=softening_value,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    galaxy = DecomposedGalaxy(
        method="clustering",
        component_name_mapping={0: "disk", 1: "halo"},
        stars=stars,
        dark_matter=dark_matter,
        gas=gas,
    )

    galaxy_copy = galaxy.copy()

    assert galaxy_copy.method == galaxy.method
    assert galaxy_copy.component_name_mapping == galaxy.component_name_mapping
    assert galaxy_copy.stars is not galaxy.stars
    assert galaxy_copy.dark_matter is not galaxy.dark_matter
    assert galaxy_copy.gas is not galaxy.gas


@pytest.mark.model
def test_DecomposedGalaxy_empty_method():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

    stars = DecomposedParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3, 4]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=0.1,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    dark_matter = DecomposedParticleSet(
        ptype=ParticleSetType.DARK_MATTER,
        m=stars.m,
        x=stars.x,
        y=stars.y,
        z=stars.z,
        vx=stars.vx,
        vy=stars.vy,
        vz=stars.vz,
        potential=stars.potential,
        softening=0.1,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    gas = DecomposedParticleSet(
        ptype=ParticleSetType.GAS,
        m=stars.m,
        x=stars.x,
        y=stars.y,
        z=stars.z,
        vx=stars.vx,
        vy=stars.vy,
        vz=stars.vz,
        potential=stars.potential,
        softening=0.1,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    with pytest.raises(ValueError, match="method cannot be empty"):
        DecomposedGalaxy(
            method="",
            component_name_mapping={0: "disk", 1: "halo"},
            stars=stars,
            dark_matter=dark_matter,
            gas=gas,
        )


def make_galaxy(has_probs=True):
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = None
    if has_probs:
        probabilities = np.array(
            [
                [0.8, 0.2],
                [0.1, 0.9],
                [0.7, 0.3],
                [0.2, 0.8],
            ]
        )

    softening_value = 0.1

    # Datos toy
    m = np.array([1, 2, 3, 4])
    x = np.array([0, 1, 2, 3])
    y = np.array([1, 2, 3, 4])
    z = np.array([2, 3, 4, 5])
    vx = np.array([3, 4, 5, 6])
    vy = np.array([4, 5, 6, 7])
    vz = np.array([5, 6, 7, 8])
    potential = np.array([6, 7, 8, 9])

    stars = DecomposedParticleSet(
        ptype=ParticleSetType.STARS,
        m=m,
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        potential=potential,
        softening=softening_value,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=has_probs,
    )

    dark_matter = DecomposedParticleSet(
        ptype=ParticleSetType.DARK_MATTER,
        m=m,
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        potential=potential,
        softening=softening_value,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=has_probs,
    )

    gas = DecomposedParticleSet(
        ptype=ParticleSetType.GAS,
        m=m,
        x=x,
        y=y,
        z=z,
        vx=vx,
        vy=vy,
        vz=vz,
        potential=potential,
        softening=softening_value,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=has_probs,
    )

    return DecomposedGalaxy(
        method="clustering",
        component_name_mapping={0: "disk", 1: "halo"},
        stars=stars,
        dark_matter=dark_matter,
        gas=gas,
    )


def test_valid_galaxy_creation():
    galaxy = make_galaxy()
    assert galaxy.method == "clustering"
    assert galaxy.has_probabilities is True
    assert galaxy.unique_components_labels == {"disk", "halo"}


def test_empty_method_raises():
    with pytest.raises(ValueError, match="method cannot be empty"):
        DecomposedGalaxy(
            method="",
            component_name_mapping={0: "disk"},
            stars=make_galaxy().stars,
            dark_matter=make_galaxy().dark_matter,
            gas=make_galaxy().gas,
        )


def test_invalid_particle_set_type():
    with pytest.raises(TypeError, match=r"stars must be of type"):
        DecomposedGalaxy(
            method="clustering",
            component_name_mapping={0: "disk"},
            stars=make_galaxy().dark_matter,
            dark_matter=make_galaxy().dark_matter,
            gas=make_galaxy().gas,
        )


def test_inconsistent_probabilities():
    stars = make_galaxy(has_probs=True).stars
    dark_matter = make_galaxy(has_probs=True).dark_matter

    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])

    gas = DecomposedParticleSet(
        ptype=ParticleSetType.GAS,
        m=stars.m,
        x=stars.x,
        y=stars.y,
        z=stars.z,
        vx=stars.vx,
        vy=stars.vy,
        vz=stars.vz,
        potential=stars.potential,
        softening=0.1,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    try:
        DecomposedGalaxy(
            method="clustering",
            component_name_mapping={0: "disk", 1: "halo"},
            stars=stars,
            dark_matter=dark_matter,
            gas=gas,
        )
    except Exception as e:
        print(type(e), e)


def test_repr_contains_expected_fields():
    galaxy = make_galaxy()
    text = repr(galaxy)
    assert "DecomposedGalaxy" in text
    assert "method='clustering'" in text
    assert "probabilities=True" in text
    assert "components" in text


def test_copy_creates_independent_instance():
    galaxy = make_galaxy()
    galaxy_copy = galaxy.copy()
    assert galaxy is not galaxy_copy
    assert galaxy.stars is not galaxy_copy.stars
    assert galaxy.component_name_mapping == galaxy_copy.component_name_mapping


def test_unique_components_and_labels():
    galaxy = make_galaxy()
    comps = galaxy.unique_components
    labels = galaxy.unique_components_labels
    assert isinstance(comps, set)
    assert isinstance(labels, set)
    assert "disk" in labels
    assert "halo" in labels


@pytest.mark.model
def test_DecomposedGalaxy_repr():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

    stars = DecomposedParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3, 4]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=0.1,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    dark_matter = DecomposedParticleSet(
        ptype=ParticleSetType.DARK_MATTER,
        m=stars.m,
        x=stars.x,
        y=stars.y,
        z=stars.z,
        vx=stars.vx,
        vy=stars.vy,
        vz=stars.vz,
        potential=stars.potential,
        softening=0.1,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    gas = DecomposedParticleSet(
        ptype=ParticleSetType.GAS,
        m=stars.m,
        x=stars.x,
        y=stars.y,
        z=stars.z,
        vx=stars.vx,
        vy=stars.vy,
        vz=stars.vz,
        potential=stars.potential,
        softening=0.1,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    galaxy = DecomposedGalaxy(
        method="clustering",
        component_name_mapping={0: "disk", 1: "halo"},
        stars=stars,
        dark_matter=dark_matter,
        gas=gas,
    )

    expected_repr = (
        "<DecomposedGalaxy method='clustering', dark_matter=4, gas=4, "
        "probabilities=True, components=['disk', 'halo']>"
    )
    assert repr(galaxy) == expected_repr


@pytest.mark.model
def test_DecomposedGalaxy_total_mass():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.full((4, 2), np.nan)

    stars = DecomposedParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1.0, 2.0, 3.0, 4.0]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=0.1,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=False,
    )
    dark_matter = DecomposedParticleSet(
        ptype=ParticleSetType.DARK_MATTER,
        m=np.array([5.0, 5.0]),
        x=np.array([0, 1]),
        y=np.array([1, 2]),
        z=np.array([2, 3]),
        vx=np.array([3, 4]),
        vy=np.array([4, 5]),
        vz=np.array([5, 6]),
        potential=np.array([6, 7]),
        softening=0.1,
        components=np.array([0, 0]),
        labels=np.array(["dark_matter", "dark_matter"]),
        probabilities=np.full((2, 1), np.nan),
        has_probabilities=False,
    )
    gas = DecomposedParticleSet(
        ptype=ParticleSetType.GAS,
        m=np.array([2.0]),
        x=np.array([0]),
        y=np.array([1]),
        z=np.array([2]),
        vx=np.array([3]),
        vy=np.array([4]),
        vz=np.array([5]),
        potential=np.array([6]),
        softening=0.1,
        components=np.array([0]),
        labels=np.array(["gas"]),
        probabilities=np.full((1, 1), np.nan),
        has_probabilities=False,
    )

    galaxy = DecomposedGalaxy(
        method="clustering",
        component_name_mapping={0: "disk", 1: "halo"},
        stars=stars,
        dark_matter=dark_matter,
        gas=gas,
    )

    result = galaxy.total_mass()

    assert result.index.names == ["ptype", "components"]
    assert list(result.columns) == ["labels", "particles", "m", "mf"]

    stars_rows = result.loc["stars"]
    assert set(stars_rows["particles"]) == {2, 2}
    assert stars_rows["m"].sum() == pytest.approx(10.0)

    dm_rows = result.loc["dark_matter"]
    assert dm_rows["particles"].sum() == 2
    assert dm_rows["m"].sum() == pytest.approx(10.0)

    gas_rows = result.loc["gas"]
    assert gas_rows["particles"].sum() == 1
    assert gas_rows["m"].sum() == pytest.approx(2.0)


@pytest.mark.model
def test_DecomposedGalaxy_repr_html():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

    stars = DecomposedParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1.0, 2.0, 3.0, 4.0]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=0.1,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )
    dark_matter = DecomposedParticleSet(
        ptype=ParticleSetType.DARK_MATTER,
        m=stars.m,
        x=stars.x,
        y=stars.y,
        z=stars.z,
        vx=stars.vx,
        vy=stars.vy,
        vz=stars.vz,
        potential=stars.potential,
        softening=0.1,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )
    gas = DecomposedParticleSet(
        ptype=ParticleSetType.GAS,
        m=stars.m,
        x=stars.x,
        y=stars.y,
        z=stars.z,
        vx=stars.vx,
        vy=stars.vy,
        vz=stars.vz,
        potential=stars.potential,
        softening=0.1,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    galaxy = DecomposedGalaxy(
        method="clustering",
        component_name_mapping={0: "disk", 1: "halo"},
        stars=stars,
        dark_matter=dark_matter,
        gas=gas,
    )

    html = galaxy._repr_html_()

    assert isinstance(html, str)
    assert "🌌🧩" in html
    assert "DecomposedGalaxy" in html
    assert "method</b>='clustering'" in html
    assert "Components breakdown" in html
    assert "Probabilities:</b> Yes" in html


# =============================================================================
# COMPONENTPARTICLESET
# =============================================================================


@pytest.mark.model
def test_DecomposedParticleSet():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

    pset = gchop.core.ParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3, 4]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=0.1,
    )

    cps = DecomposedParticleSet.from_pset(
        pset, components, labels, probabilities, has_probabilities=True
    )

    assert cps.has_probabilities is True
    assert cps.probabilities_n == 2
    assert (
        cps.get_value_makers()["components"]().tolist() == components.tolist()
    )
    assert cps.get_value_makers()["labels"]().tolist() == labels.tolist()
    assert (
        cps.get_value_makers()["prob_0"]().tolist()
        == probabilities[:, 0].tolist()
    )
    cps_copy = cps.copy()
    assert np.array_equal(cps_copy.components, cps.components)
    assert np.array_equal(cps_copy.labels, cps.labels)
    assert np.array_equal(cps_copy.probabilities, cps.probabilities)


@pytest.mark.model
def test_DecomposedParticleSet_validations():
    components = np.array([0, 1, 0])
    labels = np.array(["disk", "halo", "disk"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])

    pset = gchop.core.ParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3]),
        x=np.array([0, 1, 2]),
        y=np.array([1, 2, 3]),
        z=np.array([2, 3, 4]),
        vx=np.array([3, 4, 5]),
        vy=np.array([4, 5, 6]),
        vz=np.array([5, 6, 7]),
        potential=np.array([6, 7, 8]),
        softening=0.1,
    )

    with pytest.raises(ValueError, match="probabilities must be in the range"):
        probabilities_invalid = np.array([[1.2, -0.2], [0.1, 0.9], [0.7, 0.3]])
        DecomposedParticleSet.from_pset(
            pset,
            components,
            labels,
            probabilities_invalid,
            has_probabilities=True,
        )

    with pytest.raises(
        ValueError, match="galaxy length.*must match components length"
    ):
        components_invalid = np.array([0, 1])
        DecomposedParticleSet.from_pset(
            pset,
            components_invalid,
            labels,
            probabilities,
            has_probabilities=True,
        )


@pytest.mark.model
def test_DecomposedParticleSet_from_pset():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

    pset = gchop.core.ParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3, 4]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=0.1,
    )

    cps = DecomposedParticleSet.from_pset(
        pset, components, labels, probabilities, has_probabilities=True
    )

    assert np.array_equal(cps.components, components)
    assert np.array_equal(cps.labels, labels)
    assert np.array_equal(cps.probabilities, probabilities)


@pytest.mark.model
def test_DecomposedParticleSet_post_init_validations():
    components = np.array([0, 1, 0])
    labels = np.array(["disk", "halo", "disk"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])

    pset = gchop.core.ParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3]),
        x=np.array([0, 1, 2]),
        y=np.array([1, 2, 3]),
        z=np.array([2, 3, 4]),
        vx=np.array([3, 4, 5]),
        vy=np.array([4, 5, 6]),
        vz=np.array([5, 6, 7]),
        potential=np.array([6, 7, 8]),
        softening=0.1,
    )

    with pytest.raises(
        ValueError, match="galaxy length.*must match components length"
    ):
        components_invalid = np.array([0, 1])
        DecomposedParticleSet.from_pset(
            pset,
            components_invalid,
            labels,
            probabilities,
            has_probabilities=True,
        )

    with pytest.raises(ValueError, match="probabilities must be in the range"):
        probabilities_invalid = np.array([[1.2, -0.2], [0.1, 0.9], [0.7, 0.3]])
        DecomposedParticleSet.from_pset(
            pset,
            components,
            labels,
            probabilities_invalid,
            has_probabilities=True,
        )


@pytest.mark.model
def test_DecomposedParticleSet_has_probabilities():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities_empty = np.empty((4, 0))

    pset = gchop.core.ParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3, 4]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=0.1,
    )

    cps = DecomposedParticleSet.from_pset(
        pset, components, labels, probabilities_empty, has_probabilities=True
    )

    assert cps.has_probabilities is True


@pytest.mark.model
def test_DecomposedParticleSet_get_value_makers():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

    pset = gchop.core.ParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3, 4]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=0.1,
    )

    cps = DecomposedParticleSet.from_pset(
        pset, components, labels, probabilities, has_probabilities=True
    )

    value_makers = cps.get_value_makers()

    assert value_makers["components"]().tolist() == components.tolist()
    assert value_makers["labels"]().tolist() == labels.tolist()

    for i in range(cps.probabilities_n):
        assert (
            value_makers[f"prob_{i}"]().tolist()
            == probabilities[:, i].tolist()
        )


@pytest.mark.model
def test_DecomposedParticleSet_copy():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

    pset = gchop.core.ParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3, 4]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=0.1,
    )

    cps = DecomposedParticleSet.from_pset(
        pset, components, labels, probabilities, has_probabilities=True
    )

    cps_copy = cps.copy()

    assert np.array_equal(cps_copy.components, cps.components)
    assert np.array_equal(cps_copy.labels, cps.labels)
    assert np.array_equal(cps_copy.probabilities, cps.probabilities)

    new_components = np.array([99, 1, 0, 1])
    new_labels = np.array(["modified", "halo", "disk", "halo"])
    new_probabilities = np.array(
        [[0.99, 0.01], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]]
    )

    cps_copy = DecomposedParticleSet.from_pset(
        pset,
        new_components,
        new_labels,
        new_probabilities,
        has_probabilities=True,
    )

    assert not np.array_equal(cps_copy.components, cps.components)
    assert not np.array_equal(cps_copy.labels, cps.labels)
    assert not np.array_equal(cps_copy.probabilities, cps.probabilities)


@pytest.mark.model
def test_DecomposedParticleSet_probabilities_validation():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities_invalid = np.array(
        [[1.2, -0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]]
    )

    pset = gchop.core.ParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3, 4]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=0.1,
    )

    with pytest.raises(ValueError, match="probabilities must be in the range"):
        DecomposedParticleSet.from_pset(
            pset,
            components,
            labels,
            probabilities_invalid,
            has_probabilities=True,
        )


@pytest.mark.model
def test_DecomposedParticleSet_read_only_arrays():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

    pset = gchop.core.ParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3, 4]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=0.1,
    )

    cps = DecomposedParticleSet.from_pset(
        pset, components, labels, probabilities, has_probabilities=True
    )

    with pytest.raises(
        ValueError, match="assignment destination is read-only"
    ):
        cps.components[0] = 99

    with pytest.raises(
        ValueError, match="assignment destination is read-only"
    ):
        cps.labels[0] = "modified"

    with pytest.raises(
        ValueError, match="assignment destination is read-only"
    ):
        cps.probabilities[0, 0] = 0.99


@pytest.mark.model
def test_DecomposedParticleSet_get_value_makers_probabilities():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

    pset = gchop.core.ParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3, 4]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=0.1,
    )

    cps = DecomposedParticleSet.from_pset(
        pset, components, labels, probabilities, has_probabilities=True
    )

    value_makers = cps.get_value_makers()

    for i in range(cps.probabilities_n):
        assert np.array_equal(value_makers[f"prob_{i}"](), probabilities[:, i])


@pytest.mark.model
def test_DecomposedParticleSet_copy_independence():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

    pset = gchop.core.ParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1, 2, 3, 4]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=0.1,
    )

    cps = DecomposedParticleSet.from_pset(
        pset, components, labels, probabilities, has_probabilities=True
    )

    cps_copy = cps.copy()

    new_components = cps_copy.components.copy()
    new_components[0] = 99

    new_labels = cps_copy.labels.copy()
    new_labels[0] = "modified"

    new_probabilities = cps_copy.probabilities.copy()
    new_probabilities[0, 0] = 0.99

    assert not np.array_equal(new_components, cps.components)
    assert not np.array_equal(new_labels, cps.labels)
    assert not np.array_equal(new_probabilities, cps.probabilities)


@pytest.mark.model
def test_DecomposedParticleSet_total_mass():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.full((4, 2), np.nan)

    stars = DecomposedParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1.0, 2.0, 3.0, 4.0]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=0.1,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=False,
    )

    result = stars.total_mass()

    assert list(result.columns) == ["labels", "particles", "m", "mf"]

    disk = result[result["labels"] == "disk"].iloc[0]
    halo = result[result["labels"] == "halo"].iloc[0]

    assert disk["particles"] == 2
    assert disk["m"] == pytest.approx(4.0)  # particles 0 and 2: m=1+3
    assert disk["mf"] == pytest.approx(0.4)

    assert halo["particles"] == 2
    assert halo["m"] == pytest.approx(6.0)  # particles 1 and 3: m=2+4
    assert halo["mf"] == pytest.approx(0.6)


@pytest.mark.model
def test_DecomposedParticleSet_total_mass_probabilistic():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])

    stars = DecomposedParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1.0, 2.0, 3.0, 4.0]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=0.1,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=True,
    )

    result = stars.total_mass()

    assert list(result.columns) == [
        "labels",
        "particles",
        "m",
        "mf",
        "pm",
        "pmf",
    ]

    disk = result[result["labels"] == "disk"].iloc[0]
    # particles 0 and 2 (m=1, m=3) weighted by their prob_0
    expected_pm = 1.0 * 0.8 + 3.0 * 0.7
    assert disk["pm"] == pytest.approx(expected_pm)
    assert disk["pmf"] == pytest.approx(expected_pm / 10.0)


@pytest.mark.model
def test_DecomposedParticleSet_repr_html():
    components = np.array([0, 1, 0, 1])
    labels = np.array(["disk", "halo", "disk", "halo"])
    probabilities = np.full((4, 2), np.nan)

    stars = DecomposedParticleSet(
        ptype=ParticleSetType.STARS,
        m=np.array([1.0, 2.0, 3.0, 4.0]),
        x=np.array([0, 1, 2, 3]),
        y=np.array([1, 2, 3, 4]),
        z=np.array([2, 3, 4, 5]),
        vx=np.array([3, 4, 5, 6]),
        vy=np.array([4, 5, 6, 7]),
        vz=np.array([5, 6, 7, 8]),
        potential=np.array([6, 7, 8, 9]),
        softening=0.1,
        components=components,
        labels=labels,
        probabilities=probabilities,
        has_probabilities=False,
    )

    html = stars._repr_html_()

    assert isinstance(html, str)
    assert "⭐🧩" in html
    assert "DecomposedParticleSet" in html
    assert "components</b>=2" in html
    assert "probabilities</b>=False" in html
