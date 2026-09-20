import warnings

import ase
import numpy as np
import pytest

import abtem
from abtem.core.axes import OrdinalAxis
from abtem.ebsd import (
    EBSD,
    BackscatterDetector,
    SparseProjectionWarning,
    SphericalPattern,
    SquareLambertProjection,
    StereographicProjection,
    bin_directions,
    bulk_block,
    estimate_repetitions,
    fibonacci_hemisphere,
    pixel_centers,
    rotated_slab,
    validate_projection,
    write_emsoft_master_pattern,
    zone_axis_rotation,
)
from abtem.ebsd.reciprocity import _validate_depth_weight
from abtem.ebsd.reference import (
    EBSDReferencePattern,
    patch_half_angle,
    recommended_sampling,
)

PROJECTIONS = [StereographicProjection(), SquareLambertProjection()]
PROJECTION_IDS = [p.name for p in PROJECTIONS]


def random_hemisphere_directions(n, seed=0):
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions[:, 2] = np.abs(directions[:, 2])
    return directions


class TestProjections:
    @pytest.mark.parametrize("projection", PROJECTIONS, ids=PROJECTION_IDS)
    def test_round_trip(self, projection):
        directions = random_hemisphere_directions(500)
        assert np.allclose(
            projection.unproject(projection.project(directions)), directions
        )

    @pytest.mark.parametrize("projection", PROJECTIONS, ids=PROJECTION_IDS)
    def test_unproject_gives_unit_northern_vectors(self, projection):
        grid = projection.grid(64)
        assert np.allclose(np.linalg.norm(grid, axis=1), 1.0)
        assert np.all(grid[:, 2] >= -1e-12)

    @pytest.mark.parametrize("projection", PROJECTIONS, ids=PROJECTION_IDS)
    def test_projects_into_the_unit_square(self, projection):
        xy = projection.project(random_hemisphere_directions(500))
        assert np.all(np.abs(xy) <= 1.0 + 1e-12)

    @pytest.mark.parametrize("projection", PROJECTIONS, ids=PROJECTION_IDS)
    def test_pole_maps_to_the_origin(self, projection):
        xy = projection.project(np.array([[0.0, 0.0, 1.0]]))
        assert np.allclose(xy, 0.0)

    def test_stereographic_equator_maps_to_the_unit_circle(self):
        theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        equator = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
        xy = StereographicProjection().project(equator)
        assert np.allclose(np.linalg.norm(xy, axis=1), 1.0)

    def test_lambert_is_equal_area(self):
        # An equal-area projection sends a uniform distribution on the
        # hemisphere to a uniform distribution on the square, so every pixel of
        # a coarse binning should receive the same count to within noise.
        directions = random_hemisphere_directions(400_000, seed=3)
        counts = bin_directions(
            directions, np.ones(len(directions)), gpts=1, projection="lambert"
        )
        assert counts.shape == (1, 1)

        xy = SquareLambertProjection().project(directions)
        counts, _, _ = np.histogram2d(
            xy[:, 0], xy[:, 1], bins=8, range=[[-1, 1], [-1, 1]]
        )
        assert counts.std() / counts.mean() < 0.02

    def test_stereographic_grid_fills_the_inscribed_disk_only(self):
        grid = StereographicProjection().grid(64)
        # pi / 4 of the square's points lie in the inscribed disk
        assert 0.75 < len(grid) / 64**2 < 0.81

    def test_lambert_grid_fills_the_whole_square(self):
        assert len(SquareLambertProjection().grid(64)) == 64**2

    def test_validate_projection(self):
        assert isinstance(validate_projection("lambert"), SquareLambertProjection)
        instance = StereographicProjection()
        assert validate_projection(instance) is instance
        with pytest.raises(ValueError, match="projection must be one of"):
            validate_projection("gnomonic")


class TestBinDirections:
    def test_averages_within_a_pixel(self):
        directions = np.array([[0.0, 0.0, 1.0], [1e-6, 1e-6, 1.0]])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        image = bin_directions(directions, np.array([1.0, 3.0]), gpts=2)
        assert image.sum() == pytest.approx(2.0)

    def test_empty_pixels_are_zero(self):
        directions = np.array([[0.0, 0.0, 1.0]])
        image = bin_directions(directions, np.array([5.0]), gpts=8)
        assert (image != 0).sum() == 1
        assert image.max() == pytest.approx(5.0)

    def test_southern_directions_are_dropped(self):
        directions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        image = bin_directions(directions, np.array([2.0, 100.0]), gpts=4)
        assert image.max() == pytest.approx(2.0)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="values has length"):
            bin_directions(np.zeros((3, 3)), np.zeros(2), gpts=4)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
            bin_directions(np.zeros((3, 2)), np.zeros(3), gpts=4)


class TestFibonacciHemisphere:
    def test_unit_vectors_on_the_northern_hemisphere(self):
        directions = fibonacci_hemisphere(500)
        assert directions.shape == (500, 3)
        assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
        assert np.all(directions[:, 2] > 0.0)

    def test_approximately_equal_area(self):
        # z is uniformly spaced, and equal spacing in z is equal solid angle.
        z = fibonacci_hemisphere(1000)[:, 2]
        assert np.allclose(np.diff(z), np.diff(z)[0])

    def test_rejects_non_positive_n(self):
        with pytest.raises(ValueError, match="n must be at least 1"):
            fibonacci_hemisphere(0)


class TestZoneAxisRotation:
    @pytest.mark.parametrize(
        "zone_axis",
        [(0, 0, 1), (1, 1, 1), (1, 0, 0), (2, 2, 3), (-1, 2, -3)],
    )
    def test_maps_the_zone_axis_onto_z(self, zone_axis):
        rotation = zone_axis_rotation(np.array(zone_axis, dtype=float))
        unit = np.array(zone_axis, dtype=float)
        unit /= np.linalg.norm(unit)
        assert np.allclose(rotation @ unit, [0.0, 0.0, 1.0])

    @pytest.mark.parametrize(
        "zone_axis", [(0, 0, 1), (1, 1, 1), (1, 0, 0), (2, 2, 3), (-1, 2, -3)]
    )
    def test_is_a_proper_rotation(self, zone_axis):
        rotation = zone_axis_rotation(np.array(zone_axis, dtype=float))
        assert np.allclose(rotation @ rotation.T, np.eye(3))
        assert np.linalg.det(rotation) == pytest.approx(1.0)

    @pytest.mark.parametrize("zone_axis", [(1, 1, 1), (2, 2, 3), (-1, 2, -3)])
    def test_matches_ase(self, zone_axis):
        # rotated_slab relies on this agreement: it rotates the atoms with ASE
        # but returns this matrix for transforming directions.
        unit = np.array(zone_axis, dtype=float)
        unit /= np.linalg.norm(unit)

        atoms = ase.Atoms("H", positions=[(0, 0, 0)], cell=np.eye(3))
        atoms.rotate(unit, "z", rotate_cell=True)
        ase_rotation = np.linalg.inv(np.eye(3)) @ atoms.cell[:]

        assert np.allclose(ase_rotation, zone_axis_rotation(unit).T)

    def test_rejects_zero_and_wrong_shape(self):
        with pytest.raises(ValueError, match="must be nonzero"):
            zone_axis_rotation(np.zeros(3))
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            zone_axis_rotation(np.zeros(2))


class TestRotatedSlab:
    @pytest.fixture
    def silicon(self):
        return ase.build.bulk("Si", "diamond", a=5.431)

    def test_cell_matches_the_request(self, silicon):
        cell = (10.0, 12.0, 20.0)
        slab, _ = rotated_slab(silicon, np.array([1.0, 1.0, 1.0]), cell)
        assert np.allclose(slab.cell.lengths(), cell)

    def test_atoms_lie_inside_the_cell(self, silicon):
        cell = (10.0, 10.0, 20.0)
        slab, _ = rotated_slab(silicon, np.array([1.0, 2.0, 3.0]), cell)
        assert len(slab) > 0
        assert np.all(slab.positions >= 0.0)
        assert np.all(slab.positions <= np.array(cell))

    def test_density_is_preserved(self, silicon):
        cell = (12.0, 12.0, 24.0)
        slab, _ = rotated_slab(silicon, np.array([1.0, 1.0, 1.0]), cell)
        expected = len(silicon) / silicon.get_volume() * float(np.prod(cell))
        assert len(slab) == pytest.approx(expected, rel=0.05)

    def test_rotation_takes_the_zone_axis_to_the_beam_direction(self, silicon):
        zone_axis = np.array([2.0, 2.0, 3.0])
        _, rotation = rotated_slab(silicon, zone_axis, (10.0, 10.0, 10.0))
        unit = zone_axis / np.linalg.norm(zone_axis)
        assert np.allclose(rotation @ unit, [0.0, 0.0, 1.0])

    def test_estimate_repetitions_covers_the_diagonal(self, silicon):
        cell = (10.0, 10.0, 40.0)
        repetitions = estimate_repetitions(silicon, cell)
        spanned = np.array(repetitions) * silicon.cell.lengths()
        assert np.all(spanned >= np.linalg.norm(cell))

    def test_rejects_wrong_cell_shape(self, silicon):
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            rotated_slab(silicon, np.array([0.0, 0.0, 1.0]), (10.0, 10.0))

    def test_a_reused_block_gives_the_same_slab(self, silicon):
        # Building the block is the expensive part, so a reference pattern
        # builds it once and cuts every zone axis out of it. That must give
        # exactly what repeating per call does.
        cell = (10.0, 10.0, 20.0)
        zone_axis = np.array([1.0, 2.0, 3.0])

        block = bulk_block(silicon, cell, (12, 12, 12))
        reused, _ = rotated_slab(block, zone_axis, cell, repetitions=(1, 1, 1))
        rebuilt, _ = rotated_slab(silicon, zone_axis, cell, repetitions=(12, 12, 12))

        assert np.array_equal(reused.positions, rebuilt.positions)
        assert np.array_equal(reused.numbers, rebuilt.numbers)

    def test_a_reused_block_is_not_mutated(self, silicon):
        # rotated_slab rotates and recentres in place, so it has to copy.
        cell = (10.0, 10.0, 20.0)
        block = bulk_block(silicon, cell, (12, 12, 12))
        before = block.positions.copy()

        rotated_slab(block, np.array([1.0, 1.0, 1.0]), cell, repetitions=(1, 1, 1))

        assert np.array_equal(block.positions, before)

    def test_bulk_block_defaults_to_the_estimated_repetitions(self, silicon):
        cell = (10.0, 10.0, 20.0)
        block = bulk_block(silicon, cell)
        expected = silicon * estimate_repetitions(silicon, cell)
        assert len(block) == len(expected)


class TestBackscatterDetector:
    def test_grid_directions_are_unit_vectors_into_the_crystal(self):
        detector = BackscatterDetector(max_angle=200, gpts=9)
        assert detector.directions.shape == (81, 3)
        assert np.allclose(np.linalg.norm(detector.directions, axis=1), 1.0)
        assert np.all(detector.directions[:, 2] > 0.0)

    def test_grid_is_centred_on_the_beam_direction(self):
        detector = BackscatterDetector(max_angle=200, gpts=9)
        centre = detector.directions.reshape(9, 9, 3)[9 // 2, 9 // 2]
        assert np.allclose(centre, [0.0, 0.0, 1.0])

    def test_grid_half_width_matches_max_angle(self):
        detector = BackscatterDetector(max_angle=200, gpts=64)
        transverse = np.abs(detector.directions[:, :2]).max()
        assert transverse == pytest.approx(np.sin(0.2), rel=0.05)

    def test_base_shape(self):
        assert BackscatterDetector(max_angle=100, gpts=7).base_shape == (7, 7)
        directions = random_hemisphere_directions(5)
        assert BackscatterDetector(directions=directions).base_shape == (5,)

    def test_explicit_directions_are_normalized(self):
        detector = BackscatterDetector(directions=np.array([[0.0, 0.0, 2.0]]))
        assert np.allclose(detector.directions, [[0.0, 0.0, 1.0]])

    def test_transverse_wave_vectors_scale_with_the_wave_number(self):
        from abtem.core.energy import energy2wavelength

        detector = BackscatterDetector(max_angle=100, gpts=5)
        for energy in (30e3, 200e3):
            expected = detector.directions[:, :2] / energy2wavelength(energy)
            assert np.allclose(detector.transverse_wave_vectors(energy), expected)

    def test_reciprocal_sampling_spans_the_grid(self):
        detector = BackscatterDetector(max_angle=100, gpts=16)
        vectors = detector.transverse_wave_vectors(30e3)
        step = np.diff(np.unique(np.round(vectors[:, 0], 9)))
        assert np.allclose(step, detector.reciprocal_sampling(30e3))

    def test_reciprocal_sampling_rejects_explicit_directions(self):
        detector = BackscatterDetector(directions=random_hemisphere_directions(4))
        with pytest.raises(RuntimeError, match="no regular sampling"):
            detector.reciprocal_sampling(30e3)

    def test_is_grid(self):
        assert BackscatterDetector(max_angle=100, gpts=4).is_grid
        assert not BackscatterDetector(
            directions=random_hemisphere_directions(4)
        ).is_grid

    def test_warns_when_the_small_angle_approximation_fails(self):
        with pytest.warns(UserWarning, match="small-angle approximation"):
            BackscatterDetector(max_angle=700, gpts=5)

    def test_rejects_backward_directions(self):
        with pytest.raises(ValueError, match="positive z component"):
            BackscatterDetector(directions=np.array([[0.0, 0.0, -1.0]]))

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({}, "give either"),
            ({"max_angle": 100}, "must be given together"),
            ({"gpts": 4}, "must be given together"),
            (
                {"max_angle": 100, "gpts": 4, "directions": np.zeros((1, 3))},
                "not both",
            ),
            ({"directions": np.zeros((3, 2))}, r"shape \(N, 3\)"),
        ],
    )
    def test_rejects_bad_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            BackscatterDetector(**kwargs)


class TestSphericalPattern:
    @pytest.fixture
    def pattern(self):
        directions = fibonacci_hemisphere(500)
        return SphericalPattern(
            1.0 + directions[:, 2], directions, metadata={"energy": 30e3}
        )

    def test_project_shape_and_axes(self, pattern):
        # A Fibonacci lattice is not a projection grid, so the binned image has
        # holes; that is what TestSparseProjection covers, not this.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseProjectionWarning)
            images = pattern.project(32)
        assert images.array.shape == (32, 32)
        assert [axis.units for axis in images.base_axes_metadata] == ["", ""]
        assert images.metadata["projection"] == "stereographic"

    def test_project_preserves_the_ensemble(self):
        directions = fibonacci_hemisphere(200)
        array = np.stack([np.ones(200), 2 * np.ones(200)])
        pattern = SphericalPattern(
            array, directions, ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseProjectionWarning)
            images = pattern.project(16)
        assert images.array.shape == (2, 16, 16)
        assert images.array[1].max() == pytest.approx(2.0)

    def test_zarr_round_trip(self, pattern, tmp_path):
        url = str(tmp_path / "pattern.zarr")
        pattern.to_zarr(url)
        restored = SphericalPattern.from_zarr(url)
        assert np.array_equal(restored.array, pattern.array)
        assert np.array_equal(restored.directions, pattern.directions)
        assert restored.metadata == pattern.metadata

    def test_concatenate(self, pattern):
        halves = [
            SphericalPattern(pattern.array[:200], pattern.directions[:200]),
            SphericalPattern(pattern.array[200:], pattern.directions[200:]),
        ]
        joined = SphericalPattern.concatenate(halves)
        assert np.array_equal(joined.array, pattern.array)
        assert np.array_equal(joined.directions, pattern.directions)

    def test_concatenate_rejects_mismatched_ensembles(self, pattern):
        other = SphericalPattern(
            np.stack([pattern.array, pattern.array]),
            pattern.directions,
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
        )
        with pytest.raises(ValueError, match="mismatched ensemble shapes"):
            SphericalPattern.concatenate([pattern, other])

    def test_concatenate_rejects_empty(self):
        with pytest.raises(ValueError, match="empty sequence"):
            SphericalPattern.concatenate([])

    def test_rejects_mismatched_array_and_directions(self):
        with pytest.raises(ValueError, match="last axis of array"):
            SphericalPattern(np.zeros(4), fibonacci_hemisphere(5))

    def test_rejects_wrong_ensemble_axis_count(self):
        with pytest.raises(ValueError, match="ensemble axes"):
            SphericalPattern(np.zeros((2, 5)), fibonacci_hemisphere(5))


def silicon_slab(thickness=8.0, zone_axis=(0.0, 0.0, 1.0)):
    return rotated_slab(
        ase.build.bulk("Si", "diamond", a=5.431),
        np.array(zone_axis),
        (10.0, 10.0, thickness),
    )[0]


def make_ebsd(atoms, detector, sampling=0.15, **kwargs):
    return EBSD(
        abtem.Potential(
            atoms, sampling=sampling, slice_thickness=1.0, projection="finite"
        ),
        probe=abtem.Probe(semiangle_cutoff=10, energy=30e3),
        detector=detector,
        **kwargs,
    )


class TestDepthWeight:
    @pytest.mark.parametrize(
        "weight, expected",
        [
            (None, [0.25] * 4),
            (np.ones(4), [0.25] * 4),
            ([1.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]),
        ],
    )
    def test_normalized_to_sum_to_one(self, weight, expected):
        depths = np.arange(1.0, 5.0)
        assert np.allclose(_validate_depth_weight(weight, depths), expected)

    def test_escape_depth_decays_exponentially(self):
        depths = np.arange(1.0, 5.0)
        weights = _validate_depth_weight(2.0, depths)
        ratios = weights[1:] / weights[:-1]
        assert np.allclose(ratios, np.exp(-0.5))

    def test_callable(self):
        depths = np.arange(1.0, 5.0)
        weights = _validate_depth_weight(lambda z: np.ones_like(z), depths)
        assert np.allclose(weights, 0.25)

    @pytest.mark.parametrize(
        "weight, match",
        [
            (-1.0, "escape depth must be positive"),
            (np.ones(3), "one weight per slice"),
            (np.array([-1.0, 1.0, 1.0, 1.0]), "must not be negative"),
            (np.zeros(4), "must not be all zero"),
        ],
    )
    def test_rejects_bad_weights(self, weight, match):
        with pytest.raises(ValueError, match=match):
            _validate_depth_weight(weight, np.arange(1.0, 5.0))


class TestEBSD:
    def test_vacuum_gives_unit_yield(self):
        # With no specimen the reciprocity plane waves stay plane waves, so the
        # overlap with the beam is the beam's own intensity and the yield is 1
        # by construction of the normalization.
        vacuum = ase.Atoms(cell=(10.0, 10.0, 8.0), pbc=True)
        patterns = make_ebsd(
            vacuum,
            BackscatterDetector(max_angle=50, gpts=5),
            sampling=0.1,
            potential_weighting=False,
        ).scan()
        assert np.allclose(patterns.array, 1.0, atol=5e-3)

    def test_vacuum_with_potential_weighting_generates_nothing(self):
        # Every slice is empty, so there is nothing to scatter off. This must
        # come out as zero rather than 0/0.
        vacuum = ase.Atoms(cell=(10.0, 10.0, 8.0), pbc=True)
        patterns = make_ebsd(
            vacuum,
            BackscatterDetector(max_angle=50, gpts=5),
            sampling=0.1,
            potential_weighting=True,
        ).scan()
        assert np.all(np.isfinite(patterns.array))
        assert np.allclose(patterns.array, 0.0)

    def test_uniform_depth_weight_matches_the_default(self):
        atoms = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        default = make_ebsd(atoms, detector).scan()
        explicit = make_ebsd(
            atoms, detector, depth_weight=np.ones(int(atoms.cell[2, 2]))
        ).scan()
        assert np.allclose(default.array, explicit.array)

    def test_a_long_escape_depth_approaches_uniform_weighting(self):
        atoms = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        uniform = make_ebsd(atoms, detector).scan()
        deep = make_ebsd(atoms, detector, depth_weight=1e6).scan()
        assert np.allclose(uniform.array, deep.array, rtol=1e-4)

    def test_a_short_escape_depth_weights_the_surface(self):
        # A shallow escape depth must give a different answer from a uniform
        # one; otherwise the weighting is not reaching the accumulation.
        atoms = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        uniform = make_ebsd(atoms, detector).scan()
        shallow = make_ebsd(atoms, detector, depth_weight=2.0).scan()
        assert not np.allclose(uniform.array, shallow.array, rtol=1e-3)

    def test_batching_does_not_change_the_result(self):
        atoms = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=6)
        whole = make_ebsd(atoms, detector).scan(max_batch_directions=10_000)
        split = make_ebsd(atoms, detector).scan(max_batch_directions=4)
        assert np.allclose(whole.array, split.array, rtol=1e-5)

    def test_grid_detector_returns_diffraction_patterns(self):
        patterns = make_ebsd(
            silicon_slab(), BackscatterDetector(max_angle=50, gpts=8)
        ).scan()
        assert isinstance(patterns, abtem.DiffractionPatterns)
        assert patterns.array.shape == (8, 8)
        assert patterns.metadata["energy"] == 30e3
        assert "antialias_loss_max" in patterns.metadata

    def test_explicit_directions_reproduce_the_equivalent_grid(self):
        # The two ways of building a detector describe the same directions, so
        # they must give the same intensities -- only the measurement type and
        # its shape differ.
        atoms = silicon_slab()
        grid = BackscatterDetector(max_angle=50, gpts=4)
        explicit = BackscatterDetector(directions=grid.directions)

        from_grid = make_ebsd(atoms, grid).scan()
        from_explicit = make_ebsd(atoms, explicit).scan()

        assert isinstance(from_explicit, SphericalPattern)
        assert from_explicit.array.shape == (16,)
        assert np.allclose(from_explicit.directions, grid.directions)
        assert np.allclose(from_grid.array.ravel(), from_explicit.array, rtol=1e-6)

    def test_scan_adds_a_leading_ensemble_axis(self):
        scan = abtem.CustomScan([[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]])
        patterns = make_ebsd(
            silicon_slab(), BackscatterDetector(max_angle=50, gpts=4)
        ).scan(scan=scan)
        assert patterns.array.shape == (3, 4, 4)
        assert len(patterns.ensemble_axes_metadata) == 1

    def test_each_scan_position_matches_its_own_calculation(self):
        atoms = silicon_slab()
        detector = BackscatterDetector(max_angle=50, gpts=4)
        positions = [[2.0, 2.0], [7.0, 3.0]]

        together = make_ebsd(atoms, detector).scan(
            scan=abtem.CustomScan(positions)
        )
        for i, position in enumerate(positions):
            alone = make_ebsd(atoms, detector).scan(
                scan=abtem.CustomScan([position])
            )
            assert np.allclose(together.array[i], alone.array[0], rtol=1e-5)

    def test_yield_is_of_order_one(self):
        # The normalization is relative to a featureless specimen, so a real
        # one should land near unity rather than at an arbitrary scale.
        patterns = make_ebsd(
            silicon_slab(), BackscatterDetector(max_angle=50, gpts=8)
        ).scan()
        assert 0.5 < patterns.array.mean() < 2.0

    def test_lazy_is_refused(self):
        with pytest.raises(NotImplementedError, match="lazy=False"):
            make_ebsd(
                silicon_slab(), BackscatterDetector(max_angle=50, gpts=4)
            ).scan(lazy=True)

    def test_warns_when_the_antialias_aperture_clips(self):
        # A plane wave launched near the aperture edge scatters straight past
        # it, so collecting wide angles on a coarse grid silently loses
        # intensity. The calculation measures that loss and says so.
        with pytest.warns(UserWarning, match="antialias aperture removed"):
            make_ebsd(
                silicon_slab(),
                BackscatterDetector(max_angle=150, gpts=4),
                sampling=0.2,
            ).scan()

    def test_antialias_loss_is_reported_in_the_metadata(self):
        patterns = make_ebsd(
            silicon_slab(), BackscatterDetector(max_angle=50, gpts=4)
        ).scan()
        assert 0.0 <= patterns.metadata["antialias_loss_mean"] < 0.05
        assert (
            patterns.metadata["antialias_loss_mean"]
            <= patterns.metadata["antialias_loss_max"]
        )


class TestPatchGeometry:
    def test_half_angle_shrinks_as_patches_are_added(self):
        assert patch_half_angle(100) > patch_half_angle(400)
        # solid angle per patch goes as 1 / n, so the radius goes as 1 / sqrt(n)
        assert patch_half_angle(100) / patch_half_angle(400) == pytest.approx(2.0)

    def test_half_angle_covers_the_hemisphere(self):
        # The discs must at least cover the hemisphere: n * (1 - cos(a)) >= 1,
        # since a cap of half-angle a subtends 2*pi*(1 - cos a) of the 2*pi.
        for n_patches in (50, 200, 400, 1000):
            half_angle = patch_half_angle(n_patches) * 1e-3
            assert n_patches * (1.0 - np.cos(half_angle)) > 1.0

    def test_half_angle_rejects_non_positive(self):
        with pytest.raises(ValueError, match="n_patches must be at least 1"):
            patch_half_angle(0)

    @pytest.mark.parametrize("max_angle", [50.0, 132.0, 300.0])
    def test_recommended_sampling_stays_inside_the_aperture(self, max_angle):
        from abtem.core.energy import energy2wavelength

        sampling = recommended_sampling(30e3, max_angle)
        k_collected = np.sin(max_angle * 1e-3) / energy2wavelength(30e3)
        k_aperture = 2.0 / 3.0 / (2.0 * sampling)
        assert k_collected < k_aperture


class TestEBSDReferencePattern:
    @pytest.fixture
    def builder(self):
        return EBSDReferencePattern(
            ase.build.bulk("Si", "diamond", a=5.431),
            probe=abtem.Probe(semiangle_cutoff=10, energy=30e3),
            n_patches=1,
            slab_cell=(6.0, 6.0, 4.0),
            gpts=8,
            direction_gpts=30,
            max_angle=200.0,
        )

    def test_defaults_are_derived_from_the_patch_count(self):
        builder = EBSDReferencePattern(
            ase.build.bulk("Si", "diamond", a=5.431),
            probe=abtem.Probe(semiangle_cutoff=10, energy=30e3),
            n_patches=400,
            gpts=128,
        )
        assert builder.max_angle == pytest.approx(patch_half_angle(400))
        assert builder.sampling == pytest.approx(
            recommended_sampling(30e3, patch_half_angle(400))
        )
        assert len(builder.zone_axes) == 400

    def test_warns_about_too_coarse_a_sampling(self):
        with pytest.warns(UserWarning, match="cannot resolve"):
            EBSDReferencePattern(
                ase.build.bulk("Si", "diamond", a=5.431),
                probe=abtem.Probe(semiangle_cutoff=10, energy=30e3),
                n_patches=400,
                sampling=1.0,
            )

    def test_every_direction_is_assigned_exactly_once(self):
        builder = EBSDReferencePattern(
            ase.build.bulk("Si", "diamond", a=5.431),
            probe=abtem.Probe(semiangle_cutoff=10, energy=30e3),
            n_patches=40,
            gpts=16,
        )
        assignment = builder._assign_directions()
        counts = np.bincount(
            np.concatenate(assignment), minlength=len(builder.directions)
        )
        assert np.all(counts == 1)

    def test_overlap_tolerance_shares_directions_between_patches(self):
        common = dict(
            atoms=ase.build.bulk("Si", "diamond", a=5.431),
            probe=abtem.Probe(semiangle_cutoff=10, energy=30e3),
            n_patches=40,
            gpts=16,
        )
        without = EBSDReferencePattern(**common)
        with_overlap = EBSDReferencePattern(**common, overlap_tolerance=0.1)

        assert sum(map(len, with_overlap._assign_directions())) > sum(
            map(len, without._assign_directions())
        )

    def test_assignment_covers_the_whole_direction_grid(self):
        builder = EBSDReferencePattern(
            ase.build.bulk("Si", "diamond", a=5.431),
            probe=abtem.Probe(semiangle_cutoff=10, energy=30e3),
            n_patches=40,
            gpts=16,
        )
        assigned = np.unique(np.concatenate(builder._assign_directions()))
        assert len(assigned) == len(builder.directions)

    def test_compute_returns_directions_within_the_patch(self, builder):
        pattern = builder.compute(pbar=False)
        assert isinstance(pattern, SphericalPattern)
        assert 0 < len(pattern) <= len(builder.directions)

        zone_axis = builder.zone_axes[0]
        cosines = pattern.directions @ zone_axis
        assert np.all(cosines > np.cos(builder.max_angle * 1e-3))

    def test_compute_matches_a_direct_calculation(self, builder):
        # The builder is bookkeeping around EBSD.scan; running the one patch by
        # hand must give the same numbers.
        pattern = builder.compute(pbar=False)

        slab, rotation = rotated_slab(
            builder.atoms, builder.zone_axes[0], builder.slab_cell
        )
        direct = EBSD(
            abtem.Potential(
                slab,
                sampling=builder.sampling,
                slice_thickness=1.0,
                projection="finite",
            ),
            probe=abtem.Probe(semiangle_cutoff=10, energy=30e3),
            detector=BackscatterDetector(
                directions=pattern.directions @ rotation.T
            ),
        ).scan()

        assert np.allclose(pattern.array, direct.array, rtol=1e-6)

    def test_compute_records_the_setup_in_the_metadata(self, builder):
        metadata = builder.compute(pbar=False).metadata
        assert metadata["n_patches"] == 1
        assert metadata["max_angle"] == pytest.approx(200.0)
        assert metadata["energy"] == 30e3
        assert metadata["projection"] == "stereographic"

    def test_compute_projects_to_an_image(self, builder):
        # One patch covers a cap, not the hemisphere, so most of the image is
        # empty by construction.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseProjectionWarning)
            images = builder.compute(pbar=False).project(8)
        assert images.array.shape == (8, 8)
        assert np.all(np.isfinite(images.array))

    def test_energy_ensemble_is_refused(self):
        # The reciprocity waves would have to propagate at the backscattered
        # energy while the beam stays at its own; say so rather than failing
        # deep inside with "Energy is not defined".
        with pytest.raises(NotImplementedError, match="energy ensemble"):
            EBSD(
                silicon_slab(),
                probe=abtem.Probe(semiangle_cutoff=10, energy=[30e3, 20e3]),
                detector=BackscatterDetector(max_angle=50, gpts=4),
            )


class TestDetectorRecommendedSampling:
    def test_sized_from_the_corners_not_the_half_width(self):
        # A grid's corners sit sqrt(2) further out than max_angle, and sizing
        # the sampling from max_angle leaves them outside the antialias
        # aperture, where they lose essentially all their intensity.
        detector = BackscatterDetector(max_angle=150, gpts=64)
        assert detector.recommended_sampling(30e3) < recommended_sampling(30e3, 150)

    def test_keeps_every_direction_inside_the_aperture(self):
        from abtem.core.energy import energy2wavelength

        detector = BackscatterDetector(max_angle=150, gpts=64)
        sampling = detector.recommended_sampling(30e3)

        k_corner = np.sin(detector.max_scattering_angle) / energy2wavelength(30e3)
        assert k_corner < 2.0 / 3.0 / (2.0 * sampling)

    def test_a_detector_sized_this_way_barely_clips(self):
        detector = BackscatterDetector(max_angle=100, gpts=8)
        patterns = make_ebsd(
            silicon_slab(),
            detector,
            sampling=detector.recommended_sampling(30e3),
        ).scan()
        assert patterns.metadata["antialias_loss_max"] < 0.05


class TestSparseProjection:
    @staticmethod
    def pattern_sampled_in(projection, gpts=128):
        directions = validate_projection(projection).grid(gpts)
        return SphericalPattern(1.0 + directions[:, 2], directions)

    @pytest.mark.parametrize("projection", ["stereographic", "lambert"])
    def test_matching_projection_does_not_warn(self, projection):
        pattern = self.pattern_sampled_in(projection, gpts=148)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseProjectionWarning)
            pattern.project(128, projection)

    @pytest.mark.parametrize(
        "sampled, viewed",
        [("stereographic", "lambert"), ("lambert", "stereographic")],
    )
    def test_mismatched_projection_warns(self, sampled, viewed):
        # An even grid in one projection is uneven in the other, so binning
        # through the wrong one leaves a moire of empty pixels.
        pattern = self.pattern_sampled_in(sampled, gpts=148)
        with pytest.warns(SparseProjectionWarning, match="no sampled direction"):
            pattern.project(128, viewed)

    def test_too_many_pixels_warns(self):
        pattern = self.pattern_sampled_in("lambert", gpts=32)
        with pytest.warns(SparseProjectionWarning):
            pattern.project(256, "lambert")

    def test_stereographic_corners_do_not_count_as_holes(self):
        # The corners outside the disk are legitimately empty; only pixels
        # inside the projection's domain are holes.
        pattern = self.pattern_sampled_in("stereographic", gpts=148)
        image = pattern.project(128, "stereographic")
        assert np.count_nonzero(image.array) < 128**2
        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseProjectionWarning)
            pattern.project(128, "stereographic")


class TestPixelCenters:
    def test_spans_the_square_without_touching_the_edges(self):
        centers = pixel_centers(4)
        assert centers.shape == (16, 2)
        assert centers[:, 0].min() == pytest.approx(-0.75)
        assert centers[:, 0].max() == pytest.approx(0.75)

    def test_ordering_matches_the_raveled_image(self):
        gpts = 5
        centers = pixel_centers(gpts).reshape(gpts, gpts, 2)
        # first axis varies x, second varies y, as histogram2d bins them
        assert centers[1, 0, 0] < centers[2, 0, 0]
        assert centers[0, 1, 1] < centers[0, 2, 1]

    def test_bin_directions_counts(self):
        directions = np.array([[0.0, 0.0, 1.0]])
        image, counts = bin_directions(
            directions, np.array([7.0]), gpts=4, return_counts=True
        )
        assert counts.sum() == 1
        assert image[counts > 0] == pytest.approx(7.0)


class TestInterpolate:
    def test_exact_when_sampled_on_the_same_grid(self):
        # Sampling the pattern on the grid it is asked for makes the
        # interpolation a repackaging: the node values are the computed ones.
        gpts = 61
        directions = SquareLambertProjection().grid(gpts)
        values = 1.0 + directions[:, 2] + 0.3 * directions[:, 0]
        pattern = SphericalPattern(values, directions)

        image = np.asarray(pattern.interpolate(gpts, "lambert").array)

        nodes = np.linspace(-1, 1, gpts)
        x, y = np.meshgrid(nodes, nodes, indexing="ij")
        expected = SquareLambertProjection().unproject(
            np.stack([x.ravel(), y.ravel()], axis=1)
        )
        expected = 1.0 + expected[:, 2] + 0.3 * expected[:, 0]
        assert np.allclose(image.ravel(), expected, atol=1e-10)

    def test_no_nearest_fill_when_sampled_in_the_same_projection(self):
        directions = SquareLambertProjection().grid(101)
        pattern = SphericalPattern(1.0 + directions[:, 2], directions)
        with warnings.catch_warnings():
            warnings.simplefilter("error", SparseProjectionWarning)
            pattern.interpolate(51, "lambert")

    def test_warns_when_the_grid_reaches_past_the_sampled_directions(self):
        # A stereographic sampling does not reach the corners of the Lambert
        # square, so those nodes have to be filled from the nearest sample.
        directions = StereographicProjection().grid(101)
        pattern = SphericalPattern(1.0 + directions[:, 2], directions)
        with pytest.warns(SparseProjectionWarning, match="outside the sampled"):
            pattern.interpolate(51, "lambert")

    def test_stereographic_corners_are_zero(self):
        directions = StereographicProjection().grid(101)
        pattern = SphericalPattern(1.0 + directions[:, 2], directions)
        image = np.asarray(pattern.interpolate(51, "stereographic").array)
        assert image[0, 0] == 0.0
        assert image[25, 25] > 0.0

    def test_preserves_the_ensemble(self):
        directions = SquareLambertProjection().grid(41)
        array = np.stack([np.ones(len(directions)), 2 * np.ones(len(directions))])
        pattern = SphericalPattern(
            array, directions, ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))]
        )
        image = pattern.interpolate(21, "lambert")
        assert image.array.shape == (2, 21, 21)
        assert np.allclose(image.array[1], 2.0)


class TestEMsoftWriter:
    @pytest.fixture
    def silicon(self):
        return ase.build.bulk("Si", "diamond", a=5.431)

    @pytest.fixture
    def pattern(self):
        directions = SquareLambertProjection().grid(81)
        values = 1.0 + 0.5 * directions[:, 2] ** 2
        return SphericalPattern(values, directions, metadata={"energy": 30e3})

    @pytest.fixture
    def written(self, tmp_path, pattern, silicon):
        h5py = pytest.importorskip("h5py")
        pytest.importorskip("spglib")
        path = str(tmp_path / "master.h5")
        write_emsoft_master_pattern(path, pattern, silicon, npx=20)
        return path, h5py

    def test_array_shapes_follow_the_format(self, written):
        path, h5py = written
        with h5py.File(path) as f:
            assert f["EMData/EBSDmaster/mLPNH"].shape == (1, 1, 41, 41)
            assert f["EMData/EBSDmaster/mLPSH"].shape == (1, 1, 41, 41)
            assert f["EMData/EBSDmaster/masterSPNH"].shape == (1, 41, 41)
            assert f["CrystalData/AtomData"].shape[0] == 5
            assert f["CrystalData/LatticeParameters"].shape == (6,)

    def test_scalars_are_one_element_arrays(self, written):
        # EMsoft writes Fortran rank-1 arrays and readers index them as
        # `dataset[:][0]`, which raises on a true scalar dataspace.
        path, h5py = written
        with h5py.File(path) as f:
            for name in [
                "EMData/EBSDmaster/numset",
                "EMData/EBSDmaster/numEbins",
                "EMheader/EBSDmaster/ProgramName",
                "CrystalData/SpaceGroupNumber",
                "NMLparameters/EBSDMasterNameList/npx",
            ]:
                assert f[name].shape == (1,), name
                f[name][:][0]  # must not raise

    def test_writes_the_conventional_cell_in_nanometres(self, written):
        # ase.build.bulk gives the primitive rhombohedral cell, whose axes
        # would contradict the cubic space group recorded next to them.
        path, h5py = written
        with h5py.File(path) as f:
            lattice = f["CrystalData/LatticeParameters"][()]
            assert f["CrystalData/SpaceGroupNumber"][:][0] == 227
            assert np.allclose(lattice[:3], 0.5431, atol=1e-4)  # nm, not Å
            assert np.allclose(lattice[3:], 90.0)
            assert f["CrystalData/CrystalSystem"][:][0] == 1  # cubic

    def test_southern_hemisphere_is_the_centrosymmetric_image(self, written):
        path, h5py = written
        with h5py.File(path) as f:
            north = f["EMData/EBSDmaster/mLPNH"][0, 0]
            south = f["EMData/EBSDmaster/mLPSH"][0, 0]
        assert np.array_equal(south, north[::-1, ::-1])

    def test_energy_is_recorded_in_kev(self, written):
        path, h5py = written
        with h5py.File(path) as f:
            assert f["EMData/EBSDmaster/EkeVs"][()] == pytest.approx([30.0])

    def test_refuses_an_ensemble(self, tmp_path, silicon):
        pytest.importorskip("h5py")
        directions = SquareLambertProjection().grid(21)
        pattern = SphericalPattern(
            np.ones((2, len(directions))),
            directions,
            ensemble_axes_metadata=[OrdinalAxis(values=(0, 1))],
            metadata={"energy": 30e3},
        )
        with pytest.raises(ValueError, match="only a single pattern"):
            write_emsoft_master_pattern(
                str(tmp_path / "m.h5"), pattern, silicon, npx=10
            )

    def test_requires_an_energy(self, tmp_path, silicon):
        pytest.importorskip("h5py")
        directions = SquareLambertProjection().grid(21)
        pattern = SphericalPattern(np.ones(len(directions)), directions)
        with pytest.raises(ValueError, match="no energy"):
            write_emsoft_master_pattern(
                str(tmp_path / "m.h5"), pattern, silicon, npx=10
            )

    def test_rejects_an_impossible_space_group(self, tmp_path, pattern, silicon):
        pytest.importorskip("h5py")
        with pytest.raises(ValueError, match="between 1 and 230"):
            write_emsoft_master_pattern(
                str(tmp_path / "m.h5"), pattern, silicon, npx=10, space_group=300
            )
