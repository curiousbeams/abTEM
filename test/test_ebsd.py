import ase
import numpy as np
import pytest

from abtem.ebsd import (
    SquareLambertProjection,
    StereographicProjection,
    bin_directions,
    estimate_repetitions,
    fibonacci_hemisphere,
    rotated_slab,
    validate_projection,
    zone_axis_rotation,
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
