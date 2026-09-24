import numpy as np
import pytest

import emst_constants as C
import emst_scenario as S


@pytest.mark.parametrize("kind", C.KINDS)
@pytest.mark.parametrize("n", [5, 30, 120])
def test_generation_gives_n_distinct_points_inside_the_area(kind, n):
    inst = S.generate(n, kind, 4, 0.0, 3)
    assert inst.n == n and inst.kind == kind and len({tuple(p) for p in inst.xy}) == n
    if kind != "grid":
        assert inst.xy.min() >= 0.0 and inst.xy.max() <= C.AREA


def test_generation_is_deterministic_and_seed_dependent():
    a, b, c = S.generate(40, "uniform", 4, 0.0, 4), S.generate(40, "uniform", 4, 0.0, 4), S.generate(40, "uniform", 4, 0.0, 5)
    assert np.array_equal(a.xy, b.xy) and not np.array_equal(a.xy, c.xy)
    assert np.array_equal(S.generate(40, "clusters", 3, 0.0, 4).xy, S.generate(40, "clusters", 3, 0.0, 4).xy)


def test_grid_has_integer_coordinates_and_many_equal_distances():
    inst = S.generate(50, "grid", 4, 0.0, 1)
    assert (inst.xy % C.GRID_SPACING == 0).all() and inst.xy[:, 0].max() == C.GRID_SPACING * 7
    ds = [w for _u, _v, w in S.complete_edges(inst)]
    assert len(set(ds)) < len(ds) / 10


def test_clusters_are_tighter_than_uniform_points():
    c = S.generate(120, "clusters", 3, 0.0, 7)
    u = S.generate(120, "uniform", 3, 0.0, 7)
    nn = lambda inst: np.sort(S.cost_matrix(inst) + np.diag(np.full(inst.n, np.inf)), axis=1)[:, 0].mean()
    assert nn(c) < nn(u)


def test_more_clusters_are_accepted_and_the_count_is_stored_only_for_clusters():
    assert S.generate(60, "clusters", 8, 0.0, 1).clusters == 8 and S.generate(60, "uniform", 8, 0.0, 1).clusters == 0


def test_zero_terrain_gives_exactly_euclidean_costs_and_no_factor():
    inst = S.generate(20, "uniform", 4, 0.0, 3)
    assert inst.factor is None
    assert all(w == pytest.approx(float(np.hypot(*(inst.xy[u] - inst.xy[v])))) for u, v, w in S.complete_edges(inst))


def test_terrain_factor_is_symmetric_in_range_and_keeps_the_points():
    plain, rough = S.generate(30, "uniform", 4, 0.0, 9), S.generate(30, "uniform", 4, 0.6, 9)
    assert np.array_equal(plain.xy, rough.xy)
    f = rough.factor
    assert np.allclose(f, f.T) and f[np.triu_indices(30, 1)].min() >= 1.0 and f[np.triu_indices(30, 1)].max() <= 1.6
    for u, v, w in S.complete_edges(rough):
        d = float(np.hypot(*(rough.xy[u] - rough.xy[v])))
        assert d - 1e-9 <= w <= d * 1.6 + 1e-9


def test_complete_edges_and_edge_list_are_canonical():
    inst = S.generate(15, "uniform", 4, 0.3, 2)
    e = S.complete_edges(inst)
    assert len(e) == 15 * 14 // 2 and all(u < v for u, v, _w in e) and list(e) == sorted(e, key=lambda t: (t[0], t[1]))
    sub = S.edge_list(inst, [(3, 5), (0, 2), (1, 4)])
    assert [(u, v) for u, v, _w in sub] == [(0, 2), (1, 4), (3, 5)] and sub[0][2] == pytest.approx(next(w for u, v, w in e if (u, v) == (0, 2)))


def test_cost_matrix_is_symmetric():
    c = S.cost_matrix(S.generate(20, "uniform", 4, 0.4, 5))
    assert np.allclose(c, c.T) and np.allclose(np.diag(c), 0.0)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        S.generate(10, "spirale", 4, 0.0, 1)
