"""Die Korrektheits-Kette der Delaunay-Triangulierung: leere Umkreise, Euler-Formel, Kantenmenge und Hülle gleich scipy, Sonderfälle, Buchführung der Ereignisse."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull, Delaunay

import emst_algorithm as A
import emst_delaunay as D
import emst_scenario as S


def _scipy_edges(pts):
    out = set()
    for s in Delaunay(pts).simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            out.add((min(s[a], s[b]), max(s[a], s[b])))
    return out


def _cases():
    for seed in range(60):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(4, 130))
        kind = seed % 5
        if kind == 0:
            pts = rng.uniform(0, 100, (n, 2))
        elif kind == 1:
            c = rng.uniform(10, 90, (4, 2))
            pts = c[rng.integers(0, 4, n)] + rng.normal(0, 3, (n, 2))
        elif kind == 2:
            pts = rng.uniform(0, 1, (n, 2)) * np.array([100, rng.choice([1, 0.1, 0.01])])
        elif kind == 3:
            pts = rng.normal(50, 20, (n, 2))
        else:
            th = rng.uniform(0, 2 * np.pi, n)
            pts = 50 + 40 * np.c_[np.cos(th), np.sin(th)] + rng.normal(0, 0.01, (n, 2))
        yield pytest.param(pts, id=f"seed{seed}_n{n}_kind{kind}")


CASES = list(_cases())


def _area(pts, tri):
    a, b, c = (pts[i] for i in tri)
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


@pytest.mark.parametrize("pts", CASES)
def test_edges_and_hull_equal_scipy_in_general_position(pts):
    r = D.delaunay(pts)
    assert set(r.edges) == _scipy_edges(pts) and set(r.hull) == set(ConvexHull(pts).vertices)


@pytest.mark.parametrize("pts", CASES)
def test_euler_counts_orientation_and_area(pts):
    r = D.delaunay(pts)
    n, h = len(pts), len(r.hull)
    assert len(r.triangles) == 2 * n - 2 - h and len(r.edges) == 3 * n - 3 - h
    assert all(_area(pts, t) > 0 for t in r.triangles)
    assert sum(_area(pts, t) for t in r.triangles) == pytest.approx(ConvexHull(pts).volume, rel=1e-9)


@pytest.mark.parametrize("pts", CASES[:20])
def test_every_circumcircle_is_empty(pts):
    r = D.delaunay(pts)
    xs, ys = pts[:, 0], pts[:, 1]
    scale = max(np.ptp(xs), np.ptp(ys)) ** 4
    for a, b, c in r.triangles:
        for p in range(len(pts)):
            if p not in (a, b, c):
                assert D.incircle(xs[a], ys[a], xs[b], ys[b], xs[c], ys[c], xs[p], ys[p]) <= 1e-9 * scale


@pytest.mark.parametrize("pts", CASES[:20])
def test_interior_edges_have_two_triangles_and_hull_edges_one(pts):
    r = D.delaunay(pts)
    count = {}
    for a, b, c in r.triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            count[(min(u, v), max(u, v))] = count.get((min(u, v), max(u, v)), 0) + 1
    assert set(count.values()) <= {1, 2} and {x for e, k in count.items() if k == 1 for x in e} == set(r.hull)


def test_lattice_with_cocircular_points_is_a_valid_triangulation():
    pts = np.array([(10.0 * i, 10.0 * j) for i in range(12) for j in range(9)])
    r = D.delaunay(pts)
    n, h = len(pts), len(r.hull)
    xs, ys = pts[:, 0], pts[:, 1]
    assert h == 2 * (12 + 9) - 4 and len(r.edges) == 3 * n - 3 - h and len(r.triangles) == 2 * n - 2 - h
    assert all(D.incircle(xs[a], ys[a], xs[b], ys[b], xs[c], ys[c], xs[p], ys[p]) <= 1e-9 for a, b, c in r.triangles for p in range(n) if p not in (a, b, c))
    unit = {(min(a, b), max(a, b)) for a in range(n) for b in range(a + 1, n) if np.hypot(*(pts[a] - pts[b])) == 10.0}
    assert unit <= set(r.edges)


def test_generated_instances_of_every_kind_are_triangulated():
    for kind in ("uniform", "clusters", "grid"):
        inst = S.generate(120, kind, 4, 0.0, 5)
        r = D.delaunay(inst.xy)
        assert len(r.edges) == 3 * 120 - 3 - len(r.hull) and len(r.triangles) == 2 * 120 - 2 - len(r.hull)


def test_result_does_not_depend_on_the_order_of_the_points_in_general_position():
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 100, (80, 2))
    perm = rng.permutation(80)
    a = {tuple(sorted((tuple(pts[u]), tuple(pts[v])))) for u, v in D.delaunay(pts).edges}
    b = {tuple(sorted((tuple(pts[perm][u]), tuple(pts[perm][v])))) for u, v in D.delaunay(pts[perm]).edges}
    assert a == b


def test_extremely_flat_point_clouds_still_give_the_exact_triangulation():
    rng = np.random.default_rng(11)
    for aspect in (1e2, 1e3, 1e4, 1e6):
        pts = rng.uniform(0, 1, (60, 2)) * np.array([aspect, 1.0])
        r = D.delaunay(pts)
        assert set(r.edges) == _scipy_edges(pts) and set(r.hull) == set(ConvexHull(pts).vertices)


def test_three_and_four_points():
    r3 = D.delaunay(np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 3.0]]))
    assert r3.triangles == [(0, 1, 2)] and r3.edges == [(0, 1), (0, 2), (1, 2)] and r3.hull == [0, 1, 2]
    convex = D.delaunay(np.array([[0.0, 0.0], [4.0, 0.0], [4.1, 3.0], [0.0, 3.2]]))
    assert len(convex.triangles) == 2 and len(convex.edges) == 5
    concave = D.delaunay(np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0], [5.0, 2.0]]))
    assert len(concave.triangles) == 3 and len(concave.edges) == 6 and concave.hull == [0, 1, 2]


def test_collinear_points_give_a_path_and_tiny_inputs_are_handled():
    pts = np.array([[3.0, 3.0], [0.0, 0.0], [2.0, 2.0], [1.0, 1.0]])
    r = D.delaunay(pts)
    assert r.collinear and r.triangles == [] and r.edges == [(1, 3), (2, 3), (0, 2)] and r.ops == 0
    assert D.delaunay(np.array([[1.0, 1.0]])).edges == [] and D.delaunay(np.array([[1.0, 1.0], [2.0, 2.0]])).edges == [(0, 1)] and D.delaunay(np.empty((0, 2))).edges == []


def test_duplicate_points_are_rejected():
    with pytest.raises(ValueError):
        D.delaunay(np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [5.0, 0.0]]))


def test_orient_and_incircle_by_hand():
    assert D.orient(0, 0, 1, 0, 0, 1) > 0 and D.orient(0, 0, 1, 0, 0, -1) < 0 and D.orient(0, 0, 1, 0, 2, 0) == 0
    assert D.incircle(0, 0, 2, 0, 0, 2, 1, 1) > 0 and D.incircle(0, 0, 2, 0, 0, 2, 5, 5) < 0 and D.incircle(0, 0, 2, 0, 0, 2, 2, 2) == 0


# --- Buchführung ---------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("pts", CASES[:15])
def test_events_replay_to_the_final_triangulation_and_count_consistently(pts):
    r = D.delaunay(pts)
    n = len(pts)
    assert D.replay(r, 0) == {(n, n + 1, n + 2)}
    final = D.replay(r, n)
    assert {D.canonical(t) for t in final if max(t) < n} == set(r.triangles)
    assert 1 + sum(len(a) - len(rem) for _p, rem, a in r.events) == len(final)
    assert all(len(rem) >= 1 and len(a) == len(rem) + 2 for _p, rem, a in r.events)                        # jeder Hohlraum ist ein Vieleck mit (Randkanten) = (entfernte Dreiecke) + 2
    assert r.created == sum(len(a) for _p, _rem, a in r.events) and r.cavity_sizes == [len(rem) for _p, rem, _a in r.events]
    assert sorted(r.order) == list(range(n)) and [e[0] for e in r.events] == r.order
    assert len(r.per_insert) == n and sum(x[0] for x in r.per_insert) == r.orient_tests and sum(x[1] for x in r.per_insert) == r.walk_steps and sum(x[2] for x in r.per_insert) == r.incircle_tests
    assert r.ops == r.sort_comparisons + r.orient_tests + r.incircle_tests + r.created and r.walk_steps <= r.orient_tests


def test_spatial_order_is_a_permutation_and_counts_its_comparisons():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 100, (64, 2))
    order, cmp = D.spatial_order(list(pts[:, 0]), list(pts[:, 1]))
    assert sorted(order) == list(range(64)) and 64 * 3 <= cmp <= 64 * 7
    assert A.counted_sort([(0, 0.0, i) for i in range(64)])[1] == 192


def test_build_cost_grows_roughly_linearly_and_stays_far_below_the_pair_count():
    ops = {}
    for n in (50, 100, 200, 400):
        pts = np.random.default_rng(n).uniform(0, 100, (n, 2))
        ops[n] = D.delaunay(pts).ops
    assert all(ops[n] < 0.5 * n * (n - 1) / 2 for n in (200, 400)) and ops[400] / ops[200] < 2.6 and ops[200] / ops[100] < 2.6


def test_results_are_deterministic():
    pts = np.random.default_rng(7).uniform(0, 100, (100, 2))
    a, b = D.delaunay(pts), D.delaunay(pts)
    assert a.edges == b.edges and a.ops == b.ops and a.events == b.events
