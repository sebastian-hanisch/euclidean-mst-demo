"""Die Korrektheits-Kette für die Wege zum Euklidischen MST: alle vier Wege derselbe Baum; MST ⊆ RNG ⊆ Gabriel ⊆ Delaunay; die Annahme "euklidisch" bricht mit dem Geländezuschlag; Buchführung;
Single-Linkage aus dem MST gleich dem naiven Verfahren; kNN-Näherung."""

import itertools
import math

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

import emst_algorithm as A
from emst_unionfind import UnionFind
import emst_delaunay as D
import emst_methods as M
import emst_scenario as S


def _scipy_cost(n, edges):
    us, vs, ws = zip(*edges)
    return float(minimum_spanning_tree(coo_matrix((ws, (us, vs)), shape=(n, n)).tocsr()).sum())


def _is_tree(n, pairs):
    if len(pairs) != n - 1 or len(set(pairs)) != n - 1:
        return False
    uf = UnionFind(n, "full")
    return all(uf.union(u, v) for u, v in pairs) and uf.components == 1


def _instances():
    for kind in ("uniform", "clusters", "grid"):
        for n in (5, 12, 40, 90):
            for seed in (1, 2, 3):
                yield S.generate(n, kind, 3, 0.0, seed)


ALL = list(_instances())


def _brute_gabriel(xy):
    n = len(xy)
    out = set()
    for u in range(n):
        for v in range(u + 1, n):
            if not any((xy[u][0] - xy[r][0]) * (xy[v][0] - xy[r][0]) + (xy[u][1] - xy[r][1]) * (xy[v][1] - xy[r][1]) < 0 for r in range(n) if r not in (u, v)):
                out.add((u, v))
    return out


def _brute_rng(xy):
    n = len(xy)
    d = lambda a, b: math.hypot(xy[a][0] - xy[b][0], xy[a][1] - xy[b][1])
    out = set()
    for u in range(n):
        for v in range(u + 1, n):
            if not any(max(d(u, r), d(v, r)) < d(u, v) for r in range(n) if r not in (u, v)):
                out.add((u, v))
    return out


# --- alle Wege derselbe Baum ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("inst", ALL)
def test_all_four_ways_give_the_same_tree_even_with_tied_distances(inst):
    dt, comp = D.delaunay(inst.xy), S.complete_edges(inst)
    runs = {w: M.run_way(inst, w, dt, comp) for w in M.WAYS}
    ref = frozenset(runs["kruskal_complete"].tree)
    assert all(frozenset(r.tree) == ref and _is_tree(inst.n, r.tree) and r.connected for r in runs.values())
    assert all(r.cost == pytest.approx(runs["kruskal_complete"].cost) for r in runs.values()) and runs["kruskal_complete"].cost == pytest.approx(_scipy_cost(inst.n, comp))


@pytest.mark.parametrize("seed", range(6))
def test_matches_brute_force_on_tiny_instances(seed):
    inst = S.generate(6, "uniform", 3, 0.0, seed)
    edges = S.complete_edges(inst)
    best = min(sum(edges[i][2] for i in combo) for combo in itertools.combinations(range(len(edges)), 5) if _is_tree(6, [(edges[i][0], edges[i][1]) for i in combo]))
    assert M.run_way(inst, "delaunay_kruskal").cost == pytest.approx(best)


@pytest.mark.parametrize("inst", ALL)
def test_the_mst_lies_inside_the_delaunay_graph(inst):
    dt = D.delaunay(inst.xy)
    assert set(M.run_way(inst, "kruskal_complete").tree) <= set(dt.edges)


@pytest.mark.parametrize("inst", [i for i in ALL if i.kind != "grid"])
def test_hierarchy_mst_in_rng_in_gabriel_in_delaunay_in_general_position(inst):
    dt = D.delaunay(inst.xy)
    mst = set(M.run_way(inst, "kruskal_complete", dt).tree)
    gab, rg = set(M.gabriel_edges(inst.xy, dt)), set(M.rng_edges(inst.xy, dt))
    assert mst <= rg <= gab <= set(dt.edges) and len(mst) == inst.n - 1


@pytest.mark.parametrize("inst", [i for i in ALL if i.kind != "grid" and i.n <= 40])
def test_gabriel_and_rng_filters_equal_the_definitions_over_all_pairs(inst):
    dt = D.delaunay(inst.xy)
    assert set(M.gabriel_edges(inst.xy, dt)) == _brute_gabriel(inst.xy) and set(M.rng_edges(inst.xy, dt)) == _brute_rng(inst.xy)


def test_hierarchy_on_the_lattice_mst_is_unit_edges_and_stays_inside_the_layers():
    inst = S.generate(60, "grid", 3, 0.0, 1)
    dt = D.delaunay(inst.xy)
    mst, gab, rg = set(M.run_way(inst, "kruskal_complete", dt).tree), set(M.gabriel_edges(inst.xy, dt)), set(M.rng_edges(inst.xy, dt))
    assert mst <= rg <= gab <= set(dt.edges) and all(inst.xy[u].tolist() != inst.xy[v].tolist() for u, v in mst)
    assert all(math.hypot(*(inst.xy[u] - inst.xy[v])) == 10.0 for u, v in mst) and all(math.hypot(*(inst.xy[u] - inst.xy[v])) == 10.0 for u, v in rg)


def test_edge_counts_of_the_layers_follow_the_bounds():
    for inst in [S.generate(150, "uniform", 3, 0.0, s) for s in range(5)]:
        dt = D.delaunay(inst.xy)
        h = len(dt.hull)
        rg, gab = M.rng_edges(inst.xy, dt), M.gabriel_edges(inst.xy, dt)
        assert len(dt.edges) == 3 * inst.n - 3 - h and inst.n - 1 <= len(rg) <= len(gab) <= len(dt.edges)


# --- die Annahme "euklidisch" bricht ---------------------------------------------------------------------------------------------------


def test_with_terrain_the_mst_can_leave_the_delaunay_graph_and_never_without():
    misses = []
    for seed in range(40):
        inst = S.generate(60, "uniform", 3, 0.8, seed)
        dt = D.delaunay(inst.xy)
        true = M.run_way(inst, "kruskal_complete", dt)
        if not set(true.tree) <= set(dt.edges):
            misses.append(seed)
            restricted = M.run_way(inst, "delaunay_kruskal", dt)
            assert _is_tree(inst.n, restricted.tree) and restricted.cost >= true.cost - 1e-9 and restricted.cost > true.cost
    assert misses
    for seed in range(40):
        plain = S.generate(60, "uniform", 3, 0.0, seed)
        assert set(M.run_way(plain, "kruskal_complete").tree) <= set(D.delaunay(plain.xy).edges)


# --- Buchführung ---------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("inst", [ALL[5], ALL[20], ALL[35]])
def test_cost_breakdown_identities(inst):
    dt, comp = D.delaunay(inst.xy), S.complete_edges(inst)
    n = inst.n
    pa, kc, dk, dp = (M.run_way(inst, w, dt, comp) for w in M.WAYS)
    assert (pa.build_ops, pa.dist_ops, pa.edges_used) == (0, n * (n - 1) // 2, len(comp)) and (kc.build_ops, kc.dist_ops) == (0, len(comp))
    assert dk.build_ops == dp.build_ops == dt.ops and dk.dist_ops == dp.dist_ops == dk.edges_used == len(dt.edges) <= 3 * n - 6
    for r in (pa, kc, dk, dp):
        assert r.ops == r.build_ops + r.dist_ops + r.mst_ops
    assert dk.mst_ops == A.kruskal(n, S.edge_list(inst, dt.edges)).ops and dp.mst_ops == A.prim(n, S.edge_list(inst, dt.edges), "eager", 0).ops


def test_unknown_way_is_rejected():
    with pytest.raises(ValueError):
        M.run_way(S.generate(10, "uniform", 3, 0.0, 1), "boruvka")


def test_runs_are_deterministic():
    inst = S.generate(50, "clusters", 3, 0.0, 4)
    a, b = M.run_way(inst, "delaunay_prim"), M.run_way(inst, "delaunay_prim")
    assert a.tree == b.tree and a.ops == b.ops


# --- Single-Linkage aus dem MST --------------------------------------------------------------------------------------------------------------


def _naive_single_linkage(cost, n):
    """Bottom-up: verschmilzt immer die zwei Cluster mit dem kleinsten Abstand zwischen zwei ihrer Punkte (O(n³))."""
    clusters = [{i} for i in range(n)]
    heights = []
    while len(clusters) > 1:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = min(cost[u, v] for u in clusters[a] for v in clusters[b])
                if best is None or d < best[0]:
                    best = (d, a, b)
        d, a, b = best
        heights.append(d)
        clusters[a] |= clusters[b]
        del clusters[b]
    return heights, clusters


@pytest.mark.parametrize("kind", ["uniform", "clusters"])
@pytest.mark.parametrize("seed", range(4))
def test_single_linkage_heights_and_clusters_equal_the_naive_algorithm(kind, seed):
    inst = S.generate(28, kind, 3, 0.0, seed)
    comp = S.complete_edges(inst)
    run = M.run_way(inst, "delaunay_kruskal")
    cost = {(u, v): w for u, v, w in comp}
    tree = [(u, v, cost[(u, v)]) for u, v in run.tree]
    merges = M.single_linkage(inst.n, tree)
    naive_heights, _ = _naive_single_linkage(S.cost_matrix(inst), inst.n)
    assert [m[0] for m in merges] == pytest.approx(naive_heights) and all(a[0] <= b[0] for a, b in zip(merges, merges[1:])) and len(merges) == inst.n - 1
    assert merges[-1][1] + merges[-1][2] == inst.n
    for k in (1, 2, 3, 5, 9, inst.n):
        labels = M.clusters_from_tree(inst.n, tree, k)
        assert len(set(labels)) == k
        # k Cluster des naiven Verfahrens: bis nur noch k übrig sind
        cl = [{i} for i in range(inst.n)]
        cm = S.cost_matrix(inst)
        while len(cl) > k:
            d, a, b = min((min(cm[u, v] for u in cl[a] for v in cl[b]), a, b) for a in range(len(cl)) for b in range(a + 1, len(cl)))
            cl[a] |= cl[b]
            del cl[b]
        assert {frozenset(i for i in range(inst.n) if labels[i] == c) for c in set(labels)} == {frozenset(c) for c in cl}


def test_single_linkage_special_cases():
    tree = [(0, 1, 1.0), (1, 2, 2.0), (2, 3, 5.0)]
    assert M.single_linkage(4, tree) == [(1.0, 1, 1), (2.0, 2, 1), (5.0, 3, 1)]
    assert M.clusters_from_tree(4, tree, 1) == (0, 0, 0, 0) and M.clusters_from_tree(4, tree, 2) == (0, 0, 0, 1) and M.clusters_from_tree(4, tree, 4) == (0, 1, 2, 3)
    assert M.clusters_from_tree(4, tree, 99) == (0, 1, 2, 3) and M.clusters_from_tree(4, tree, 0) == (0, 0, 0, 0)


# --- kNN-Näherung ---------------------------------------------------------------------------------------------------------------------------------


def test_exact_emst_equals_kruskal_and_scipy():
    for seed in range(5):
        pts = np.random.default_rng(seed).uniform(0, 1, (40, 3))
        tree, cost = M.exact_emst(pts)
        edges = tuple((u, v, float(np.linalg.norm(pts[u] - pts[v]))) for u in range(40) for v in range(u + 1, 40))
        kr = A.kruskal(40, edges)
        assert _is_tree(40, tree) and cost == pytest.approx(kr.cost) == pytest.approx(_scipy_cost(40, edges)) and set(tree) == {(edges[i][0], edges[i][1]) for i in kr.tree}


def test_knn_edges_are_the_symmetric_union_of_the_nearest_neighbours():
    pts = np.random.default_rng(1).uniform(0, 1, (30, 4))
    e = set(M.knn_edges(pts, 3))
    dm = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    ref = set()
    for u in range(30):
        for v in sorted((x for x in range(30) if x != u), key=lambda x: (dm[u, x], x))[:3]:
            ref.add((min(u, v), max(u, v)))
    assert e == ref and all(u < v for u, v in e) and M.knn_edges(pts, 29) == [(u, v) for u in range(30) for v in range(u + 1, 30)] == M.knn_edges(pts, 99)


def test_knn_mst_with_all_neighbours_is_exact():
    pts = np.random.default_rng(2).uniform(0, 1, (25, 5))
    r = M.knn_mst(pts, 24)
    assert r.coverage == 1.0 and r.error == pytest.approx(0.0, abs=1e-12) and r.connected and r.repairs == 0 and r.missing == []


@pytest.mark.parametrize("d", [2, 5, 20])
@pytest.mark.parametrize("k", [1, 3, 8])
def test_knn_mst_is_a_spanning_tree_with_nonnegative_error(d, k):
    pts = np.random.default_rng(d * 10 + k).uniform(0, 1, (60, d))
    r = M.knn_mst(pts, k)
    assert _is_tree(60, r.tree) and r.error >= -1e-12 and 0.0 <= r.coverage <= 1.0
    assert r.cost == pytest.approx(sum(np.linalg.norm(pts[u] - pts[v]) for u, v in r.tree))
    assert r.repairs == (0 if r.connected else r.repairs) and (r.connected or r.repairs >= 1)
    assert len(r.missing) == round((1.0 - r.coverage) * 59)


def test_knn_graph_of_two_far_apart_clusters_needs_repairs():
    rng = np.random.default_rng(4)
    pts = np.vstack([rng.normal(0, 0.1, (15, 2)), rng.normal(50, 0.1, (15, 2))])
    r = M.knn_mst(pts, 3)
    assert not r.connected and r.repairs == 1 and r.error == pytest.approx(0.0, abs=1e-9)


def test_knn_special_cases():
    two = M.knn_mst(np.array([[0.0, 0.0], [3.0, 4.0]]), 1)
    assert two.tree == [(0, 1)] and two.cost == pytest.approx(5.0) and two.coverage == 1.0
    line = np.array([[float(i), 0.0] for i in range(6)])
    assert M.knn_mst(line, 1).error == pytest.approx(0.0, abs=1e-12)
    assert M.exact_emst(np.array([[1.0, 1.0]])) == ([], 0.0)
