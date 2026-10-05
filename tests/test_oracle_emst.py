"""Unabhängige Orakel (anderer Rechenweg als der Demo-Code): scipy.cluster.hierarchy (Single-Linkage), scipy.sparse.csgraph (MST des kNN-Graphen samt Reparatur über die Komponenten),
networkx (MST auf den Delaunay-Kanten mit Geländezuschlag, fehlende MST-Kanten, Kostenaufschlag)."""

import numpy as np
import pytest

import emst_evaluation as ev
import emst_methods as M
import emst_scenario as S

scipy_h = pytest.importorskip("scipy.cluster.hierarchy")
csgraph = pytest.importorskip("scipy.sparse.csgraph")
nx = pytest.importorskip("networkx")


def _dist(xy):
    return np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])


@pytest.mark.parametrize("seed", range(12))
def test_single_linkage_equals_scipy_linkage(seed):
    n = 8 + 3 * seed
    inst = S.generate(n, "clusters" if seed % 2 else "uniform", 3, 0.0, 400 + seed)
    d = _dist(inst.xy)
    tree = [(u, v, d[u, v]) for u, v in M.run_way(inst, "kruskal_complete").tree]
    z = scipy_h.linkage(inst.xy, "single")
    merges = M.single_linkage(n, tree)
    assert np.allclose(sorted(m[0] for m in merges), sorted(z[:, 2]))
    assert sorted(m[1] + m[2] for m in merges) == sorted(int(x) for x in z[:, 3])
    for k in range(1, 7):
        lab, ref = M.clusters_from_tree(n, tree, k), scipy_h.fcluster(z, k, "maxclust")
        assert {frozenset(i for i in range(n) if lab[i] == c) for c in set(lab)} == {frozenset(i for i in range(n) if ref[i] == c) for c in set(ref)}


@pytest.mark.parametrize("seed", range(10))
def test_terrain_costs_missing_edges_and_overhead_equal_networkx(seed):
    n, terrain = 10 + 2 * seed, (0.2, 0.8)[seed % 2]
    an = ev.analyse(ev.Settings("uniform", n, 3, terrain, 900 + seed))
    cm = S.cost_matrix(an.inst)
    full = nx.Graph()
    full.add_weighted_edges_from((u, v, cm[u, v]) for u in range(n) for v in range(u + 1, n))
    true = {(min(u, v), max(u, v)) for u, v in nx.minimum_spanning_edges(full, data=False)}
    assert set(an.true_tree) == true
    restricted = nx.Graph()
    restricted.add_weighted_edges_from((u, v, cm[u, v]) for u, v in an.dt.edges)
    cost_dt = sum(d["weight"] for *_e, d in nx.minimum_spanning_edges(restricted, data=True))
    cost_true = sum(cm[u, v] for u, v in true)
    assert an.runs["delaunay_prim"].cost == pytest.approx(cost_dt) and an.runs["delaunay_kruskal"].cost == pytest.approx(cost_dt)
    assert an.overhead == pytest.approx(100.0 * (cost_dt / cost_true - 1.0))
    assert sorted(an.missing) == sorted(e for e in true if e not in set(an.dt.edges))


@pytest.mark.parametrize("seed", range(20))
def test_knn_mst_equals_scipy_mst_of_the_knn_graph_plus_cheapest_component_links(seed):
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(8, 35)), int(rng.integers(1, 5))
    pts = rng.uniform(size=(n, 3))
    dm = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    g = np.zeros((n, n))
    for u, v in M.knn_edges(pts, k):
        g[u, v] = dm[u, v]
    ncomp, labs = csgraph.connected_components(g, directed=False)
    extra = 0.0
    if ncomp > 1:
        cd = np.zeros((ncomp, ncomp))
        for a in range(ncomp):
            for b in range(a + 1, ncomp):
                cd[a, b] = dm[np.ix_(labs == a, labs == b)].min()
        extra = csgraph.minimum_spanning_tree(cd).sum()
    run = M.knn_mst(pts, k)
    assert run.connected == (ncomp == 1) and run.repairs == ncomp - 1
    assert run.cost == pytest.approx(csgraph.minimum_spanning_tree(g).sum() + extra)
    assert run.cost >= csgraph.minimum_spanning_tree(dm).sum() - 1e-12
