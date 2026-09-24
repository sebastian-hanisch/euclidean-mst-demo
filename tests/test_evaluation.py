"""Auswertung: Analysis-Felder gegen unabhängige Neuberechnung, Gewinner, run_config/sweep, Kreuzungspunkt, Gelände-, Chaining- und Hochdimensions-Experiment."""

import numpy as np
import pytest

import emst_constants as C
import emst_delaunay as D
import emst_evaluation as ev
import emst_methods as M
import emst_scenario as S

SMALL = dict(seeds=C.SWEEP_SEEDS[:2])


def test_default_settings_come_from_the_constants_and_are_members_of_the_controls():
    s = ev.Settings()
    assert (s.n, s.clusters, s.terrain, s.seed, s.kind) == (C.DEFAULT_N, C.DEFAULT_CLUSTERS, C.DEFAULT_TERRAIN, C.DEFAULT_SEED, "uniform")
    assert s.terrain in C.TERRAIN_OPTIONS and s.kind in C.KINDS and C.DEFAULT_LAYER in C.LAYERS and C.CLUSTERS_MIN <= s.clusters <= C.CLUSTERS_MAX
    assert set(ev.SWEEP_VALUES["terrain"]) <= set(C.TERRAIN_OPTIONS) and max(C.N_SWEEP) <= C.N_MAX


@pytest.mark.parametrize("kind", C.KINDS)
def test_analysis_fields_match_an_independent_recomputation(kind):
    a = ev.analyse(ev.Settings(kind=kind, n=50, seed=7))
    inst = S.generate(50, kind, C.DEFAULT_CLUSTERS, 0.0, 7)
    dt = D.delaunay(inst.xy)
    assert a.dt.edges == dt.edges and a.ops == {w: M.run_way(inst, w).ops for w in ev.WAYS}
    assert a.same_tree and a.missing == [] and a.overhead == 0.0 and a.bound == len(dt.edges) == 3 * 50 - 3 - len(dt.hull)
    lay = a.layers
    assert set(lay["mst"]) <= set(lay["rng"]) <= set(lay["gabriel"]) <= set(lay["delaunay"]) and len(lay["mst"]) == 49
    assert a.gabriel == M.gabriel_edges(inst.xy, dt) and a.rng == M.rng_edges(inst.xy, dt)


def test_winner_is_the_cheapest_way_with_a_deterministic_tie_break():
    a = ev.analyse(ev.Settings())
    assert a.ops[a.winner] == min(a.ops.values()) and a.winner == "delaunay_prim"
    assert ev.analyse(ev.Settings(n=20)).winner == "prim_array"


def test_per_point_costs_are_consistent_with_the_counters():
    a = ev.analyse(ev.Settings(n=80))
    pp = a.per_point
    assert pp["build"] == pytest.approx(a.dt.ops / 80) and pp["orient"] == pytest.approx(a.dt.orient_tests / 80) and pp["cavity"] == pytest.approx(np.mean(a.dt.cavity_sizes))
    assert pp["created"] == pytest.approx(a.dt.created / 80)


def test_terrain_makes_the_dt_ways_differ_from_the_true_tree_and_costs_at_least_as_much():
    a = ev.analyse(ev.Settings(terrain=0.4, seed=35))
    assert a.missing and not a.same_tree and a.overhead > 0
    assert a.runs["delaunay_kruskal"].cost >= a.runs["kruskal_complete"].cost


def test_run_config_keys_ranges_and_base_seed_independence():
    r = ev.run_config(ev.Settings(seed=1, n=30), **SMALL)
    assert r["n_runs"] == 2 and r["same_tree_share"] == 100.0 and r["miss_share"] == 0.0 and sum(r[f"wins_{w}"] for w in ev.WAYS) == 2
    for key in ("ops_prim_array", "ops_kruskal_complete", "ops_delaunay_kruskal", "ops_delaunay_prim", "pairs", "dt_edges", "gabriel_edges", "rng_edges", "mst_edges", "bound", "hull", "build", "build_per_point",
                "orient_per_point", "incircle_per_point", "cavity", "speedup_prim", "speedup_kruskal", "overhead", "missing"):
        assert r[f"{key}_lo"] <= r[key] <= r[f"{key}_hi"]
    assert r == pytest.approx(ev.run_config(ev.Settings(seed=999, n=30), **SMALL))


def test_sweep_rows_and_labels():
    rows = ev.sweep("n", ev.Settings(seed=0), values=(10, 20))
    assert [r["value"] for r in rows] == [10, 20] and rows[1]["ops_kruskal_complete"] > rows[0]["ops_kruskal_complete"]
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS) == {"n", "terrain"}
    assert [r["value"] for r in ev.sweep("terrain", ev.Settings(seed=0, n=20), values=(0.0, 0.4))] == [0.0, 0.4]


def test_crossover_returns_the_smallest_n_from_which_delaunay_prim_stays_cheaper():
    n_star, rows = ev.crossover(ev.Settings(seed=0), ns=(10, 20, 30, 40), seeds=C.SWEEP_SEEDS[:3])
    assert [r["n"] for r in rows] == [10, 20, 30, 40] and n_star is not None
    assert all(r["ops_delaunay_prim"] < r["ops_prim_array"] for r in rows if r["n"] >= n_star)
    assert ev.crossover(ev.Settings(seed=0), ns=(10, 15), seeds=C.SWEEP_SEEDS[:2])[0] is None


def test_terrain_miss_rows_and_the_euclidean_case():
    rows = ev.terrain_miss(ev.Settings(n=30), terrains=(0.0, 0.4), seeds=C.TERRAIN_SEEDS[:10])
    assert [r["terrain"] for r in rows] == [0.0, 0.4] and rows[0]["miss_share"] == 0.0 and rows[0]["overhead_max"] == 0.0 and rows[1]["miss_share"] > 0 and rows[1]["overhead_max"] >= rows[1]["overhead_mean"] >= 0
    assert all(r["n_runs"] == 10 for r in rows)


def test_linkage_profile_of_uniform_and_cluster_instances():
    u = ev.linkage_profile(ev.analyse(ev.Settings(n=40)))
    assert u["purity"] is None and len(u["heights"]) == 39 and u["heights"] == sorted(u["heights"]) and len(u["tree"]) == 39
    c = ev.linkage_profile(ev.analyse(ev.Settings(kind="clusters", n=60, clusters=3)))
    assert 0.0 < c["purity"] <= 1.0 and c["merges"][-1][1] + c["merges"][-1][2] == 60


def test_chaining_rows_are_purities_over_the_cloud_counts():
    rows = ev.chaining(ev.Settings(n=40), clusters=(2, 6), seeds=C.TERRAIN_SEEDS[:10])
    assert [r["clusters"] for r in rows] == [2, 6] and all(0 < r["purity_lo"] <= r["purity"] <= r["purity_hi"] <= 1 and 0 <= r["perfect_share"] <= 100 for r in rows)


def test_highdim_rows_shape_and_invariants():
    rows = ev.highdim(ds=(2, 10), ks=(2, 5), n=40, seeds=C.SWEEP_SEEDS[:3])
    assert [(r["d"], r["k"]) for r in rows] == [(2, 2), (2, 5), (10, 2), (10, 5)]
    for r in rows:
        assert 0 <= r["coverage_lo"] <= r["coverage"] <= r["coverage_hi"] <= 1 and r["error_lo"] <= r["error"] <= r["error_hi"] and r["error"] >= -1e-9 and 0 <= r["connected_share"] <= 100
    assert rows[1]["coverage"] >= rows[0]["coverage"] and rows[1]["edges"] > rows[0]["edges"]
