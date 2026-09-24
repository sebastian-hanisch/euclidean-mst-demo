"""Jede Zahl, die App-Text und README nennen, wird hier über die echten Auswertungsfunktionen (ev.run_config / ev.sweep / ev.crossover / ev.terrain_miss / ev.chaining / ev.highdim) belegt - nie über ein
Ad-hoc-Skript. Die Elementarschritte sind ganzzahlig und plattformunabhängig (Delaunay in reiner Python-Arithmetik, eigener Mergesort und Heap); Kennzahlen aus Matrixprodukten (kNN) nur mit Rändern."""

from dataclasses import replace
from functools import lru_cache

import pytest

import emst_constants as C
import emst_evaluation as ev

BASE = ev.Settings(seed=0)


@lru_cache(maxsize=None)
def _cfg(**kw):
    return ev.run_config(replace(BASE, **kw))


@lru_cache(maxsize=None)
def _sweep(param, **kw):
    return ev.sweep(param, replace(BASE, **kw))


@lru_cache(maxsize=None)
def _cross(**kw):
    return ev.crossover(replace(BASE, **kw))


@lru_cache(maxsize=None)
def _terrain(**kw):
    return ev.terrain_miss(replace(BASE, **kw))


@lru_cache(maxsize=None)
def _chain():
    return ev.chaining(BASE)


@lru_cache(maxsize=None)
def _hd():
    return ev.highdim()


def _col(rows, key, digits=2):
    return [round(r[key], digits) for r in rows]


def test_same_tree_and_mst_inside_delaunay_for_every_kind_and_size():
    for kind in C.KINDS:
        for r in _sweep("n", kind=kind):
            assert r["same_tree_share"] == 100.0 and r["miss_share"] == 0.0


def test_uniform_step_counts_over_n():
    rows = _sweep("n")
    assert _col(rows, "ops_prim_array", 0) == [126, 551, 2301, 9401, 38001, 134251] and _col(rows, "ops_kruskal_complete", 0) == [277, 1540, 7591, 36986, 171916, 684295]
    assert _col(rows, "ops_delaunay_kruskal", 0) == [319, 823, 1973, 4638, 10348, 21111] and _col(rows, "ops_delaunay_prim", 0) == [286, 681, 1576, 3647, 8072, 16766]


def test_speedup_ratios_over_n():
    rows = _sweep("n")
    assert _col(rows, "speedup_prim") == [0.44, 0.81, 1.46, 2.58, 4.71, 8.01] and _col(rows, "speedup_kruskal", 1) == [0.9, 1.9, 3.9, 8.0, 16.6, 32.4]


def test_winners_switch_from_the_array_to_delaunay_prim_between_n20_and_n40_and_kruskal_never_wins():
    for kind in C.KINDS:
        rows = _sweep("n", kind=kind)
        assert [r["wins_prim_array"] for r in rows] == [5, 5, 0, 0, 0, 0] and [r["wins_delaunay_prim"] for r in rows] == [0, 0, 5, 5, 5, 5]
        assert all(r["wins_kruskal_complete"] == r["wins_delaunay_kruskal"] == 0 for r in rows)


def test_crossover_points():
    assert _cross()[0] == 30 and _cross(kind="clusters")[0] == 30 and _cross(kind="grid")[0] == 25
    rows = dict((r["n"], r) for r in _cross()[1])
    assert rows[25]["ops_prim_array"] < rows[25]["ops_delaunay_prim"] and rows[30]["ops_delaunay_prim"] < rows[30]["ops_prim_array"]


def test_candidate_graph_edge_counts_over_n():
    rows = _sweep("n")
    assert _col(rows, "dt_edges", 0) == [22, 49, 109, 227, 465, 882] == _col(rows, "bound", 0)
    assert _col(rows, "gabriel_edges", 0) == [14, 30, 69, 143, 291, 562] and _col(rows, "rng_edges", 0) == [10, 21, 47, 96, 192, 365] and _col(rows, "mst_edges", 0) == [9, 19, 39, 79, 159, 299]
    last = rows[-1]
    assert (round(last["dt_edges"] / 300, 1), round(last["gabriel_edges"] / 300, 1), round(last["rng_edges"] / 300, 1), last["pairs"]) == (2.9, 1.9, 1.2, 44850)
    grid = _sweep("n", kind="grid")[-1]
    assert (grid["dt_edges"], grid["gabriel_edges"]) == (837, 832)


def test_build_cost_per_point_grows_slowly():
    rows = _sweep("n")
    assert _col(rows, "build_per_point", 1) == [19.1, 22.4, 25.0, 28.7, 31.8, 34.5] and _col(rows, "orient_per_point", 1) == [7.4, 7.3, 7.9, 8.8, 9.8, 10.1]
    assert _col(rows, "incircle_per_point", 1) == [5.2, 6.5, 7.7, 8.6, 9.5, 10.6] and _col(rows, "cavity", 1) == [2.8, 3.2, 3.6, 4.0, 4.4, 4.9]
    assert round(_sweep("n", kind="grid")[-1]["build_per_point"]) == 44


def test_terrain_break_shares_and_overheads():
    rows = _terrain()
    assert [r["terrain"] for r in rows] == [0.0, 0.05, 0.1, 0.2, 0.4, 0.8] and [r["miss_share"] for r in rows] == [0.0, 4.0, 4.0, 18.0, 52.0, 92.0]
    assert [round(r["overhead_mean"], 2) for r in rows] == [0.0, 0.0, 0.0, 0.02, 0.11, 0.55] and round(rows[-1]["overhead_max"], 2) == 2.33 and rows[0]["overhead_max"] == 0.0
    big = ev.terrain_miss(replace(BASE, n=160), terrains=(0.0, 0.1, 0.4))
    assert [r["miss_share"] for r in big] == [0.0, 24.0, 86.0]


def test_chaining_purity_falls_with_the_number_of_clouds():
    rows = _chain()
    assert [r["clusters"] for r in rows] == [2, 3, 4, 6, 8] and [round(r["purity"], 2) for r in rows] == [1.0, 0.74, 0.77, 0.68, 0.64] and [r["perfect_share"] for r in rows] == [60.0, 34.0, 6.0, 0.0, 0.0]


def test_high_dimension_knn_is_exact_for_k_at_least_5_and_weakest_for_k_2_in_low_dimension():
    rows = {(r["d"], r["k"]): r for r in _hd()}
    for d in C.HIGHDIM_DIMS:
        for k in (5, 10, 20):
            assert rows[(d, k)]["coverage"] >= 0.995 and rows[(d, k)]["error"] <= 0.05 and rows[(d, k)]["connected_share"] >= 80.0
    assert rows[(2, 2)]["coverage"] == pytest.approx(0.93, abs=0.02) and rows[(2, 2)]["error"] == pytest.approx(1.0, abs=0.3) and rows[(2, 2)]["connected_share"] == 0.0
    assert rows[(50, 2)]["coverage"] == pytest.approx(0.98, abs=0.02) and rows[(50, 2)]["error"] <= 0.15 and rows[(50, 2)]["connected_share"] >= 80.0
    errors = [rows[(d, 2)]["error"] for d in (5, 10, 20, 50)]
    assert errors == sorted(errors, reverse=True) and rows[(3, 2)]["error"] == pytest.approx(1.6, abs=0.4) and max(rows[(2, 2)]["error"], rows[(3, 2)]["error"]) > 2 * errors[0]
    assert rows[(50, 2)]["coverage"] > rows[(2, 2)]["coverage"] and rows[(50, 2)]["error"] < rows[(2, 2)]["error"] / 5


def test_the_stated_ratio_between_the_ways_at_the_extremes():
    r = _cfg(n=300)
    assert round(r["ops_prim_array"] / r["ops_delaunay_prim"], 1) == 8.0 and round(r["ops_kruskal_complete"] / r["ops_delaunay_prim"]) == 41
    small = _cfg(n=20)
    assert small["ops_prim_array"] < small["ops_delaunay_prim"] < small["ops_delaunay_kruskal"] < small["ops_kruskal_complete"]


def test_default_instance_medians_over_five_instances():
    r = _cfg()
    assert (r["ops_prim_array"], r["ops_kruskal_complete"], r["ops_delaunay_kruskal"], r["ops_delaunay_prim"]) == (5251, 19206, 3253, 2591)


@pytest.mark.parametrize("n", [50, 400])
def test_knn_picture_is_the_same_for_other_sizes(n):
    rows = {(r["d"], r["k"]): r for r in ev.highdim(ds=(2, 50), ks=(2, 5), n=n)}
    assert rows[(2, 5)]["coverage"] >= 0.995 and rows[(50, 5)]["coverage"] == 1.0 and rows[(2, 5)]["error"] <= 0.05
    assert rows[(2, 2)]["connected_share"] <= 20.0 and rows[(2, 2)]["error"] > rows[(50, 2)]["error"] and rows[(2, 2)]["coverage"] < rows[(50, 2)]["coverage"] < 1.0
