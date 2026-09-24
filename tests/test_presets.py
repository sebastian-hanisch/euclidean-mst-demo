"""Presets: Vollständigkeit, gültige Werte, der Median der Elementarschritte von "Delaunay + Prim" bleibt in der gemessenen Spannweite, und jedes Preset zeigt, was sein Name und sein Hilfetext sagen."""

import pytest

import emst_constants as C
import emst_evaluation as ev
import emst_presets as P


def _settings(p):
    return ev.Settings(kind=p["kind"], n=p["n"], clusters=p["clusters"], terrain=p["terrain"], seed=p["seed"])


def _analyse(name):
    return ev.analyse(_settings(C.PRESETS[name]))


def test_every_preset_has_help_and_a_band():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS) and len(C.PRESETS) == 8
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_members_of_the_controls():
    for p in C.PRESETS.values():
        assert p["kind"] in C.KINDS and p["terrain"] in C.TERRAIN_OPTIONS and p["layer"] in C.LAYERS
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.CLUSTERS_MIN <= p["clusters"] <= C.CLUSTERS_MAX and 0 <= p["seed"] <= C.SEED_MAX and 1 <= p["cut"] <= min(C.CUT_MAX, p["n"])


def test_default_preset_equals_the_default_settings_and_the_defaults_of_the_specs():
    p = C.PRESETS["Standardfall (Voreinstellung)"]
    assert _settings(p) == ev.Settings() and p["layer"] == C.DEFAULT_LAYER and p["cut"] == C.DEFAULT_CUT
    assert all(P.SETTING_SPECS[k].default == v for k, v in (("layer_select", p["layer"]), ("cut_slider", p["cut"]), ("terrain_select", p["terrain"]), ("kind_select", p["kind"])))


@pytest.mark.parametrize("name", list(C.PRESET_EXPECTED_BANDS))
def test_preset_median_ops_of_delaunay_prim_stay_in_the_measured_band(name):
    lo, hi = C.PRESET_EXPECTED_BANDS[name]
    assert lo <= ev.run_config(_settings(C.PRESETS[name]))["ops_delaunay_prim"] <= hi


def test_standard_preset_numbers():
    a = _analyse("Standardfall (Voreinstellung)")
    assert a.ops == {"prim_array": 5251, "kruskal_complete": 19238, "delaunay_kruskal": 3320, "delaunay_prim": 2634} and a.winner == "delaunay_prim" and a.same_tree
    assert len(a.dt.edges) == 165 and a.inst.n * (a.inst.n - 1) // 2 == 1770 and a.runs["kruskal_complete"].cost == pytest.approx(538.18, abs=0.01)


def test_small_preset_array_still_wins():
    a = _analyse("Kleine Instanz (n = 20)")
    assert a.ops == {"prim_array": 551, "kruskal_complete": 1578, "delaunay_kruskal": 830, "delaunay_prim": 685} and a.winner == "prim_array"


def test_big_preset_numbers():
    a = _analyse("Große Instanz (n = 300)")
    assert a.ops == {"prim_array": 134251, "kruskal_complete": 684020, "delaunay_kruskal": 21687, "delaunay_prim": 17130} and round(a.per_point["build"], 1) == 35.9
    assert round(a.ops["prim_array"] / a.ops["delaunay_prim"], 1) == 7.8 and round(a.ops["kruskal_complete"] / a.ops["delaunay_prim"]) == 40


def test_cluster_preset_numbers_and_purity():
    a = _analyse("Cluster (n = 120)")
    assert a.ops == {"prim_array": 21301, "kruskal_complete": 100976, "delaunay_kruskal": 7685, "delaunay_prim": 5709} and a.same_tree
    assert round(ev.linkage_profile(a)["purity"] * 120) == 87


def test_grid_preset_numbers_and_ties():
    a = _analyse("Gitter (n = 100)")
    assert a.ops == {"prim_array": 14751, "kruskal_complete": 57977, "delaunay_kruskal": 5392, "delaunay_prim": 4186} and a.same_tree
    assert (len(a.dt.edges), len(a.gabriel)) == (261, 261) and a.runs["kruskal_complete"].cost == pytest.approx(990.0)


def test_terrain_preset_has_a_missing_edge_and_a_small_overhead():
    a = _analyse("Geländezuschlag 0,4")
    assert a.missing == [(20, 56)] and not a.same_tree and a.overhead == pytest.approx(0.185, abs=0.001)
    assert a.runs["kruskal_complete"].cost == pytest.approx(632.77, abs=0.01) and a.runs["delaunay_kruskal"].cost == pytest.approx(633.94, abs=0.01)
    assert ev.run_config(_settings(C.PRESETS["Geländezuschlag 0,4"]))["miss_share"] == 40.0


def test_chaining_preset_cut_and_purity():
    p = C.PRESETS["Single-Linkage-Chaining"]
    a = _analyse("Single-Linkage-Chaining")
    assert p["cut"] == p["clusters"] == 6 and p["layer"] == "mst" and round(ev.linkage_profile(a)["purity"] * 120) == 65


def test_candidate_graph_preset_numbers():
    a = _analyse("Kandidatengraphen (n = 150)")
    assert (len(a.dt.edges), len(a.gabriel), len(a.rng), len(a.true_tree), len(a.dt.hull)) == (436, 262, 176, 149, 11) and 150 * 149 // 2 == 11175 and a.bound == 436


def test_bounds_and_permalink_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("cut_slider") == (1, C.CUT_MAX)
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_permalink_casters_accept_members_and_reject_everything_else():
    c = P.SETTING_SPECS
    assert c["kind_select"].caster("grid") == "grid" and c["layer_select"].caster("rng") == "rng" and c["terrain_select"].caster("0.4") == 0.4
    for key, bad in (("kind_select", "x"), ("terrain_select", "0.35"), ("layer_select", "voronoi")):
        with pytest.raises(ValueError):
            c[key].caster(bad)
