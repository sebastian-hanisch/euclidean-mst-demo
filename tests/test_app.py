"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt und jeder Einfüge-Schritt, alle Instanztypen und Ebenen, Randwerte, Würfel-Knopf, Permalink-Grenzen, Instanzwechsel, Experimente und
Sweeps auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import emst_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=1, **state):
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if step != 1:
        at.select_slider(key="emst_step").set_value(step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m for m in at.metric if m.label.startswith(label))


def test_default_run_has_no_exception_and_shows_the_four_metrics():
    at = _run()
    _ok(at)
    assert {"Gewinner", "Delaunay + Prim", "Kanten", "Aufbau je Punkt"} <= {m.label for m in at.metric}
    assert _metric(at, "Gewinner").value == "Delaunay" and _metric(at, "Delaunay + Prim").value == "2.634" and _metric(at, "Delaunay + Prim").delta == "1.99x vs. Array"
    assert _metric(at, "Kanten").value == "165" and _metric(at, "Kanten").delta == "statt 1.770"
    assert any("Gewinner: Delaunay + Prim" in c.value for c in at.caption)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["kind_select"] == p["kind"] and at.session_state["n_slider"] == p["n"] and at.session_state["terrain_select"] == p["terrain"]
    assert at.session_state["layer_select"] == p["layer"] and at.session_state["cut_slider"] == p["cut"] and at.metric


@pytest.mark.parametrize("step", [1, 2, 3, 4])
def test_every_step_runs_for_every_kind(step):
    for kind in C.KINDS:
        at = _run(kind_select=kind, n_slider=40, step=step)
        _ok(at)
        assert at.get("plotly_chart") and at.session_state["emst_step"] == step


def test_insert_slider_walks_through_the_points_and_survives_an_instance_change():
    at = _run(step=2, n_slider=25)
    _ok(at)
    slider = at.slider(key="emst_insert")
    assert slider.max == 25
    for i in (1, 2, 13, 25):
        at.slider(key="emst_insert").set_value(i).run()
        _ok(at)
        assert any(m.value.startswith(f"**Punkt {i} von 25**") for m in at.markdown)
    at.session_state["n_slider"] = C.N_MIN
    at.run()
    _ok(at)
    assert at.session_state["emst_insert"] <= C.N_MIN


@pytest.mark.parametrize("layer", C.LAYERS)
def test_every_layer_runs_and_states_the_edge_counts(layer):
    at = _run(step=3, layer_select=layer)
    _ok(at)
    assert any("Delaunay **165**" in m.value and "→ MST **59**" in m.value for m in at.markdown) and any("Derselbe Baum:" in m.value for m in at.markdown)


def test_terrain_shows_the_missing_edge_warning_and_no_same_tree_claim():
    at = _run(step=3, terrain_select=0.4)
    _ok(at)
    assert any("nicht** in der Delaunay-Triangulierung" in w.value for w in at.warning) and not any("Derselbe Baum:" in m.value for m in at.markdown)


def test_cut_slider_extremes_and_purity_text_for_clusters():
    at = _run(step=4, kind_select="clusters", n_slider=60, clusters_slider=3)
    _ok(at)
    for k in (1, 2, 3, 30):
        at.slider(key="cut_widget").set_value(k).run()
        _ok(at)
        assert any(m.value.startswith(f"**{k} Cluster:**") for m in at.markdown)
    assert any("ordnet der Schnitt" in m.value for m in at.markdown)
    small = _run(step=4, n_slider=5, cut_slider=30)
    _ok(small)
    assert small.session_state["cut_slider"] <= 5 and not any("ordnet der Schnitt" in m.value for m in small.markdown)


@pytest.mark.parametrize("kw", [
    dict(n_slider=C.N_MIN), dict(n_slider=C.N_MAX), dict(kind_select="grid", n_slider=C.N_MIN), dict(kind_select="grid", n_slider=C.N_MAX), dict(kind_select="clusters", clusters_slider=C.CLUSTERS_MIN),
    dict(kind_select="clusters", clusters_slider=C.CLUSTERS_MAX), dict(terrain_select=C.TERRAIN_OPTIONS[-1]), dict(kind_select="grid", terrain_select=0.6), dict(n_slider=C.N_MIN, terrain_select=1.0, kind_select="clusters"),
])
def test_extreme_settings_run(kw):
    for step in (1, 2, 3, 4):
        _ok(_run(step=step, **kw))


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(n="9999", clusters="1", terrain="0.35", layer="voronoi", kind="nope", cut="0").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["n_slider"], ss["clusters_slider"], ss["cut_slider"]) == (C.N_MAX, C.CLUSTERS_MIN, 1)
    assert (ss["terrain_select"], ss["layer_select"], ss["kind_select"]) == (C.DEFAULT_TERRAIN, C.DEFAULT_LAYER, "uniform")


def test_permalink_accepts_valid_values():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(kind="clusters", n="80", clusters="5", terrain="0.2", layer="rng", cut="7", seed="9").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["kind_select"], ss["n_slider"], ss["clusters_slider"], ss["terrain_select"], ss["layer_select"], ss["cut_slider"], ss["seed_input"]) == ("clusters", 80, 5, 0.2, "rng", 7, 9)


def test_sidebar_shows_only_the_controls_that_matter():
    plain = _run()
    assert not any(s.key == "clusters_widget" for s in plain.slider) and any(n.key == "seed_widget" for n in plain.number_input)
    clusters = _run(kind_select="clusters")
    assert any(s.key == "clusters_widget" for s in clusters.slider)
    grid = _run(kind_select="grid")
    assert not any(n.key == "seed_widget" for n in grid.number_input) and not any(s.key == "clusters_widget" for s in grid.slider)
    rough = _run(kind_select="grid", terrain_select=0.4)
    assert any(n.key == "seed_widget" for n in rough.number_input)


def test_changing_the_instance_while_on_step_two_does_not_crash():
    at = _run(step=2)
    _ok(at)
    at.session_state["n_slider"] = C.N_MIN
    at.run()
    _ok(at)
    at.session_state["kind_select"] = "grid"
    at.run()
    _ok(at)


@pytest.mark.parametrize("param", ["n", "terrain"])
@pytest.mark.parametrize("metric", ["ops", "ratio", "edges", "build", "miss"])
def test_sweeps_run_on_demand_for_every_metric(param, metric):
    at = _run(n_slider=15, sweep_metric=metric)
    at.selectbox(key="sweep_select").set_value(param).run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_crossover_experiment_runs_on_demand():
    at = _run()
    next(b for b in at.button if b.key == "cross_start").click().run()
    _ok(at)
    assert any("Kreuzungspunkt (Schrittweite 5): **n = 30**" in c.value for c in at.caption)


def test_terrain_experiment_runs_on_demand():
    at = _run(n_slider=20)
    next(b for b in at.button if b.key == "terrain_start").click().run()
    _ok(at)
    assert any("Die Annahme bricht früh" in c.value for c in at.caption) and len(at.get("plotly_chart")) >= 2


def test_chaining_experiment_runs_on_demand():
    at = _run(n_slider=30)
    next(b for b in at.button if b.key == "chain_start").click().run()
    _ok(at)
    assert any("Reinheit 1 = jede Wolke exakt getroffen" in c.value for c in at.caption)


def test_high_dimension_experiment_runs_on_demand():
    at = _run()
    next(b for b in at.button if b.key == "highdim_start").click().run()
    _ok(at)
    assert any("Mit k >= 5 enthält der kNN-Graph" in c.value for c in at.caption) and len(at.get("plotly_chart")) >= 4


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Shamos, M. I., & Hoey, D. (1975)" in m.value and "Bowyer, A. (1981)" in m.value for m in at.markdown)
