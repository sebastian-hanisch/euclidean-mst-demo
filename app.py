"""Euklidischer MST – Delaunay statt n² Kanten - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der Spannbaum-Reihe der "Konzepte"-Reihe: sind die Kosten euklidisch, braucht man den vollständigen Graphen nicht - der minimale Spannbaum liegt schon in der Delaunay-Triangulierung
(MST ⊆ RNG ⊆ Gabriel ⊆ Delaunay, höchstens 3n - 3 - h Kanten). Gemessen werden der Kreuzungspunkt gegen den vollständigen Graphen, die Kantenzahlen der Hierarchie, der Bruch der Annahme mit
Geländezuschlag, die Brücke zu Single-Linkage und die kNN-Näherung in hohen Dimensionen.

Lauffähig mit: streamlit run app.py
"""

from dataclasses import replace

import streamlit as st

import emst_constants as C
import emst_methods as M
from emst_evaluation import (
    SWEEP_LABELS,
    WAYS,
    Settings,
    analyse,
    chaining,
    crossover,
    highdim,
    linkage_profile,
    sweep,
    terrain_miss,
)
from emst_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    store_from_widget,
    sync_query_params,
)
from emst_visualization import (
    WAY_COLORS,
    WAY_LABELS,
    build_clusters,
    build_crossover,
    build_delaunay_step,
    build_dendrogram,
    build_highdim,
    build_layers,
    build_ops_stacked,
    build_points,
    build_sweep,
    build_terrain,
)

st.set_page_config(page_title="Euklidischer MST – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _crossover(base):
    return crossover(base)


@st.cache_data(show_spinner=False)
def _terrain(base):
    return terrain_miss(base)


@st.cache_data(show_spinner=False)
def _chaining(base):
    return chaining(base)


@st.cache_data(show_spinner=False)
def _highdim():
    return highdim()


def de(number):
    return f"{number:,}".replace(",", ".")


SHORT = {"prim_array": "Prim-Array", "kruskal_complete": "Kruskal", "delaunay_kruskal": "Delaunay", "delaunay_prim": "Delaunay"}

st.title("🔺 Euklidischer MST – Delaunay statt n² Kanten")
st.markdown(
    """
**Viertes Stück der Spannbaum-Reihe.** Kruskal, Prim und Borůvka bekamen den Graphen als Kantenliste - bei dichten Graphen war genau sie das Problem (bei 160 Punkten 12 880 Kanten). Sind die Kosten aber
**euklidisch** (Kante = Abstand), braucht man den vollständigen Graphen nicht: der minimale Spannbaum liegt schon in der **Delaunay-Triangulierung** - dem Netz aus Dreiecken, deren Umkreis keinen anderen
Punkt enthält. Sie hat höchstens **3n − 3 − h** Kanten statt n(n−1)/2 (h = Punkte auf der konvexen Hülle). Genauer gilt **MST ⊆ RNG ⊆ Gabriel-Graph ⊆ Delaunay**.

Hier wird gemessen, **ab wann sich der Umweg lohnt** (die Triangulierung muss ja erst gebaut werden), was die drei Kandidatengraphen an Kanten sparen, **wann die Annahme "euklidisch" bricht**, was der MST mit
**Single-Linkage-Clustering** zu tun hat - und ob die Näherung über nächste Nachbarn in **hohen Dimensionen** trägt, wo keine Triangulierung mehr geht. Aufwand in Elementarschritten, nicht in Laufzeit.
"""
)
st.caption(
    "Setzt auf [kruskal-demo](https://github.com/sebastian-hanisch/kruskal-demo), [prim-demo](https://github.com/sebastian-hanisch/prim-demo) und [boruvka-demo](https://github.com/sebastian-hanisch/boruvka-demo) auf "
    "(Kruskal und Prim laufen als Vergleich mit; die Triangulierung ist hier von Hand gebaut, scipy nur in den Tests). Geplante Nachfolger (nicht gebaut): Gerichteter Spannbaum, Grad-/Hop-beschränkter und "
    "Kapazitierter MST, Steiner-Baum, Prize-Collecting Steiner-Baum, Sensitivität, zufällige Spannbäume."
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Delaunay bauen:** Punkte einzeln einfügen (in räumlicher Reihenfolge). Für jeden neuen Punkt das Dreieck suchen, das ihn enthält (Walk über Nachbardreiecke), alle Dreiecke bestimmen, deren Umkreis ihn enthält (der **Hohlraum**), und den Punkt mit dem Rand des Hohlraums neu verbinden.
2. **MST auf den Dreieckskanten:** Kruskal oder Prim auf den höchstens 3n − 3 − h Kanten statt auf allen n(n−1)/2 Paaren; derselbe Baum (Schlüssel: Länge, dann Kantenindex).
3. **Vergleich:** Prim mit Array auf dem impliziten vollständigen Graphen (n(n−1)/2 Abstände, keine Kantenliste) und Kruskal auf der vollständigen Liste.
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:4], preset_names[4:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.radio("Punkte", options=list(C.KINDS), format_func=lambda v: C.KIND_LABELS[v], key="kind_select",
                    help="Cluster: lange Kanten zwischen den Wolken. Gitter: ganzzahlige Koordinaten, viele gleiche Abstände und kozirkulare Punkte (die Triangulierung ist nicht eindeutig).")
    n = st.slider("Punkte n", *bounds("n_slider"), key="n_slider",
                  help="Ab n = 30 ist Delaunay + Prim im Median billiger als Prim mit Array auf dem vollständigen Graphen (darunter das Array); bei n = 300 8-fach.")
    clusters = (st.slider("Wolken", *bounds("clusters_slider"), value=int(st.session_state["clusters_slider"]), key="clusters_widget", on_change=store_from_widget, args=("clusters_slider",),
                          help="Zahl der Gauß-Wolken; je mehr, desto stärker überlappen sie.") if kind == "clusters" else C.DEFAULT_CLUSTERS)
    terrain = st.select_slider("Geländezuschlag", options=list(C.TERRAIN_OPTIONS), key="terrain_select", format_func=lambda v: "0 (rein euklidisch)" if v == 0 else f"{v:g}",
                               help="Kosten = Abstand mal Faktor in [1, 1 + Zuschlag]. Mit Zuschlag ist der MST nicht mehr garantiert in der Delaunay-Triangulierung (n = 60: bei 0.4 fehlt in 52 % der Instanzen eine Kante).")
    if kind != "grid" or terrain > 0:
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), value=int(st.session_state["seed_input"]), key="seed_widget", step=1, on_change=store_from_widget, args=("seed_input",))
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        seed = int(st.session_state["seed_input"])

sync_query_params({"kind_select": kind, "n_slider": int(n), "clusters_slider": int(clusters), "terrain_select": float(terrain), "seed_input": int(seed),
                   "layer_select": st.session_state["layer_select"], "cut_slider": int(st.session_state["cut_slider"])})

settings = Settings(kind, int(n), int(clusters), float(terrain), int(seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
inst, dt = a.inst, a.dt
n_pts = inst.n
pairs = n_pts * (n_pts - 1) // 2

# --- In Aktion ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Delaunay und MST in Aktion")
STEP_LABELS = {1: "1 · Punkte", 2: "2 · Delaunay wächst", 3: "3 · Kandidatenkanten und MST", 4: "4 · Single-Linkage-Cluster"}
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="emst_step", format_func=lambda s: STEP_LABELS[s])

if step == 1:
    st.markdown(f"**{n_pts} Punkte**; der vollständige Graph hätte **{de(pairs)} Kanten**, die Delaunay-Triangulierung hat **{len(dt.edges)}** (Hülle: {len(dt.hull)} Punkte).")
    st.plotly_chart(build_points(inst), width="stretch", key="s1_map")
elif step == 2:
    if "emst_insert" in st.session_state:
        st.session_state["emst_insert"] = min(max(1, int(st.session_state["emst_insert"])), n_pts)
    i_now = st.slider("Eingefügte Punkte", 1, max(2, n_pts), key="emst_insert", help="Nach so vielen eingefügten Punkten (in räumlicher Reihenfolge).")
    i_now = min(i_now, n_pts)
    p, removed, added = dt.events[i_now - 1]
    orient, walk, incirc = dt.per_insert[i_now - 1]
    st.markdown(f"**Punkt {i_now} von {n_pts}** (grüner Stern): die Suche nach seinem Dreieck brauchte {orient} Orientierungstests ({walk} Schritte über Nachbardreiecke), der **Hohlraum** (rot) hat {len(removed)} Dreieck(e) "
                f"({incirc} In-Circle-Tests), neu entstehen {len(added)} Dreiecke (orange Kanten).")
    st.plotly_chart(build_delaunay_step(inst, dt, i_now), width="stretch", key=f"s2_map_{i_now}")
    st.caption("Hellgraue Punkte sind noch nicht eingefügt. Dreiecke am Rand, die einen Eckpunkt des großen Hilfsdreiecks (außerhalb des Bildes) hätten, werden nicht gezeichnet.")
elif step == 3:
    layer = st.radio("Kandidatenkanten zeigen", options=list(C.LAYERS), format_func=lambda v: C.LAYER_LABELS[v], key="layer_widget", horizontal=True,
                     index=list(C.LAYERS).index(st.session_state["layer_select"]), on_change=store_from_widget, args=("layer_select",),
                     help="Jede Ebene enthält den MST: MST ⊆ RNG ⊆ Gabriel-Graph ⊆ Delaunay. Rot: eine Kante des echten MST liegt außerhalb der Triangulierung (nur mit Geländezuschlag).")
    st.plotly_chart(build_layers(inst, a.layers, layer, a.missing), width="stretch", key=f"s3_map_{layer}")
    lay = a.layers
    st.markdown(f"**Kanten:** vollständiger Graph {de(pairs)} → Delaunay **{len(lay['delaunay'])}** (Schranke 3n − 3 − h = {a.bound}) → Gabriel **{len(lay['gabriel'])}** → RNG **{len(lay['rng'])}** → MST **{len(lay['mst'])}**.")
    if a.missing:
        st.warning(f"Mit Geländezuschlag {terrain:g} liegen {len(a.missing)} Kante(n) des echten MST **nicht** in der Delaunay-Triangulierung (rot). Der MST auf den Dreieckskanten kostet {a.overhead:.2f} % mehr; "
                   "die vier Wege liefern nicht mehr denselben Baum.")
    else:
        st.markdown(f"**Derselbe Baum:** {'ja' if a.same_tree else 'NEIN (!)'} - alle vier Wege wählen dieselben {len(lay['mst'])} Kanten (Kosten {a.runs['kruskal_complete'].cost:.2f}).")
else:
    k_max = min(n_pts, C.CUT_MAX)
    if "cut_slider" in st.session_state:
        st.session_state["cut_slider"] = min(max(1, int(st.session_state["cut_slider"])), k_max)
    k_cut = st.slider("Cluster k", 1, max(2, k_max), value=int(st.session_state["cut_slider"]), key="cut_widget", on_change=store_from_widget, args=("cut_slider",),
                      help="k Cluster = MST ohne die k − 1 teuersten Kanten (Single-Linkage).")
    k_cut = min(k_cut, k_max)
    prof = linkage_profile(a)
    labels = M.clusters_from_tree(n_pts, prof["tree"], k_cut)
    st.markdown(f"**{k_cut} Cluster:** die {k_cut - 1} teuersten MST-Kanten (rot gestrichelt) sind abgeschnitten. Der MST ist das **Single-Linkage-Dendrogramm** - dasselbe Ergebnis wie das agglomerative Verfahren "
                "mit Single-Linkage, ohne dass man alle Paarabstände verwaltet (siehe [agglomerative-demo](https://github.com/sebastian-hanisch/agglomerative-demo)).")
    if inst.truth is not None:
        pur = linkage_profile(a)["purity"]
        st.markdown(f"Bei den wahren {inst.clusters} Wolken (k = {inst.clusters}) ordnet der Schnitt {round(pur * n_pts)} von {n_pts} Punkten der richtigen Wolke zu (Reinheit {pur:.2f}); je mehr Wolken sich überlappen, desto mehr verkettet Single-Linkage sie (Chaining).")
    st.plotly_chart(build_clusters(inst, [(u, v) for u, v, _w in prof["tree"]], labels), width="stretch", key=f"s4_map_{k_cut}")
    st.plotly_chart(build_dendrogram(prof["heights"], k_cut), width="stretch", key="dendrogram")
    st.caption("Verschmelzungshöhen des Dendrogramms (Länge der MST-Kanten, teuerste zuerst; logarithmisch): ein großer Sprung nach den k − 1 hervorgehobenen Kanten zeigt eine natürliche Clusterzahl.")

st.markdown("---")

# --- Aufwand -----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ⚙️ Was kostet welcher Weg?")
st.caption(
    "**Elementarschritte:** ausgewertete Abstände (je 1), Vergleiche und Heap-Operationen der MST-Verfahren, beim Triangulieren Orientierungs- und In-Circle-Tests, neue Dreiecke und die Vergleiche der räumlichen "
    "Sortierung. Ein Näherungsmaß für den Vergleich, **keine Laufzeitmessung**: optimierte Bibliotheken (Qhull, CGAL) sind schneller als diese Python-Umsetzung, und was ein Schritt kostet, steckt nicht darin."
)
ops = a.ops
best = ops[a.winner]
r_prim = ops["prim_array"] / ops["delaunay_prim"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Gewinner", SHORT[a.winner], delta=f"{de(best)} Schritte", delta_color="off")
m2.metric("Delaunay + Prim", de(ops["delaunay_prim"]), delta=f"{r_prim:.2f}x vs. Array", delta_color="off")
m3.metric("Kanten", str(len(dt.edges)), delta=f"statt {de(pairs)}", delta_color="off")
m4.metric("Aufbau je Punkt", f"{a.per_point['build']:.1f}", delta="Schritte", delta_color="off")
st.caption(f"Gewinner: {WAY_LABELS[a.winner]}. Verhältnis Prim-Array / (Delaunay + Prim) = {r_prim:.2f} (über 1: die Triangulierung lohnt sich); Kruskal vollständig / (Delaunay + Kruskal) = {ops['kruskal_complete'] / ops['delaunay_kruskal']:.2f}.")
st.plotly_chart(build_ops_stacked(a.runs, a.winner), width="stretch", key="ops_bars")
pp = a.per_point
st.caption(f"Aufbau der Triangulierung: {dt.orient_tests} Orientierungstests ({pp['orient']:.1f} je Punkt), {dt.incircle_tests} In-Circle-Tests ({pp['incircle']:.1f} je Punkt), {dt.created} neue Dreiecke, {dt.sort_comparisons} "
           f"Sortier-Vergleiche; mittlerer Hohlraum {pp['cavity']:.1f} Dreiecke. Die Kosten wachsen nur langsam mit n (n = 10 bis 300: 19 bis 35 Schritte je Punkt).")

st.markdown("---")

# --- Experimente auf Abruf ---------------------------------------------------------------------------------------------------------------------

st.subheader("🔀 Wo liegt der Kreuzungspunkt?")
st.caption("Ab welchem n braucht Delaunay + Prim im Median (5 feste Instanzen) weniger Schritte als Prim mit Array auf dem impliziten vollständigen Graphen? Die anderen Einstellungen der Seitenleiste gelten.")
base = replace(settings, seed=0)
if st.button("Kreuzungspunkt suchen (n = 10 bis 60)", key="cross_start"):
    st.session_state["cross_done"] = st.session_state.get("cross_done", set()) | {base}
if base in st.session_state.get("cross_done", set()):
    with st.spinner("Rechne..."):
        n_star, rows_c = _crossover(base)
    st.plotly_chart(build_crossover(rows_c, n_star), width="stretch", key="crossover")
    st.caption(f"Kreuzungspunkt (Schrittweite 5): **n = {n_star}**." if n_star is not None else "Im getesteten Bereich (n = 10 bis 60) gewinnt Delaunay + Prim nicht durchgehend.")
st.markdown("---")

st.subheader("🌄 Wann bricht die Annahme \"euklidisch\"?")
st.caption("Kosten = Abstand mal Geländefaktor in [1, 1 + Zuschlag] (wie in den Vorgänger-Demos): 50 Instanzen je Zuschlag, echter MST gegen die Delaunay-Triangulierung.")
if st.button("Gelände-Experiment über 50 Instanzen", key="terrain_start"):
    st.session_state["terrain_done"] = st.session_state.get("terrain_done", set()) | {replace(base, terrain=0.0)}
if replace(base, terrain=0.0) in st.session_state.get("terrain_done", set()):
    with st.spinner("Rechne..."):
        rows_t = _terrain(replace(base, terrain=0.0))
    st.plotly_chart(build_terrain(rows_t), width="stretch", key="terrain")
    st.caption("Links: Anteil der Instanzen, in deren echtem MST mindestens eine Kante nicht in der Triangulierung liegt; rechts: Kostenaufschlag, wenn man nur auf den Dreieckskanten arbeitet. Die Annahme bricht früh, der Schaden bleibt klein.")
st.markdown("---")

st.subheader("🔗 Chaining: wie gut findet Single-Linkage die Wolken?")
st.caption("Cluster-Instanzen mit 2 bis 8 Wolken (50 Instanzen je Zahl): Schnitt des MST in so viele Cluster wie Wolken, Reinheit gegen die wahren Wolken.")
if st.button("Chaining-Experiment über 50 Instanzen", key="chain_start"):
    st.session_state["chain_done"] = st.session_state.get("chain_done", set()) | {replace(base, kind="clusters", terrain=0.0)}
if replace(base, kind="clusters", terrain=0.0) in st.session_state.get("chain_done", set()):
    with st.spinner("Rechne..."):
        rows_ch = _chaining(replace(base, kind="clusters", terrain=0.0))
    st.plotly_chart(build_sweep(rows_ch, "Zahl der Wolken", [("purity", "Reinheit (Median)", "#2F6B65")], "Reinheit", key="clusters"), width="stretch", key="chaining")
    st.caption("Median über 50 Instanzen (n wie in der Seitenleiste), Band = 10. bis 90. Perzentil. Reinheit 1 = jede Wolke exakt getroffen: " + ", ".join(f"{r['clusters']} Wolken: {r['perfect_share']:.0f} %" for r in rows_ch) + " der Instanzen.")
st.markdown("---")

st.subheader("🧭 Hohe Dimensionen: die Näherung über nächste Nachbarn")
st.caption("Keine Triangulierung mehr (ihre Größe wächst mit der Dimension). Statt dessen: MST auf dem kNN-Graphen, unverbundene Komponenten über die billigste echte Verbindung verbinden. "
           f"{C.HIGHDIM_N} gleichverteilte Punkte im Einheitswürfel, 5 feste Instanzen.")
if st.button("Näherung für d = 2 bis 50 berechnen", key="highdim_start"):
    st.session_state["highdim_done"] = True
if st.session_state.get("highdim_done"):
    with st.spinner("Rechne..."):
        rows_h = _highdim()
    h1, h2 = st.columns(2)
    with h1:
        st.plotly_chart(build_highdim(rows_h, C.HIGHDIM_DIMS, C.HIGHDIM_KS, "coverage", "Anteil der MST-Kanten im kNN-Graph"), width="stretch", key="hd_cov")
    with h2:
        st.plotly_chart(build_highdim(rows_h, C.HIGHDIM_DIMS, C.HIGHDIM_KS, "error", "Kostenfehler (%)"), width="stretch", key="hd_err")
    st.plotly_chart(build_highdim(rows_h, C.HIGHDIM_DIMS, C.HIGHDIM_KS, "connected_share", "kNN-Graph zusammenhängend (%)"), width="stretch", key="hd_conn")
    st.caption("Median über 5 Instanzen. Mit k >= 5 enthält der kNN-Graph in jeder getesteten Dimension alle MST-Kanten; die schwächsten Werte hat k = 2 in NIEDRIGER Dimension (kNN-Graph oft unverbunden). "
               "Nur für gleichverteilte Punkte gemessen - echte Daten mit Struktur können sich anders verhalten, und der kNN-Graph wird hier per Brute Force berechnet.")
st.markdown("---")

st.subheader("📐 Sweeps")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda v: SWEEP_LABELS[v], key="sweep_select")
metric = st.radio("Kennzahl", options=["ops", "ratio", "edges", "build", "miss"],
                  format_func=lambda v: {"ops": "Elementarschritte", "ratio": "Verhältnisse", "edges": "Kanten", "build": "Aufbau je Punkt", "miss": "Fehlende MST-Kanten"}[v], key="sweep_metric", horizontal=True)
if st.button("Sweep über 5 feste Instanzen berechnen (kann einige Sekunden dauern)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base)}
if (sweep_param, base) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Instanzen..."):
        rows_s = _sweep(sweep_param, base)
    label = SWEEP_LABELS[sweep_param]
    if metric == "ops":
        st.plotly_chart(build_sweep(rows_s, label, [(f"ops_{w}", WAY_LABELS[w], WAY_COLORS[w]) for w in WAYS], "Elementarschritte (logarithmisch)", log_y=True), width="stretch", key="sweep_ops")
    elif metric == "ratio":
        st.plotly_chart(build_sweep(rows_s, label, [("speedup_prim", "Prim-Array / (Delaunay + Prim)", WAY_COLORS["delaunay_prim"]), ("speedup_kruskal", "Kruskal vollständig / (Delaunay + Kruskal)", WAY_COLORS["delaunay_kruskal"])],
                                    "Verhältnis (logarithmisch)", log_y=True, ref_line=1.0, ref_label="gleich viele Schritte"), width="stretch", key="sweep_ratio")
    elif metric == "edges":
        st.plotly_chart(build_sweep(rows_s, label, [("pairs", "alle Paare", "#7b3fbf"), ("dt_edges", "Delaunay", "#e8a13a"), ("gabriel_edges", "Gabriel", "#54a24b"), ("rng_edges", "RNG", "#4c78a8"), ("mst_edges", "MST", "#2F6B65")],
                                    "Kanten (logarithmisch)", log_y=True), width="stretch", key="sweep_edges")
    elif metric == "build":
        st.plotly_chart(build_sweep(rows_s, label, [("build_per_point", "Aufbau gesamt", "#e8a13a"), ("orient_per_point", "Orientierungstests", "#4c78a8"), ("incircle_per_point", "In-Circle-Tests", "#54a24b"), ("cavity", "Hohlraum (Dreiecke)", "#d62728")],
                                    "je Punkt"), width="stretch", key="sweep_build")
    else:
        st.plotly_chart(build_sweep(rows_s, label, [("miss_share", "Instanzen mit fehlender MST-Kante (%)", "#d62728"), ("overhead", "Kostenaufschlag (%)", "#2F6B65")], "Prozent"), width="stretch", key="sweep_miss")
    st.caption("Median über 5 feste Instanzen (Seeds 100000–100004), Band = 10. bis 90. Perzentil. Die übrigen Regler stehen wie in der Seitenleiste.")
st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Triangulierung spart immer** | Erst ab n = 30 (Prim mit Array ist darunter billiger: n = 20: 551 gegen 685 Schritte); der Aufbau kostet 19 bis 35 Schritte je Punkt. Bei n = 300 ist Delaunay + Prim 8-fach billiger als das Array, Kruskal auf der vollständigen Liste 41-fach teurer. | - |
| **Elementarschritte sind Laufzeit** | Nein. Optimierte Bibliotheken (Qhull, CGAL) bauen die Triangulierung viel schneller als diese Python-Umsetzung; der Vergleich zählt Schritte einheitlich, nicht Sekunden. | Laufzeitmessung mit einer echten Bibliothek (nicht gebaut) |
| **Kosten sind euklidisch** | Mit Geländezuschlag liegen MST-Kanten außerhalb der Triangulierung (n = 60, Zuschlag 0.4: in 52 % der Instanzen mindestens eine; Kostenaufschlag im Mittel 0.11 %, im schlimmsten Fall bei Zuschlag 0.8 2.33 %). | Nicht-euklidische Kandidatengraphen (nicht gebaut) |
| **Gleiche Abstände sind harmlos** | Auf dem Gitter ist die Triangulierung nicht eindeutig (kozirkulare Punkte), Gabriel-Graph und Delaunay fast gleich; der Schlüssel (Kosten, Kantenindex) liefert trotzdem in allen Wegen denselben Baum. | - |
| **Das geht auch in hohen Dimensionen** | Keine Triangulierung: ihre Größe wächst mit der Dimension (Theorie, hier nicht gemessen). Die kNN-Näherung ist bei gleichverteilten Punkten ab k = 5 in allen getesteten Dimensionen exakt; für echte Daten mit Struktur ist das nicht gemessen. | FAMST und Verwandte (nicht gebaut) |
| **Single-Linkage findet die Wolken** | Nur bei getrennten Wolken (Reinheit 1.00 bei 2 Wolken, 0.64 bei 8); überlappende Wolken und Ausreißer verketten (Chaining). | Andere Linkage-Verfahren: agglomerative-demo |
| **Punkte in allgemeiner Lage** | Doppelte Punkte werden abgelehnt; alle Punkte auf einer Geraden ergeben keinen Dreiecksgraphen (der Pfad ist der MST). Extrem flache Punktwolken (Seitenverhältnis bis 10⁶ getestet) verlangen ein sehr großes Hilfsdreieck. | Symbolische Punkte im Unendlichen (nicht gebaut) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Delaunay-Eigenschaft.** Ein Dreieck $abc$ gehört zur Triangulierung, wenn sein Umkreis keinen anderen Punkt enthält; der In-Circle-Test ist das Vorzeichen der Determinante
$\begin{vmatrix} a_x-p_x & a_y-p_y & (a_x-p_x)^2+(a_y-p_y)^2 \\ b_x-p_x & b_y-p_y & (b_x-p_x)^2+(b_y-p_y)^2 \\ c_x-p_x & c_y-p_y & (c_x-p_x)^2+(c_y-p_y)^2 \end{vmatrix}$.
Nach dem Satz von Euler hat die Triangulierung $2n - 2 - h$ Dreiecke und $3n - 3 - h$ Kanten ($h$ Hüllpunkte), also weniger als $3n$.

**Warum der MST darin liegt.** Eine Kante $pq$ des MST hat keinen Punkt $r$ mit $\max(d(p,r), d(q,r)) < d(p,q)$ (sonst ließe sie sich tauschen): sie gehört zum **relativen Nachbarschaftsgraphen** (RNG). Ist die Linse
leer, dann auch der Kreis mit Durchmesser $pq$: **Gabriel-Graph**. Jede Gabriel-Kante liegt in der Delaunay-Triangulierung. Also $\text{MST} \subseteq \text{RNG} \subseteq \text{Gabriel} \subseteq \text{Delaunay}$.

**Single-Linkage.** Der Abstand zweier Cluster ist der kleinste Abstand zweier Punkte aus beiden. Kruskal auf dem vollständigen Graphen verschmilzt genau in dieser Reihenfolge: die Verschmelzungshöhen des
Dendrogramms sind die sortierten MST-Kantenlängen, $k$ Cluster = MST ohne die $k - 1$ längsten Kanten.

**Aufwand.** Inkrementelles Einfügen mit Punktsuche per Walk und Hohlraum über Nachbardreiecke: bei räumlich sortierten Punkten im Mittel wenige Tests je Punkt (hier gemessen 19 bis 35 Schritte je Punkt). Danach
$O(m \log m)$ für Kruskal bzw. $O(m \log n)$ für Prim mit Heap auf $m \le 3n$ Kanten, insgesamt etwa $O(n \log n)$; der vollständige Graph braucht $O(n^2)$.

**Literatur.** Shamos, M. I., & Hoey, D. (1975). *Closest-point problems.* 16th Annual Symposium on Foundations of Computer Science, 151-162. Bowyer, A. (1981). *Computing Dirichlet tessellations.* The Computer Journal
24(2), 162-166. Watson, D. F. (1981). *Computing the n-dimensional Delaunay tessellation with application to Voronoi polytopes.* The Computer Journal 24(2), 167-172. Gabriel, K. R., & Sokal, R. R. (1969). *A new
statistical approach to geographic variation analysis.* Systematic Zoology 18(3), 259-278. Toussaint, G. T. (1980). *The relative neighbourhood graph of a finite planar set.* Pattern Recognition 12(4), 261-268.

Implementiert in `emst_delaunay.py` (Bowyer-Watson von Hand), `emst_methods.py` (die vier Wege, Gabriel/RNG, Single-Linkage, kNN), `emst_scenario.py` (Instanzen), `emst_evaluation.py` (Kennzahlen, Sweeps, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
