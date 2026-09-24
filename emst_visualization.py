"""Plotly-Abbildungen: Punkte, Delaunay wächst (Hohlraum und neue Kanten), Kandidatengraph-Ebenen über dem MST, Single-Linkage-Cluster, Aufwand gestapelt, Sweeps, Kreuzungspunkt, Gelände- und
Hochdimensions-Experimente. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import emst_delaunay as D

MST_COLOR = "#2F6B65"
NEW_COLOR = "#ff7f0e"
MISS_COLOR = "#d62728"
GREY = "rgba(150,150,150,0.35)"
WAY_COLORS = {"prim_array": "#4c78a8", "kruskal_complete": "#7b3fbf", "delaunay_kruskal": "#e8a13a", "delaunay_prim": "#54a24b"}
WAY_LABELS = {"prim_array": "Prim (Array, implizit)", "kruskal_complete": "Kruskal (vollständig)", "delaunay_kruskal": "Delaunay + Kruskal", "delaunay_prim": "Delaunay + Prim"}
PART_COLORS = {"build": "#e8a13a", "dist": "#b0b0b0", "mst": "#4c78a8"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.1):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_axes(fig, height=440):
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def _segments(xy, pairs):
    xs, ys = [], []
    for u, v in pairs:
        xs += [xy[u][0], xy[v][0], None]
        ys += [xy[u][1], xy[v][1], None]
    return xs, ys


def _lines(fig, xy, pairs, color, width, dash="solid", name="", showlegend=False):
    if len(pairs):
        xs, ys = _segments(xy, pairs)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash), name=name, hoverinfo="skip", showlegend=showlegend))


def _dots(fig, xy, idx, color, size=8, name="", showlegend=False, symbol="circle"):
    idx = list(idx)
    if idx:
        fig.add_trace(go.Scatter(x=[xy[i][0] for i in idx], y=[xy[i][1] for i in idx], mode="markers", marker=dict(size=size, color=color, symbol=symbol, line=dict(width=1, color="white")),
                                 name=name, hoverinfo="skip", showlegend=showlegend))


def _tri_pairs(tris, n):
    out = set()
    for a, b, c in tris:
        if max(a, b, c) < n:
            for u, v in ((a, b), (b, c), (c, a)):
                out.add((min(u, v), max(u, v)))
    return sorted(out)


def build_points(inst):
    fig = go.Figure()
    _dots(fig, inst.xy, range(inst.n), "#4c78a8", 8)
    return _map_axes(fig)


def build_delaunay_step(inst, dt, i):
    """Zustand nach den ersten i Einfügungen: Dreiecke grau, der zuletzt eingefügte Punkt grün, sein Hohlraum (entfernte Dreiecke) rot gefüllt, die neuen Kanten orange, noch nicht eingefügte Punkte hell."""
    n = inst.n
    i = max(1, min(i, n))
    xy = inst.xy
    tris = D.replay(dt, i)
    p, removed, added = dt.events[i - 1]
    fig = go.Figure()
    xs, ys = [], []
    for a, b, c in removed:
        if max(a, b, c) < n:
            xs += [xy[a][0], xy[b][0], xy[c][0], xy[a][0], None]
            ys += [xy[a][1], xy[b][1], xy[c][1], xy[a][1], None]
    if xs:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor="rgba(214,39,40,0.18)", line=dict(width=0), hoverinfo="skip", showlegend=False))
    _lines(fig, xy, _tri_pairs(tris, n), GREY, 1.4)
    new_pairs = _tri_pairs(added, n)
    _lines(fig, xy, new_pairs, NEW_COLOR, 3)
    inserted = dt.order[:i]
    _dots(fig, xy, dt.order[i:], "rgba(150,150,150,0.5)", 6)
    _dots(fig, xy, inserted[:-1], "#4c78a8", 8)
    _dots(fig, xy, [p], "#2ca02c", 14, symbol="star")
    return _map_axes(fig)


def build_layers(inst, layers, layer, missing):
    """Kandidatenkanten der gewählten Ebene grau, darüber der MST; MST-Kanten außerhalb der Delaunay-Triangulierung (nur mit Geländezuschlag) rot."""
    xy = inst.xy
    fig = go.Figure()
    if layer != "mst":
        _lines(fig, xy, layers[layer], GREY, 1.4, name=f"{len(layers[layer])} Kandidatenkanten", showlegend=True)
    miss = set(missing)
    _lines(fig, xy, [e for e in layers["mst"] if e not in miss], MST_COLOR, 3.5, name="MST", showlegend=True)
    _lines(fig, xy, sorted(miss), MISS_COLOR, 4.5, name="MST-Kante außerhalb von Delaunay", showlegend=True)
    _dots(fig, xy, range(inst.n), "#4c78a8", 7)
    return _map_axes(fig)


def _palette(labels):
    ids, out = {}, []
    for lab in labels:
        if lab not in ids:
            ids[lab] = f"hsl({(len(ids) * 137) % 360},62%,45%)"
        out.append(ids[lab])
    return out


def build_clusters(inst, tree, labels):
    """Punktfarbe = Cluster; MST-Kanten innerhalb eines Clusters dunkel, die abgeschnittenen (zwischen Clustern) rot gestrichelt."""
    xy = inst.xy
    inside = [e for e in tree if labels[e[0]] == labels[e[1]]]
    cut = [e for e in tree if labels[e[0]] != labels[e[1]]]
    fig = go.Figure()
    _lines(fig, xy, inside, MST_COLOR, 2.5)
    _lines(fig, xy, cut, MISS_COLOR, 3.5, dash="dash")
    fig.add_trace(go.Scatter(x=[p[0] for p in xy], y=[p[1] for p in xy], mode="markers", marker=dict(size=9, color=_palette(labels), line=dict(width=1, color="white")), hoverinfo="skip", showlegend=False))
    return _map_axes(fig)


def build_dendrogram(heights, k):
    """Verschmelzungshöhen des Single-Linkage-Dendrogramms in absteigender Reihenfolge; die k - 1 teuersten (abgeschnittenen) sind hervorgehoben."""
    hs = sorted(heights, reverse=True)
    xs = list(range(1, len(hs) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=hs, mode="lines+markers", line=dict(color="#888", width=1.5), marker=dict(size=6, color=["#d62728" if x < k else "#4c78a8" for x in xs]), name="Verschmelzungshöhe", hoverinfo="skip"))
    fig.update_xaxes(title_text="Verschmelzung, teuerste zuerst")
    fig.update_yaxes(title_text="Höhe (Kantenlänge)", type="log")
    return _base(fig, 300, legend_y=-0.35)


def build_ops_stacked(runs, winner):
    """Elementarschritte je Weg, gestapelt in Aufbau der Triangulierung, Abstände und MST-Verfahren; der Gewinner ist umrandet."""
    ways = list(runs)
    labels = [WAY_LABELS[w] for w in ways]
    fig = go.Figure()
    for part, name, key in (("build", "Aufbau (Delaunay)", "build_ops"), ("dist", "Abstände", "dist_ops"), ("mst", "MST-Verfahren", "mst_ops")):
        fig.add_trace(go.Bar(x=labels, y=[getattr(runs[w], key) for w in ways], name=name, marker_color=PART_COLORS[part],
                             marker_line=dict(width=[3 if w == winner else 0 for w in ways], color="#14233B")))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Elementarschritte")
    return _base(fig, 340, legend_y=-0.3)


def build_sweep(rows, param_label, series, y_label, log_y=False, ref_line=None, ref_label=None, key="value"):
    """`series` = [(key, Name, Farbe)]: Median als Linie, 10. bis 90. Perzentil als Band (`<key>_lo`/`<key>_hi`)."""
    xs = [str(r[key]) for r in rows]
    fig = go.Figure()
    for name_key, name, color in series:
        ys = [r[name_key] for r in rows]
        lo = [r.get(f"{name_key}_lo", r[name_key]) for r in rows]
        hi = [r.get(f"{name_key}_hi", r[name_key]) for r in rows]
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], mode="lines", fill="toself", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.13)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=color, width=2.5), name=name))
    if ref_line is not None:
        fig.add_hline(y=ref_line, line=dict(color="#888", dash="dash", width=1.5), annotation_text=ref_label, annotation_position="top left")
    fig.update_xaxes(title_text=param_label, type="category")
    fig.update_yaxes(title_text=y_label, type="log" if log_y else "linear")
    return _base(fig, 360, legend_y=-0.3)


def build_crossover(rows, n_star):
    """Elementarschritte der vier Wege über n rund um den Kreuzungspunkt; senkrecht der Kreuzungspunkt Delaunay + Prim gegen Prim-Array."""
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    for w in ("prim_array", "delaunay_prim", "delaunay_kruskal", "kruskal_complete"):
        fig.add_trace(go.Scatter(x=ns, y=[r[f"ops_{w}"] for r in rows], mode="lines+markers", line=dict(color=WAY_COLORS[w], width=2.5), name=WAY_LABELS[w]))
    if n_star is not None:
        fig.add_vline(x=n_star, line=dict(color="#888", dash="dash", width=1.5), annotation_text=f"n = {n_star}", annotation_position="top left")
    fig.update_xaxes(title_text="Punkte n")
    fig.update_yaxes(title_text="Elementarschritte")
    return _base(fig, 340, legend_y=-0.3)


def build_terrain(rows):
    """Links: Anteil der Instanzen mit fehlender MST-Kante; rechts: Kostenaufschlag des MST auf Delaunay (Mittel und Maximum)."""
    xs = [str(r["terrain"]) for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Instanzen mit fehlender MST-Kante (%)", "Kostenaufschlag auf Delaunay (%)"), horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=xs, y=[r["miss_share"] for r in rows], mode="lines+markers", line=dict(color=MISS_COLOR, width=2.5), name="fehlende Kante", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["overhead_mean"] for r in rows], mode="lines+markers", line=dict(color=MST_COLOR, width=2.5), name="Mittel"), row=1, col=2)
    fig.add_trace(go.Scatter(x=xs, y=[r["overhead_max"] for r in rows], mode="lines+markers", line=dict(color="#888", width=2, dash="dot"), name="Maximum"), row=1, col=2)
    fig.update_xaxes(title_text="Geländezuschlag", type="category")
    fig.update_yaxes(rangemode="tozero")
    return _base(fig, 320, legend_y=-0.3)


def build_highdim(rows, ds, ks, key, y_label, lo_key=None, hi_key=None):
    """Eine Linie je k über die Dimension d (Kategorien); optional Band 10. bis 90. Perzentil."""
    xs = [str(d) for d in ds]
    fig = go.Figure()
    colors = ["#d62728", "#e8a13a", "#54a24b", "#4c78a8", "#7b3fbf"]
    for k, color in zip(ks, colors):
        sel = {r["d"]: r for r in rows if r["k"] == k}
        fig.add_trace(go.Scatter(x=xs, y=[sel[d][key] for d in ds], mode="lines+markers", line=dict(color=color, width=2.5), name=f"k = {k}"))
    fig.update_xaxes(title_text="Dimension d", type="category")
    fig.update_yaxes(title_text=y_label)
    return _base(fig, 320, legend_y=-0.3)
