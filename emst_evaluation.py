"""Auswertung: wann lohnt sich Delaunay statt des vollständigen Graphen, wie viele Kanten haben die Kandidatengraphen, wann bricht die Annahme "euklidisch", was zeigt der MST als Single-Linkage-
Dendrogramm, wie gut ist die kNN-Näherung in hohen Dimensionen? Vier Wege auf derselben Instanz (Prim-Array implizit, Kruskal vollständig, Delaunay + Kruskal, Delaunay + Prim), Aufwand in
**Elementarschritten** (Abstände, Vergleiche, Heap-Operationen, Tests beim Triangulieren), nicht in Laufzeit; das ist ein Näherungsmaß. Alles ist deterministisch: Kennzahlen laufen über
5 feste Instanzen (Seeds 100000-100004), Median mit 10./90. Perzentil.

- **Schranke** = 3n - 3 - h Kanten der Delaunay-Triangulierung (h = Punkte auf der konvexen Hülle).
- **Fehlanteil** = Anteil der Instanzen, in deren echtem MST (mit Geländezuschlag) mindestens eine Kante nicht in der Delaunay-Triangulierung liegt.
- **Abdeckung** (kNN) = Anteil der Kanten des exakten MST, die im kNN-Graphen vorkommen."""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import emst_constants as C
import emst_delaunay as D
import emst_methods as M
import emst_scenario as S

INF = float("inf")
WAYS = M.WAYS
WAY_LABELS = M.WAY_LABELS


@dataclass(frozen=True)
class Settings:
    kind: str = "uniform"
    n: int = C.DEFAULT_N
    clusters: int = C.DEFAULT_CLUSTERS
    terrain: float = C.DEFAULT_TERRAIN
    seed: int = C.DEFAULT_SEED


@lru_cache(maxsize=256)
def instance_of(settings):
    return S.generate(settings.n, settings.kind, settings.clusters, settings.terrain, settings.seed)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    dt: object
    runs: dict                  # Weg -> MstRun
    gabriel: list
    rng: list

    @property
    def ops(self):
        return {w: self.runs[w].ops for w in WAYS}

    @property
    def winner(self):
        ops = self.ops
        return min(WAYS, key=lambda w: (ops[w], WAYS.index(w)))

    @property
    def true_tree(self):
        """Der echte MST (Kruskal auf dem vollständigen Graphen, mit Geländezuschlag)."""
        return self.runs["kruskal_complete"].tree

    @property
    def missing(self):
        """Kanten des echten MST, die nicht in der Delaunay-Triangulierung liegen."""
        edges = set(self.dt.edges)
        return [e for e in self.true_tree if e not in edges]

    @property
    def same_tree(self):
        ref = frozenset(self.true_tree)
        return all(frozenset(r.tree) == ref for r in self.runs.values())

    @property
    def overhead(self):
        """Kostenaufschlag (in Prozent) des MST auf der Delaunay-Triangulierung gegenüber dem echten MST."""
        true = self.runs["kruskal_complete"].cost
        return 100.0 * (self.runs["delaunay_kruskal"].cost / true - 1.0) if true > 0 else 0.0

    @property
    def bound(self):
        return 3 * self.inst.n - 3 - len(self.dt.hull)

    @property
    def layers(self):
        """Kantenmengen der Ebenen: mst (echter MST), rng, gabriel, delaunay."""
        return {"mst": sorted(self.true_tree), "rng": self.rng, "gabriel": self.gabriel, "delaunay": self.dt.edges}

    @property
    def per_point(self):
        n = max(1, self.inst.n)
        return {"build": self.dt.ops / n, "orient": self.dt.orient_tests / n, "incircle": self.dt.incircle_tests / n, "created": self.dt.created / n,
                "cavity": float(np.mean(self.dt.cavity_sizes)) if self.dt.cavity_sizes else 0.0}


def analyse(settings):
    inst = instance_of(settings)
    dt = D.delaunay(inst.xy)
    comp = S.complete_edges(inst)
    runs = {w: M.run_way(inst, w, dt, comp) for w in WAYS}
    return Analysis(settings, inst, dt, runs, M.gabriel_edges(inst.xy, dt), M.rng_edges(inst.xy, dt))


# --- Sweeps ---------------------------------------------------------------------------------------------------------------------------------------


def _stats(values):
    values = [v for v in values if not np.isnan(v) and v != INF]
    if not values:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(values)), float(np.percentile(values, 10)), float(np.percentile(values, 90))


def run_config(base, seeds=C.SWEEP_SEEDS, **changes):
    s0 = replace(base, **changes)
    rows = [analyse(replace(s0, seed=seed)) for seed in seeds]
    out = {"n_runs": len(rows), "same_tree_share": 100.0 * sum(r.same_tree for r in rows) / len(rows), "miss_share": 100.0 * sum(bool(r.missing) for r in rows) / len(rows)}
    for w in WAYS:
        out[f"wins_{w}"] = sum(r.winner == w for r in rows)
    for key, values in (
        *[(f"ops_{w}", [float(r.ops[w]) for r in rows]) for w in WAYS],
        ("pairs", [float(r.inst.n * (r.inst.n - 1) // 2) for r in rows]),
        ("dt_edges", [float(len(r.dt.edges)) for r in rows]),
        ("gabriel_edges", [float(len(r.gabriel)) for r in rows]),
        ("rng_edges", [float(len(r.rng)) for r in rows]),
        ("mst_edges", [float(len(r.true_tree)) for r in rows]),
        ("bound", [float(r.bound) for r in rows]),
        ("hull", [float(len(r.dt.hull)) for r in rows]),
        ("build", [float(r.dt.ops) for r in rows]),
        ("build_per_point", [r.per_point["build"] for r in rows]),
        ("orient_per_point", [r.per_point["orient"] for r in rows]),
        ("incircle_per_point", [r.per_point["incircle"] for r in rows]),
        ("cavity", [r.per_point["cavity"] for r in rows]),
        ("speedup_prim", [r.ops["prim_array"] / r.ops["delaunay_prim"] for r in rows]),
        ("speedup_kruskal", [r.ops["kruskal_complete"] / r.ops["delaunay_kruskal"] for r in rows]),
        ("overhead", [r.overhead for r in rows]),
        ("missing", [float(len(r.missing)) for r in rows]),
    ):
        out[key], out[f"{key}_lo"], out[f"{key}_hi"] = _stats(values)
    return out


SWEEP_VALUES = {"n": C.N_SWEEP, "terrain": (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)}
SWEEP_LABELS = {"n": "Punkte n", "terrain": "Geländezuschlag"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def crossover(base, ns=tuple(range(10, 61, 5)), seeds=C.SWEEP_SEEDS):
    """Kleinstes n, ab dem "Delaunay + Prim" im Median weniger Schritte braucht als Prim mit Array auf dem impliziten vollständigen Graphen - und ab dem das für ALLE größeren getesteten n gilt.
    Rückgabe: (n_star oder None, Zeilen {n, ops_prim_array, ops_delaunay_prim, ops_delaunay_kruskal, ops_kruskal_complete})."""
    rows = []
    for n in ns:
        r = run_config(base, seeds, n=n)
        rows.append({"n": n, **{f"ops_{w}": r[f"ops_{w}"] for w in WAYS}})
    wins = [row["ops_delaunay_prim"] < row["ops_prim_array"] for row in rows]
    n_star = None
    for i, row in enumerate(rows):
        if all(wins[i:]):
            n_star = row["n"]
            break
    return n_star, rows


def terrain_miss(base, terrains=SWEEP_VALUES["terrain"], seeds=C.TERRAIN_SEEDS):
    """Wie oft bricht die Annahme "euklidisch"? Je Geländezuschlag über `seeds` Instanzen: Anteil mit fehlender MST-Kante, mittlere Zahl fehlender Kanten, mittlerer und größter Kostenaufschlag."""
    rows = []
    for t in terrains:
        an = [analyse(replace(base, terrain=t, seed=seed)) for seed in seeds]
        miss = [len(a.missing) for a in an]
        over = [a.overhead for a in an]
        rows.append({"terrain": t, "miss_share": 100.0 * sum(m > 0 for m in miss) / len(an), "missing_mean": float(np.mean(miss)), "overhead_mean": float(np.mean(over)), "overhead_max": float(max(over)), "n_runs": len(an)})
    return rows


def linkage_profile(analysis):
    """Single-Linkage aus dem MST: Verschmelzungshöhen (aufsteigend) und, bei Clustern, die Reinheit des Schnitts in k = (wahre Wolkenzahl) Cluster gegen die wahren Wolken."""
    inst = analysis.inst
    cost = {(u, v): w for u, v, w in S.complete_edges(inst)}
    tree = [(u, v, cost[(u, v)]) for u, v in analysis.true_tree]
    merges = M.single_linkage(inst.n, tree)
    out = {"tree": tree, "heights": [m[0] for m in merges], "merges": merges, "purity": None}
    if inst.truth is not None:
        labels = M.clusters_from_tree(inst.n, tree, inst.clusters)
        out["purity"] = _purity(labels, inst.truth)
    return out


def _purity(labels, truth):
    groups = {}
    for lab, t in zip(labels, truth):
        groups.setdefault(lab, []).append(int(t))
    return sum(max(np.bincount(g)) for g in groups.values()) / len(labels)


def chaining(base, clusters=C.CLUSTERS_SWEEP, seeds=C.TERRAIN_SEEDS):
    """Reinheit des Single-Linkage-Schnitts (k = wahre Wolkenzahl) über die Zahl der Wolken: mehr Wolken überlappen stärker, dann verkettet Single-Linkage sie (Chaining)."""
    rows = []
    for c in clusters:
        pur = [linkage_profile(analyse(replace(base, kind="clusters", clusters=c, seed=seed)))["purity"] for seed in seeds]
        rows.append({"clusters": c, "purity": float(np.median(pur)), "purity_lo": float(np.percentile(pur, 10)), "purity_hi": float(np.percentile(pur, 90)), "perfect_share": 100.0 * sum(p == 1.0 for p in pur) / len(pur)})
    return rows


def highdim(ds=C.HIGHDIM_DIMS, ks=C.HIGHDIM_KS, n=C.HIGHDIM_N, seeds=C.SWEEP_SEEDS):
    """kNN-Näherung im R^d (gleichverteilt im Einheitswürfel): je (d, k) Median der Abdeckung, des Kostenfehlers (Prozent) und Anteil der Instanzen, deren kNN-Graph ohne Reparatur zusammenhängt."""
    rows = []
    exact = {(d, s): M.exact_emst(S.generate_points(n, d, s)) for d in ds for s in seeds}
    for d in ds:
        for k in ks:
            runs = [M.knn_mst(S.generate_points(n, d, s), k, exact[(d, s)]) for s in seeds]
            cov = [r.coverage for r in runs]
            err = [100.0 * r.error for r in runs]
            rows.append({"d": d, "k": k, "coverage": float(np.median(cov)), "coverage_lo": float(np.percentile(cov, 10)), "coverage_hi": float(np.percentile(cov, 90)),
                         "error": float(np.median(err)), "error_lo": float(np.percentile(err, 10)), "error_hi": float(np.percentile(err, 90)),
                         "connected_share": 100.0 * sum(r.connected for r in runs) / len(runs), "edges": float(np.median([r.edges for r in runs])), "repairs": float(np.mean([r.repairs for r in runs]))})
    return rows
