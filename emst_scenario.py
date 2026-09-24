"""Die Instanz dieser Demo: n Standorte als Punkte auf einer Karte; gesucht wird das billigste Leitungsnetz, das alle verbindet. Kosten = euklidischer Abstand mal Geländefaktor u in [1, 1 + Zuschlag]
(Zuschlag 0 = rein euklidisch; nur dann liegt der MST garantiert in der Delaunay-Triangulierung). Der Faktor hängt nur vom Punktpaar und vom Seed ab.

Instanztypen: `uniform` (gleichverteilt im Quadrat), `clusters` (Gauß-Wolken: lange Kanten zwischen den Wolken, Single-Linkage-Chaining), `grid` (Gitter mit ganzzahligen Koordinaten: viele gleiche
Abstände und je vier kozirkulare Punkte - die Delaunay-Triangulierung ist dort nicht eindeutig)."""

from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np

import emst_constants as C


@dataclass(frozen=True)
class Instance:
    xy: np.ndarray                 # (n, 2)
    kind: str = "uniform"
    terrain: float = 0.0
    seed: int = 0
    clusters: int = 0
    factor: object = None          # (n, n) symmetrischer Geländefaktor oder None (= 1)
    truth: object = None           # bei Clustern: die Wolke, aus der jeder Punkt gezogen wurde (Etikett je Punkt)

    @property
    def n(self):
        return len(self.xy)


def _factor(n, terrain, seed):
    if terrain <= 0:
        return None
    rng = np.random.default_rng([int(seed), 808])
    f = rng.uniform(1.0, 1.0 + float(terrain), size=(n, n))
    return np.triu(f, 1) + np.triu(f, 1).T


def _clusters(n, clusters, rng):
    """Gibt (Punkte, Wolken-Etikett je Punkt) zurück."""
    centers = rng.uniform(15.0, C.AREA - 15.0, size=(clusters, 2))
    assign = rng.integers(0, clusters, size=n)
    pts = centers[assign] + rng.normal(0.0, C.CLUSTER_SIGMA, size=(n, 2))
    for _ in range(50):
        bad = (pts < 0).any(axis=1) | (pts > C.AREA).any(axis=1)
        if not bad.any():
            break
        pts[bad] = centers[assign[bad]] + rng.normal(0.0, C.CLUSTER_SIGMA, size=(int(bad.sum()), 2))
    return np.clip(pts, 0.0, C.AREA), assign


def generate(n=C.DEFAULT_N, kind="uniform", clusters=C.DEFAULT_CLUSTERS, terrain=C.DEFAULT_TERRAIN, seed=C.DEFAULT_SEED):
    if kind not in C.KINDS:
        raise ValueError(f"unbekannter Instanztyp {kind}")
    n = int(n)
    rng = np.random.default_rng([int(seed), 707])
    truth = None
    if kind == "grid":
        cols = int(ceil(sqrt(n)))
        pts = np.array([(C.GRID_SPACING * (i % cols), C.GRID_SPACING * (i // cols)) for i in range(n)], dtype=float)
    elif kind == "clusters":
        pts, truth = _clusters(n, int(clusters), rng)
    else:
        pts = rng.uniform(0.0, C.AREA, size=(n, 2))
    if len({(float(x), float(y)) for x, y in pts}) != n:                       # doppelte Punkte (Rand-Beschneidung) minimal verschieben
        seen = set()
        for i in range(n):
            key = (float(pts[i, 0]), float(pts[i, 1]))
            while key in seen:
                pts[i] += rng.uniform(1e-3, 2e-3, size=2)
                key = (float(pts[i, 0]), float(pts[i, 1]))
            seen.add(key)
    return Instance(pts, kind, float(terrain), int(seed), int(clusters) if kind == "clusters" else 0, _factor(n, terrain, seed), truth)


def cost_matrix(inst):
    diff = inst.xy[:, None, :] - inst.xy[None, :, :]
    d = np.hypot(diff[:, :, 0], diff[:, :, 1])
    return d if inst.factor is None else d * inst.factor


def complete_edges(inst):
    """Alle Paare (u, v, Kosten) mit u < v, sortiert nach (u, v)."""
    c = cost_matrix(inst)
    n = inst.n
    return tuple((u, v, float(c[u, v])) for u in range(n) for v in range(u + 1, n))


def edge_list(inst, pairs):
    """Kanten (u, v, Kosten) für die gegebenen Paare (u < v), sortiert nach (u, v)."""
    c = cost_matrix(inst)
    return tuple((u, v, float(c[u, v])) for u, v in sorted(pairs))


def generate_points(n, d, seed):
    """n gleichverteilte Punkte im Einheitswürfel [0, 1]^d (für die Näherung in hohen Dimensionen)."""
    return np.random.default_rng([int(seed), 909]).uniform(0.0, 1.0, size=(int(n), int(d)))
