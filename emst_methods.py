"""Wege zum Euklidischen MST, Kandidatengraph-Hierarchie, Single-Linkage aus dem MST und die kNN-Näherung für hohe Dimensionen.

Vier Wege (derselbe Baum, sehr verschiedener Aufwand; Aufwand in **Elementarschritten**, Abstände zählen je einen Schritt):
- `prim_array`:       Prim mit Array auf dem IMPLIZITEN vollständigen Graphen (n(n-1)/2 Abstände, keine Kantenliste).
- `kruskal_complete`: Kruskal auf der vollständigen Kantenliste (m Abstände + Sortieren + Union-Find).
- `delaunay_kruskal`: Delaunay-Triangulierung bauen, dann Kruskal auf den höchstens 3n - 3 Kanten.
- `delaunay_prim`:    Delaunay bauen, dann Prim mit Decrease-Key auf denselben Kanten.
Kanten stehen immer nach (u, v) sortiert, der Schlüssel (Kosten, Kantenindex) ordnet strikt: auf Teil- und Gesamtgraph kommt derselbe Baum heraus, auch bei Gleichständen (Gitter)."""

from dataclasses import dataclass, field

import numpy as np

import emst_algorithm as A
import emst_delaunay as D
import emst_scenario as S
from emst_unionfind import UnionFind

WAYS = ("prim_array", "kruskal_complete", "delaunay_kruskal", "delaunay_prim")
WAY_LABELS = {"prim_array": "Prim (Array, implizit vollständig)", "kruskal_complete": "Kruskal (vollständige Liste)", "delaunay_kruskal": "Delaunay + Kruskal", "delaunay_prim": "Delaunay + Prim (Decrease-Key)"}


@dataclass
class MstRun:
    way: str
    tree: list                          # Kantenpaare (u, v) mit u < v in Annahmereihenfolge
    cost: float
    build_ops: int = 0                  # Aufbau der Triangulierung
    dist_ops: int = 0                   # ausgewertete Abstände
    mst_ops: int = 0                    # Elementarschritte des MST-Verfahrens selbst
    edges_used: int = 0                 # Kanten, auf denen das MST-Verfahren arbeitet
    connected: bool = True

    @property
    def ops(self):
        return self.build_ops + self.dist_ops + self.mst_ops


def _pairs(edges, tree_idx):
    return [(edges[i][0], edges[i][1]) for i in tree_idx]


def run_way(inst, way, dt=None, complete=None):
    """`dt` (DelaunayResult) und `complete` (vollständige Kantenliste) können übergeben werden, damit sie nicht mehrfach entstehen."""
    n = inst.n
    if way in ("prim_array", "kruskal_complete"):
        edges = complete if complete is not None else S.complete_edges(inst)
        if way == "prim_array":
            r = A.prim(n, edges, "array", 0)
            return MstRun(way, _pairs(edges, r.tree), r.cost, 0, n * (n - 1) // 2, r.ops, len(edges), r.connected)
        r = A.kruskal(n, edges)
        return MstRun(way, _pairs(edges, r.tree), r.cost, 0, len(edges), r.ops, len(edges), r.connected)
    if way in ("delaunay_kruskal", "delaunay_prim"):
        dt = dt if dt is not None else D.delaunay(inst.xy)
        edges = S.edge_list(inst, dt.edges)
        if way == "delaunay_kruskal":
            r = A.kruskal(n, edges)
        else:
            r = A.prim(n, edges, "eager", 0)
        return MstRun(way, _pairs(edges, r.tree), r.cost, dt.ops, len(edges), r.ops, len(edges), r.connected)
    raise ValueError(f"unbekannter Weg {way}")


# --- Kandidatengraph-Hierarchie: MST ⊆ RNG ⊆ Gabriel ⊆ Delaunay --------------------------------------------------------------------------------


def gabriel_edges(xy, dt):
    """Delaunay-Kanten, in deren Durchmesserkreis (offen) kein Scheitel der anliegenden Dreiecke liegt. Gleich dem Gabriel-Graphen aller Punktpaare, sobald keine vier Punkte kozirkular sind."""
    opp = dt.opposite
    out = []
    for (u, v), apexes in opp.items():
        ux, uy, vx, vy = xy[u][0], xy[u][1], xy[v][0], xy[v][1]
        if all((ux - xy[r][0]) * (vx - xy[r][0]) + (uy - xy[r][1]) * (vy - xy[r][1]) >= 0 for r in apexes):
            out.append((u, v))
    return sorted(out)


def rng_edges(xy, dt):
    """Delaunay-Kanten (u, v), zu denen kein Punkt r mit max(d(u, r), d(v, r)) < d(u, v) existiert (relativer Nachbarschaftsgraph, Linse leer). Gegen ALLE Punkte geprüft: eine Prüfung nur
    unter den Delaunay-Nachbarn von u und v genügt nicht (der Zeuge kann ein entfernterer Punkt sein)."""
    pts = np.asarray(xy, dtype=float)
    out = []
    for u, v in dt.edges:
        du = ((pts - pts[u]) ** 2).sum(axis=1)
        dv = ((pts - pts[v]) ** 2).sum(axis=1)
        duv = du[v]
        inside = np.maximum(du, dv) < duv
        inside[u] = inside[v] = False
        if not inside.any():
            out.append((u, v))
    return sorted(out)


# --- Single-Linkage aus dem MST -------------------------------------------------------------------------------------------------------------------


def single_linkage(n, tree_edges):
    """`tree_edges` = [(u, v, Kosten)] eines MST. Gibt die Verschmelzungen (Höhe, Größe A, Größe B) in aufsteigender Höhe zurück - das Single-Linkage-Dendrogramm."""
    uf = UnionFind(n, "full")
    merges = []
    for u, v, w in sorted(tree_edges, key=lambda e: (e[2], e[0], e[1])):
        ru, rv = uf.find(u), uf.find(v)
        if ru != rv:
            merges.append((w, uf.size[ru], uf.size[rv]))
            uf.union(u, v)
    return merges


def clusters_from_tree(n, tree_edges, k):
    """k Cluster = MST ohne die k - 1 teuersten Kanten (bei Gleichstand die mit dem größeren (u, v)). Etiketten 0, 1, ... nach dem kleinsten Knoten."""
    k = max(1, min(int(k), n))
    ordered = sorted(tree_edges, key=lambda e: (e[2], e[0], e[1]))
    keep = ordered[: len(ordered) - (k - 1)] if k > 1 else ordered
    uf = UnionFind(n, "full")
    for u, v, _w in keep:
        uf.union(u, v)
    ids, out = {}, []
    for x in range(n):
        out.append(ids.setdefault(uf.find(x), len(ids)))
    return tuple(out)


# --- hohe Dimensionen: kNN-Graph -------------------------------------------------------------------------------------------------------------------


def distance_matrix(points):
    sq = (points ** 2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * points @ points.T
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2)


def exact_emst(points):
    """Exakter MST im R^d per Prim (O(n²), numpy): (Baum als Paare (u, v) mit u < v, Kosten). Gleichstände (Gitter) brechen nach kleinstem Index."""
    n = len(points)
    if n < 2:
        return [], 0.0
    dm = distance_matrix(points)
    in_tree = np.zeros(n, dtype=bool)
    in_tree[0] = True
    best = dm[0].copy()
    parent = np.zeros(n, dtype=int)
    tree, cost = [], 0.0
    for _ in range(n - 1):
        cand = np.where(in_tree, np.inf, best)
        v = int(np.argmin(cand))
        tree.append((min(v, int(parent[v])), max(v, int(parent[v]))))
        cost += float(best[v])
        in_tree[v] = True
        closer = dm[v] < best
        best = np.where(closer & ~in_tree, dm[v], best)
        parent = np.where(closer & ~in_tree, v, parent)
    return tree, cost


def knn_edges(points, k):
    """Symmetrischer kNN-Graph: (u, v) mit u < v, wenn v unter den k nächsten Nachbarn von u ist oder umgekehrt."""
    n = len(points)
    dm = distance_matrix(points)
    np.fill_diagonal(dm, np.inf)
    order = np.argsort(dm, axis=1, kind="stable")
    edges = set()
    for u in range(n):
        for v in order[u, : min(k, n - 1)]:
            edges.add((min(u, int(v)), max(u, int(v))))
    return sorted(edges)


@dataclass
class KnnRun:
    tree: list
    cost: float
    edges: int                          # Kanten des kNN-Graphen
    connected: bool                     # war der kNN-Graph ohne Reparatur zusammenhängend?
    repairs: int                        # zum Verbinden ergänzte (billigste echte) Kanten
    coverage: float = 1.0               # Anteil der Kanten des exakten MST im kNN-Graph
    error: float = 0.0                  # Kosten / exakte Kosten - 1
    exact_cost: float = 0.0
    missing: list = field(default_factory=list)


def knn_mst(points, k, exact=None):
    """FAMST-artige Näherung: MST auf dem kNN-Graphen; ist er nicht zusammenhängend, verbindet man die Komponenten der Reihe nach über die jeweils billigste echte Kante zwischen zwei verschiedenen Komponenten."""
    n = len(points)
    dm = distance_matrix(points)
    kedges = knn_edges(points, k)
    edges = tuple((u, v, float(dm[u, v])) for u, v in kedges)
    kr = A.kruskal(n, edges)
    tree = [(edges[i][0], edges[i][1]) for i in kr.tree]
    cost = kr.cost
    uf = UnionFind(n, "full")
    for u, v in tree:
        uf.union(u, v)
    connected, repairs = uf.components == 1, 0
    while uf.components > 1:
        roots = np.array([uf.find(x) for x in range(n)])
        cross = roots[:, None] != roots[None, :]
        masked = np.where(cross, dm, np.inf)
        flat = int(np.argmin(masked))
        u, v = divmod(flat, n)
        tree.append((min(u, v), max(u, v)))
        cost += float(dm[u, v])
        uf.union(u, v)
        repairs += 1
    ex_tree, ex_cost = exact if exact is not None else exact_emst(points)
    present = set(kedges)
    missing = [e for e in ex_tree if e not in present]
    coverage = 1.0 - len(missing) / max(1, len(ex_tree))
    return KnnRun(tree, cost, len(kedges), connected, repairs, coverage, (cost / ex_cost - 1.0) if ex_cost > 0 else 0.0, ex_cost, missing)
