"""Prim-Algorithmus in drei Umsetzungen, Kruskal als Vergleich und Referenzen.

Prim (Jarník 1930, Prim 1957): der Baum wächst von EINEM Startknoten aus; in jedem Schritt kommt die billigste Kante hinzu, die den Baum mit einem noch nicht angeschlossenen Knoten
verbindet (Schnitt-Eigenschaft: die billigste Kante über den Schnitt Baum | Rest gehört zu einem MST). Die Kanten zwischen Baum und Rest heißen der **Rand**. Kruskal
(`kruskal` unten, wortgleich aus kruskal-demo, dazu ein Zähler für die Sortier-Vergleiche) sortiert dagegen alle Kanten und fügt Fragmente zusammen.

Der Schlüssel jeder Kante ist (Kosten, Kantenindex): eine STRIKTE Gesamtordnung. Damit ist der MST eindeutig, und Prim (von jedem Startknoten) und Kruskal (Tie-Regel "lex")
liefern dieselbe Kantenmenge - auch bei Gleichständen in den Kosten.

Umsetzungen (verschiedener Aufwand, derselbe Baum):
- `array`: je Runde alle noch nicht angeschlossenen Knoten scannen (O(n²), keine Datenstruktur).
- `lazy`:  binärer Heap aus KANTEN; beim Entnehmen werden veraltete Einträge (Ziel schon im Baum) verworfen. Der Heap kann bis zu m Einträge halten.
- `eager`: binärer Heap aus KNOTEN mit Schlüssel = billigste Rand-Kante zu diesem Knoten; wird eine billigere Kante gefunden, sinkt der Schlüssel (Decrease-Key). Höchstens n Einträge.

Aufwand wird in **Elementarschritten** gezählt, nicht in Laufzeit (die hängt von Rechner und Sprache ab): Schlüsselvergleiche plus je eine Einheit je Heap-Einfügung/-Entnahme/
Decrease-Key (Prim), bzw. Sortier-Vergleiche plus je Union-Find-Suche und Zeigerschritt (Kruskal). Das ist ein Näherungsmaß für den Vergleich der Verfahren, kein Wall-Clock-Beweis."""

import math
from dataclasses import dataclass, field

import numpy as np

from emst_unionfind import UnionFind

VARIANTS = ("lazy", "eager", "array")


class BinaryHeap:
    """Binärer Min-Heap mit Zählern. `indexed=True` erlaubt Decrease-Key (jedes Element höchstens einmal, per Schlüssel `item` auffindbar)."""

    def __init__(self, indexed=False):
        self.a = []
        self.pos = {} if indexed else None
        self.comparisons = self.swaps = self.pushes = self.pops = self.decrease_keys = self.max_size = 0

    def __len__(self):
        return len(self.a)

    def _less(self, i, j):
        self.comparisons += 1
        return self.a[i][0] < self.a[j][0]

    def _swap(self, i, j):
        self.swaps += 1
        self.a[i], self.a[j] = self.a[j], self.a[i]
        if self.pos is not None:
            self.pos[self.a[i][1]] = i
            self.pos[self.a[j][1]] = j

    def _up(self, i):
        while i > 0:
            p = (i - 1) // 2
            if self._less(i, p):
                self._swap(i, p)
                i = p
            else:
                break

    def _down(self, i):
        n = len(self.a)
        while True:
            l, r, m = 2 * i + 1, 2 * i + 2, i
            if l < n and self._less(l, m):
                m = l
            if r < n and self._less(r, m):
                m = r
            if m == i:
                return
            self._swap(i, m)
            i = m

    def push(self, key, item):
        self.pushes += 1
        self.a.append((key, item))
        if self.pos is not None:
            self.pos[item] = len(self.a) - 1
        self.max_size = max(self.max_size, len(self.a))
        self._up(len(self.a) - 1)

    def pop(self):
        self.pops += 1
        top = self.a[0]
        last = self.a.pop()
        if self.a:
            self.a[0] = last
            if self.pos is not None:
                self.pos[last[1]] = 0
            self._down(0)
        if self.pos is not None:
            del self.pos[top[1]]
        return top

    def decrease(self, item, key):
        """Setzt den Schlüssel von `item` auf den kleineren `key`."""
        self.decrease_keys += 1
        i = self.pos[item]
        self.a[i] = (key, item)
        self._up(i)

    def contains(self, item):
        return item in self.pos

    def valid(self):
        return all(self.a[(i - 1) // 2][0] <= self.a[i][0] for i in range(1, len(self.a)))


@dataclass
class PrimResult:
    tree: list                                     # Kantenindizes in Annahmereihenfolge
    cost: float
    variant: str = "lazy"
    start: int = 0
    steps: list = field(default_factory=list)      # je angenommene Kante: (Kantenindex, Baumgröße, Heap-/Kandidatengröße nach dem Schritt, Rand-Kanten)
    comparisons: int = 0                           # Schlüsselvergleiche gesamt (Heap + Schlüsselvergleiche + Scans)
    ops: int = 0                                   # Elementarschritte (siehe Modul-Docstring)
    pushes: int = 0
    pops: int = 0
    stale_pops: int = 0                            # lazy: verworfene veraltete Einträge
    decrease_keys: int = 0                         # eager
    key_compares: int = 0                          # Vergleich "neue Kante billiger als bekannter Schlüssel"
    scans: int = 0                                 # array: gescannte Kandidaten
    max_heap: int = 0                              # größter Heap (lazy: Kanten, eager: Knoten); array: größter Kandidatenbestand
    connected: bool = True


def prim(n, edges, variant="lazy", start=0):
    """`edges` = Sequenz (u, v, w). Bei unzusammenhängendem Graphen ein Baum der Komponente des Startknotens (`connected=False`)."""
    if variant not in VARIANTS:
        raise ValueError(f"unbekannte Variante {variant}")
    if not 0 <= start < max(n, 1):
        raise ValueError("Startknoten außerhalb")
    adj = [[] for _ in range(n)]
    for idx, (u, v, w) in enumerate(edges):
        adj[u].append((v, w, idx))
        adj[v].append((u, w, idx))
    seen = [False] * n
    res = PrimResult([], 0.0, variant, start)
    frontier = 0                                                 # Kanten zwischen Baum und Rest

    def account_add(u):
        nonlocal frontier
        for v, _w, _idx in adj[u]:
            frontier += -1 if seen[v] else 1

    if n == 0:
        return res
    if variant == "array":
        best = [None] * n
        seen[start] = True
        account_add(start)
        for v, w, idx in adj[start]:
            cand = (w, idx)
            if best[v] is None:
                best[v] = (cand, start)
            else:
                res.key_compares += 1
                if cand < best[v][0]:
                    best[v] = (cand, start)
        while len(res.tree) < n - 1:
            pick, pick_key = -1, None
            candidates = 0
            for v in range(n):
                if seen[v]:
                    continue
                candidates += 1
                res.scans += 1
                if best[v] is not None and (pick_key is None or best[v][0] < pick_key):
                    pick, pick_key = v, best[v][0]
            res.max_heap = max(res.max_heap, candidates)
            if pick < 0:
                res.connected = False
                break
            seen[pick] = True
            account_add(pick)
            res.tree.append(pick_key[1])
            res.cost += pick_key[0]
            for v, w, idx in adj[pick]:
                if not seen[v]:
                    cand = (w, idx)
                    if best[v] is None:
                        best[v] = (cand, pick)
                    else:
                        res.key_compares += 1
                        if cand < best[v][0]:
                            best[v] = (cand, pick)
            res.steps.append((pick_key[1], len(res.tree) + 1, candidates - 1, frontier))
        res.comparisons = res.scans + res.key_compares
        res.ops = res.comparisons
        return res

    heap = BinaryHeap(indexed=variant == "eager")
    key_of = [None] * n

    def discover(u):
        for v, w, idx in adj[u]:
            if seen[v]:
                continue
            cand = (w, idx)
            if variant == "lazy":
                heap.push(cand, (v, idx))
            elif key_of[v] is None:
                key_of[v] = cand
                heap.push(cand, v)
            else:
                res.key_compares += 1
                if cand < key_of[v]:
                    key_of[v] = cand
                    heap.decrease(v, cand)

    seen[start] = True
    account_add(start)
    discover(start)
    while heap and len(res.tree) < n - 1:
        key, item = heap.pop()
        v, idx = item if variant == "lazy" else (item, key[1])
        if seen[v]:
            res.stale_pops += 1
            continue
        seen[v] = True
        account_add(v)
        res.tree.append(idx)
        res.cost += key[0]
        discover(v)
        res.steps.append((idx, len(res.tree) + 1, len(heap), frontier))
    res.connected = len(res.tree) == n - 1
    res.comparisons = heap.comparisons + res.key_compares
    res.pushes, res.pops, res.decrease_keys, res.max_heap = heap.pushes, heap.pops, heap.decrease_keys, heap.max_size
    res.ops = res.comparisons + heap.pushes + heap.pops + heap.decrease_keys
    return res


# --- Kruskal (wortgleich aus kruskal-demo, ohne Filter-Variante; mit Zähler für die Sortier-Vergleiche) --------------------------------------------


@dataclass
class KruskalResult:
    tree: list
    cost: float
    steps: list = field(default_factory=list)      # je betrachtete Kante: (Kantenindex, angenommen, Komponentenzahl danach)
    examined: int = 0
    sort_comparisons: int = 0
    finds: int = 0
    find_steps: int = 0
    connected: bool = True

    @property
    def ops(self):
        return self.sort_comparisons + self.finds + self.find_steps


def counted_sort(items):
    """Sortiert `items` mit einem eigenen Mergesort und zählt die Vergleiche (unabhängig von der Sortierung der Python-Version). Gibt (sortierte Liste, Vergleiche) zurück."""
    count = 0

    def merge(a, b):
        nonlocal count
        out, i, j = [], 0, 0
        while i < len(a) and j < len(b):
            count += 1
            if b[j] < a[i]:
                out.append(b[j])
                j += 1
            else:
                out.append(a[i])
                i += 1
        return out + a[i:] + b[j:]

    def sort(xs):
        if len(xs) <= 1:
            return xs
        mid = len(xs) // 2
        return merge(sort(xs[:mid]), sort(xs[mid:]))

    return sort(list(items)), count


def kruskal(n, edges, uf_mode="full"):
    """Kruskal mit der Tie-Regel "lex" (Kantenindex): Schlüssel (Kosten, Index) - dieselbe Ordnung wie bei Prim."""
    m = len(edges)
    keyed, comparisons = counted_sort([(edges[i][2], i) for i in range(m)])
    order = [i for _w, i in keyed]
    uf = UnionFind(n, uf_mode)
    res = KruskalResult([], 0.0, sort_comparisons=comparisons)
    for i in order:
        if len(res.tree) >= n - 1:
            break
        res.examined += 1
        u, v, w = edges[i]
        if uf.union(u, v):
            res.tree.append(i)
            res.cost += w
            res.steps.append((i, True, uf.components))
        else:
            res.steps.append((i, False, uf.components))
    res.finds, res.find_steps = uf.finds, uf.find_steps
    res.connected = uf.components == 1
    return res


# --- Referenzen -----------------------------------------------------------------------------------------------------------------------------------


def start_node(inst, choice):
    """Startknoten: 'depot' = Knoten 0; 'far' = der vom Depot am weitesten entfernte Knoten; 'center' = der dem Schwerpunkt nächste Knoten."""
    if choice == "depot" or inst.n <= 1:
        return 0
    if choice == "far":
        d = np.hypot(inst.xy[:, 0] - inst.xy[0, 0], inst.xy[:, 1] - inst.xy[0, 1])
        return int(np.argmax(d))
    if choice == "center":
        c = inst.xy.mean(axis=0)
        d = np.hypot(inst.xy[:, 0] - c[0], inst.xy[:, 1] - c[1])
        return int(np.argmin(d))
    raise ValueError(f"unbekannter Start {choice}")


def spearman(order_a, order_b):
    """Rangkorrelation zweier Anordnungen DERSELBEN Elemente (Listen von Elementen); 1 = gleiche Reihenfolge, -1 = umgekehrt."""
    n = len(order_a)
    if n < 2:
        return 1.0
    rank_b = {x: i for i, x in enumerate(order_b)}
    d2 = sum((i - rank_b[x]) ** 2 for i, x in enumerate(order_a))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def expected_decrease_keys(n, m):
    """Erwartungswert der Decrease-Keys bei Zufallsreihenfolge der Kanten: etwa n * ln(m / n) (Knuth-artige Schranke; Vergleichsgröße, keine Garantie)."""
    return n * math.log(m / n) if m > n else 0.0
