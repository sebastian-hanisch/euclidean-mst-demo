"""Delaunay-Triangulierung von Hand (numpy-frei im Kern, damit jeder Schritt zählbar ist): Bowyer-Watson mit Kantenspeicher.

Ein Dreieck gehört zur Delaunay-Triangulierung, wenn sein Umkreis keinen anderen Punkt enthält. Der Euklidische MST liegt in ihr (MST ⊆ RNG ⊆ Gabriel ⊆ Delaunay), sie hat höchstens 3n - 3 - h Kanten
(h = Punkte auf der konvexen Hülle) statt n(n-1)/2.

Aufbau (Punkte einzeln einfügen):
1. **Reihenfolge:** räumlich (Bänder in Schlangenlinie), damit der nächste Punkt meist nahe am zuletzt eingefügten liegt; die Vergleiche der Sortierung zählen mit.
2. **Punktsuche (Walk):** vom zuletzt entstandenen Dreieck aus über die Nachbarn zum Dreieck, das den Punkt enthält (Orientierungstests).
3. **Hohlraum:** alle Dreiecke, deren Umkreis den Punkt STRIKT enthält (In-Circle-Test), per Breitensuche über die Nachbardreiecke. Strikt, damit kozirkulare Punkte (Gitter) den Hohlraum nicht vergrößern.
4. **Neu verbinden:** Hohlraum entfernen, den Punkt mit jeder Randkante des Hohlraums zu einem Dreieck verbinden.
Ein großes Super-Dreieck umschließt alle Punkte; seine Ecken und Kanten werden am Ende entfernt.

Gezählt werden Orientierungs- und In-Circle-Tests, neue Dreiecke und Sortier-Vergleiche: `ops` = Elementarschritte des Aufbaus."""

from dataclasses import dataclass, field

from emst_algorithm import counted_sort

SUPER_SCALE = 1.0e6


def orient(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def incircle(ax, ay, bx, by, cx, cy, px, py):
    """> 0, wenn p strikt im Umkreis des GEGEN DEN UHRZEIGERSINN orientierten Dreiecks abc liegt."""
    adx, ady, bdx, bdy, cdx, cdy = ax - px, ay - py, bx - px, by - py, cx - px, cy - py
    return ((adx * adx + ady * ady) * (bdx * cdy - cdx * bdy) - (bdx * bdx + bdy * bdy) * (adx * cdy - cdx * ady) + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady))


@dataclass
class DelaunayResult:
    n: int
    triangles: list                                 # (a, b, c) gegen den Uhrzeigersinn, nur echte Punkte
    edges: list                                     # (u, v) mit u < v, sortiert
    hull: list = field(default_factory=list)        # Punkte auf dem Rand der Triangulierung (Hüllpunkte, ungeordnet, sortiert)
    order: list = field(default_factory=list)       # Einfüge-Reihenfolge
    events: list = field(default_factory=list)      # je Einfügung (Punkt, entfernte Dreiecke, neue Dreiecke); Dreiecke mit Super-Punkten (Index >= n) inklusive
    orient_tests: int = 0
    incircle_tests: int = 0
    created: int = 0
    walk_steps: int = 0
    sort_comparisons: int = 0
    cavity_sizes: list = field(default_factory=list)
    per_insert: list = field(default_factory=list)  # je Einfügung (Orientierungstests der Punktsuche, Walk-Schritte, In-Circle-Tests)
    collinear: bool = False

    @property
    def ops(self):
        return self.sort_comparisons + self.orient_tests + self.incircle_tests + self.created

    @property
    def opposite(self):
        """Kante -> Scheitel gegenüber in den anliegenden Dreiecken (1 bei Hüllkanten, sonst 2)."""
        opp = {}
        for a, b, c in self.triangles:
            for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
                opp.setdefault((min(u, v), max(u, v)), []).append(w)
        return opp


def canonical(t):
    """Dasselbe Dreieck (gleiche Orientierung) mit dem kleinsten Index zuerst."""
    i = t.index(min(t))
    return t[i:] + t[:i]


def spatial_order(xs, ys):
    """Bänder in Schlangenlinie: (Band, x aufsteigend bzw. absteigend). Gibt (Reihenfolge, Vergleiche der Sortierung) zurück."""
    n = len(xs)
    lo_y, hi_y = min(ys), max(ys)
    span = (hi_y - lo_y) or 1.0
    bands = max(1, int(round((n / 2.0) ** 0.5)))
    keyed = []
    for i in range(n):
        b = min(bands - 1, int((ys[i] - lo_y) / span * bands))
        keyed.append((b, xs[i] if b % 2 == 0 else -xs[i], i))
    out, comparisons = counted_sort(keyed)
    return [k[2] for k in out], comparisons


def _collinear(xs, ys):
    n = len(xs)
    if n < 3:
        return True
    i0 = 0
    j = next((j for j in range(1, n) if (xs[j], ys[j]) != (xs[0], ys[0])), None)
    if j is None:
        return True
    return all(orient(xs[i0], ys[i0], xs[j], ys[j], xs[k], ys[k]) == 0 for k in range(n))


def delaunay(points, super_scale=SUPER_SCALE):
    """Delaunay-Triangulierung von `points` (n x 2, verschiedene Punkte). Alle Punkte auf einer Geraden: keine Dreiecke, die Kanten sind der Pfad in Reihenfolge entlang der Geraden."""
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    n = len(xs)
    if len(set(zip(xs, ys))) != n:
        raise ValueError("doppelte Punkte")
    if n == 0:
        return DelaunayResult(0, [], [])
    if _collinear(xs, ys):
        order = sorted(range(n), key=lambda i: (xs[i], ys[i]))
        return DelaunayResult(n, [], [(min(a, b), max(a, b)) for a, b in zip(order, order[1:])], hull=sorted(range(n)), order=order, collinear=True)

    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    ext = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    short = min(max(xs) - min(xs), max(ys) - min(ys))
    m = super_scale * ext * max(1.0, ext / short if short > 0 else 1.0)          # flache Punktwolken brauchen ein entsprechend größeres Super-Dreieck
    sx = xs + [cx - m, cx + m, cx]
    sy = ys + [cy - m, cy - m, cy + 2 * m]
    order, sort_cmp = spatial_order(xs, ys)
    res = DelaunayResult(n, [], [], order=order, sort_comparisons=sort_cmp)

    tri, owner = {}, {}
    next_id = [0]

    def add(a, b, c):
        tid = next_id[0]
        next_id[0] += 1
        tri[tid] = (a, b, c)
        owner[(a, b)] = tid
        owner[(b, c)] = tid
        owner[(c, a)] = tid
        return tid

    def drop(tid):
        a, b, c = tri.pop(tid)
        for e in ((a, b), (b, c), (c, a)):
            del owner[e]

    last = add(n, n + 1, n + 2)
    for p in order:
        px, py = sx[p], sy[p]
        o0, w0, c0 = res.orient_tests, res.walk_steps, res.incircle_tests
        if last not in tri:
            last = next(iter(tri))
        t, steps = last, 0
        limit = 3 * len(tri) + 10
        while True:
            a, b, c = tri[t]
            moved = False
            for u, v in ((a, b), (b, c), (c, a)):
                res.orient_tests += 1
                if orient(sx[u], sy[u], sx[v], sy[v], px, py) < 0:
                    t = owner[(v, u)]
                    moved = True
                    res.walk_steps += 1
                    break
            steps += 1
            if not moved:
                break
            if steps > limit:                                # Sicherheitsnetz gegen Zyklen durch Rundungsfehler: das Dreieck linear suchen
                for tid, (a, b, c) in tri.items():
                    res.orient_tests += 3
                    if all(orient(sx[u], sy[u], sx[v], sy[v], px, py) >= 0 for u, v in ((a, b), (b, c), (c, a))):
                        t = tid
                        break
                break
        cavity, seen, stack = {t}, {t}, [t]
        while stack:
            cur = stack.pop()
            a, b, c = tri[cur]
            for u, v in ((a, b), (b, c), (c, a)):
                nb = owner.get((v, u))
                if nb is None or nb in seen:
                    continue
                seen.add(nb)
                na, nbb, nc = tri[nb]
                res.incircle_tests += 1
                if incircle(sx[na], sy[na], sx[nbb], sy[nbb], sx[nc], sy[nc], px, py) > 0:
                    cavity.add(nb)
                    stack.append(nb)
        boundary, removed = [], []
        for tid in cavity:
            a, b, c = tri[tid]
            removed.append((a, b, c))
            for u, v in ((a, b), (b, c), (c, a)):
                if owner.get((v, u)) not in cavity:
                    boundary.append((u, v))
        for tid in list(cavity):
            drop(tid)
        added = []
        for u, v in boundary:
            last = add(u, v, p)
            added.append((u, v, p))
            res.created += 1
        res.cavity_sizes.append(len(removed))
        res.per_insert.append((res.orient_tests - o0, res.walk_steps - w0, res.incircle_tests - c0))
        res.events.append((p, removed, added))

    res.triangles = sorted(canonical(t) for t in tri.values() if max(t) < n)
    edges, count = set(), {}
    for a, b, c in res.triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            e = (min(u, v), max(u, v))
            edges.add(e)
            count[e] = count.get(e, 0) + 1
    res.edges = sorted(edges)
    res.hull = sorted({x for e, k in count.items() if k == 1 for x in e})
    return res


def replay(result, i):
    """Dreiecksmenge (mit Super-Punkten) nach den ersten i Einfügungen."""
    n = result.n
    tris = {(n, n + 1, n + 2)}
    for _p, removed, added in result.events[:i]:
        tris -= set(removed)
        tris |= set(added)
    return tris
