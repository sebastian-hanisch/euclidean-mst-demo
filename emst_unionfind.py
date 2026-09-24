"""Union-Find (Disjoint-Set-Forest) in vier Ausbaustufen, mit Zählern - Kruskal braucht es, um "sind diese zwei Knoten schon verbunden?" zu beantworten.

- `naive`:    Wurzel des ersten unter die Wurzel des zweiten hängen, keine Kompression, kein Rang. Ohne Gegenmaßnahme kann der Wald zur Kette entarten.
- `compress`: wie `naive`, aber Pfadhalbierung beim Suchen (jeder besuchte Knoten springt auf den Großelternknoten).
- `rank`:     Vereinigung nach Rang (die flachere Wurzel wandert unter die höhere), keine Kompression: Tiefe <= log2 n.
- `full`:     Rang UND Pfadhalbierung (Tarjan 1975): nahezu konstante amortisierte Kosten (inverse Ackermann-Funktion).

Gezählt werden `finds` (Aufrufe), `find_steps` (Zeigerschritte über alle Suchen), `max_find_len` (längste einzelne Suche) und `unions`. Die Zähler gehören nicht zum Ergebnis von
Kruskal: alle vier Stufen liefern denselben Baum, sie unterscheiden sich nur im Aufwand."""

MODES = ("naive", "compress", "rank", "full")


class UnionFind:
    def __init__(self, n, mode="full"):
        if mode not in MODES:
            raise ValueError(f"unbekannter Modus {mode}")
        self.n = n
        self.mode = mode
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.components = n
        self.finds = 0
        self.find_steps = 0
        self.max_find_len = 0
        self.unions = 0
        self._compress = mode in ("compress", "full")
        self._by_rank = mode in ("rank", "full")

    def find(self, x):
        self.finds += 1
        parent = self.parent
        steps = 0
        if self._compress:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
                steps += 1
        else:
            while parent[x] != x:
                x = parent[x]
                steps += 1
        self.find_steps += steps
        if steps > self.max_find_len:
            self.max_find_len = steps
        return x

    def union(self, a, b):
        """Vereinigt die Komponenten von a und b; False, wenn sie schon dieselbe waren."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self._by_rank and self.rank[ra] > self.rank[rb]:
            ra, rb = rb, ra                                  # die flachere Wurzel (ra) wandert unter rb
        self.parent[ra] = rb
        self.size[rb] += self.size[ra]
        if self._by_rank and self.rank[ra] == self.rank[rb]:
            self.rank[rb] += 1
        self.components -= 1
        self.unions += 1
        return True

    def depth(self, x):
        d = 0
        while self.parent[x] != x:
            x = self.parent[x]
            d += 1
        return d

    def depths(self):
        return [self.depth(x) for x in range(self.n)]
