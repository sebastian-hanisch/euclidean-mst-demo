"""Konstanten der Euklidischer-MST-Demo: Instanz-Geometrie, Regler, gemessene Werte, Presets."""
AREA = 100.0
N_MIN, N_MAX, DEFAULT_N, N_STEP = 5, 300, 60, 1
CLUSTERS_MIN, CLUSTERS_MAX, DEFAULT_CLUSTERS = 2, 8, 4
CLUSTER_SIGMA = 5.0
GRID_SPACING = 10.0
TERRAIN_OPTIONS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
DEFAULT_TERRAIN = 0.0
SEED_MAX = 999999
DEFAULT_SEED = 35
KINDS = ("uniform", "clusters", "grid")
KIND_LABELS = {"uniform": "gleichverteilt", "clusters": "Cluster (Gauß-Wolken)", "grid": "Gitter (gleiche Abstände)"}
LAYERS = ("mst", "rng", "gabriel", "delaunay")
LAYER_LABELS = {"mst": "nur der MST", "rng": "RNG (relative Nachbarschaft)", "gabriel": "Gabriel-Graph", "delaunay": "Delaunay-Triangulierung"}
DEFAULT_LAYER = "delaunay"
DEFAULT_CUT, CUT_MAX = 4, 30
SWEEP_SEEDS = tuple(range(100000, 100005))
TERRAIN_SEEDS = tuple(range(200000, 200050))
N_SWEEP = (10, 20, 40, 80, 160, 300)
CLUSTERS_SWEEP = (2, 3, 4, 6, 8)
HIGHDIM_DIMS = (2, 3, 5, 10, 20, 50)
HIGHDIM_KS = (2, 3, 5, 10, 20)
HIGHDIM_N = 150

# --- Gemessene Werte (MEDIAN über 5 feste Instanzen, Seeds 100000-100004, gleichverteilt, rein euklidisch (Geländezuschlag 0); 2026-09-24, alle Werte über
# --- ev.run_config/ev.sweep/ev.crossover/ev.terrain_miss/ev.chaining/ev.highdim nachgerechnet, s. tests/test_claims.py). Aufwand in ELEMENTARSCHRITTEN (Abstände, Vergleiche, Heap-Operationen,
# --- Orientierungs- und In-Circle-Tests, neue Dreiecke, Sortier-Vergleiche), NICHT in Laufzeit - ein Näherungsmaß. Alle Verfahren sind deterministisch. ---
# GLEICHER BAUM: Prim (Array, implizit vollständig), Kruskal (vollständig), Delaunay + Kruskal und Delaunay + Prim liefern in 100 % der Läufe dieselbe Kantenmenge (Schlüssel (Kosten, Kantenindex)),
#   auch auf dem Gitter mit vielen gleichen Abständen. Der MST liegt in jedem Fall in der Delaunay-Triangulierung (Zufall, Cluster, Gitter, keine Ausnahme).
# HIERARCHIE MST ⊆ RNG ⊆ Gabriel ⊆ Delaunay: Kanten bei n = 10/20/40/80/160/300: Delaunay 22/49/109/227/465/882 (= 3n - 3 - h exakt), Gabriel 14/30/69/143/291/562, RNG 10/21/47/96/192/365, MST n - 1;
#   bei n = 300 also 2.9 / 1.9 / 1.2 Kanten je Punkt gegen 299 beim vollständigen Graphen. Auf dem Gitter sind Delaunay und Gabriel fast gleich (837 gegen 832 bei n = 300).
# AUFWAND (Elementarschritte) n = 10/20/40/80/160/300: Prim-Array 126/551/2301/9401/38001/134251, Kruskal vollständig 277/1540/7591/36986/171916/684295, Delaunay + Kruskal 319/823/1973/4638/10348/21111,
#   Delaunay + Prim 286/681/1576/3647/8072/16766. Das Verhältnis Prim-Array / (Delaunay + Prim) ist 0.44/0.81/1.46/2.58/4.71/8.01, Kruskal vollständig / (Delaunay + Kruskal) 0.85/1.87/3.89/8.03/16.6/32.4.
#   KREUZUNGSPUNKT (Schrittweite 5): ab n = 30 ist Delaunay + Prim im Median billiger als Prim-Array (Cluster: 30, Gitter: 25); darunter gewinnt das Array.
# AUFBAU der Triangulierung je Punkt (n = 10 bis 300): 19 bis 35 Schritte (Orientierungstests 7.4 bis 10.1, In-Circle-Tests 5.2 bis 10.6, mittlerer Hohlraum 2.8 bis 4.9 Dreiecke) - wächst nur langsam mit n;
#   auf dem Gitter (kozirkular) bis 44 bei n = 300.
# GELÄNDEZUSCHLAG (n = 60, 50 Instanzen, Seeds 200000-200049): Anteil der Instanzen, deren echter MST eine Kante außerhalb der Delaunay-Triangulierung hat, bei Zuschlag 0/0.05/0.1/0.2/0.4/0.8:
#   0/4/4/18/52/92 %; Kostenaufschlag des MST auf Delaunay im Mittel 0/0.00/0.00/0.02/0.11/0.55 %, im schlimmsten Fall 2.33 %. Bei n = 160: 0/24/86 % bei 0/0.1/0.4.
# CHAINING (Single-Linkage-Schnitt in k = wahre Wolkenzahl Cluster, Reinheit gegen die wahren Wolken, 50 Instanzen, n = 60): Median 1.00/0.74/0.77/0.68/0.64 bei 2/3/4/6/8 Wolken; exakt getroffen in
#   60/34/6/0/0 % der Instanzen.
# HOHE DIMENSIONEN (n = 150 gleichverteilt im Einheitswürfel, 5 Instanzen): der kNN-Graph mit k >= 5 enthält in allen getesteten Dimensionen (d = 2 bis 50) alle MST-Kanten (Abdeckung 1.00, Kostenfehler
#   0.0 %); Erwartung "höhere Dimension = schlechtere Näherung" widerlegt: am schlechtesten ist k = 2 in NIEDRIGER Dimension (d = 2: Abdeckung 0.93, Kostenfehler 1.02 %, kNN-Graph nie zusammenhängend;
#   d = 3: 0.93 und 1.58 %; danach sinkt der Fehler mit d: d = 5/10/20/50: 0.66/0.31/0.16/0.05 %, bei d = 50 Abdeckung 0.98 und immer zusammenhängend). Gleiches Bild bei n = 50 und n = 400.

PRESETS = {
    "Standardfall (Voreinstellung)": {"kind": "uniform", "n": 60, "clusters": 4, "terrain": 0.0, "seed": 35, "layer": "delaunay", "cut": 4},
    "Kleine Instanz (n = 20)": {"kind": "uniform", "n": 20, "clusters": 4, "terrain": 0.0, "seed": 35, "layer": "delaunay", "cut": 4},
    "Große Instanz (n = 300)": {"kind": "uniform", "n": 300, "clusters": 4, "terrain": 0.0, "seed": 35, "layer": "delaunay", "cut": 4},
    "Cluster (n = 120)": {"kind": "clusters", "n": 120, "clusters": 4, "terrain": 0.0, "seed": 35, "layer": "delaunay", "cut": 4},
    "Gitter (n = 100)": {"kind": "grid", "n": 100, "clusters": 4, "terrain": 0.0, "seed": 35, "layer": "delaunay", "cut": 4},
    "Geländezuschlag 0,4": {"kind": "uniform", "n": 60, "clusters": 4, "terrain": 0.4, "seed": 35, "layer": "delaunay", "cut": 4},
    "Single-Linkage-Chaining": {"kind": "clusters", "n": 120, "clusters": 6, "terrain": 0.0, "seed": 35, "layer": "mst", "cut": 6},
    "Kandidatengraphen (n = 150)": {"kind": "uniform", "n": 150, "clusters": 4, "terrain": 0.0, "seed": 35, "layer": "rng", "cut": 4},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "60 Punkte, Seed 35: Delaunay + Prim braucht 2634 Elementarschritte, Prim mit Array auf dem impliziten vollständigen Graphen 5251, Kruskal auf der vollständigen Liste 19 238. Derselbe Baum (Kosten 538.18); die Triangulierung hat 165 Kanten statt 1770 Paaren.",
    "Kleine Instanz (n = 20)": "Bei 20 Punkten lohnt sich die Triangulierung noch nicht: Prim mit Array (551 Schritte) schlägt Delaunay + Prim (685) und Delaunay + Kruskal (830); Kruskal auf der vollständigen Liste braucht 1578. Der Kreuzungspunkt liegt bei n = 30.",
    "Große Instanz (n = 300)": "300 Punkte: Delaunay + Prim braucht 17 130 Schritte, Prim mit Array 134 251 (7.8-fach mehr), Kruskal auf der vollständigen Liste 684 020 (40-fach). Der Aufbau kostet 35.9 Schritte je Punkt und wächst kaum.",
    "Cluster (n = 120)": "Vier Gauß-Wolken: dieselben Verhältnisse wie bei gleichverteilten Punkten (Delaunay + Prim 5709, Prim-Array 21 301, Kruskal vollständig 100 976), aber lange MST-Kanten zwischen den Wolken. Single-Linkage-Reinheit im Schnitt mit 4 Clustern: 87 von 120 Punkten.",
    "Gitter (n = 100)": "Ganzzahlige Koordinaten: viele gleiche Abstände und je vier kozirkulare Punkte, die Triangulierung ist nicht eindeutig (Delaunay und Gabriel fast gleich: 261 gegen 261 Kanten). Der Schlüssel (Kosten, Kantenindex) liefert trotzdem in allen vier Wegen denselben Baum (Kosten 990); Delaunay + Prim 4186, Prim-Array 14 751.",
    "Geländezuschlag 0,4": "Die Annahme \"euklidisch\" bricht: mit Kosten = Länge x Faktor in [1, 1.4] liegt eine Kante des echten MST (rot) außerhalb der Triangulierung; der MST auf Delaunay kostet 0.19 % mehr (633.94 statt 632.77). Über 50 Instanzen fehlt bei Zuschlag 0.4 in 52 % mindestens eine Kante.",
    "Single-Linkage-Chaining": "Sechs überlappende Wolken: der MST ist das Single-Linkage-Dendrogramm; schneidet man die 5 teuersten Kanten ab, ist die Reinheit gegen die wahren Wolken nur 65 von 120 Punkten - Ausreißer und Brücken verketten die Wolken (Chaining). Ansicht Schritt 4.",
    "Kandidatengraphen (n = 150)": "Die Hierarchie MST ⊆ RNG ⊆ Gabriel ⊆ Delaunay bei 150 Punkten: 436 Delaunay-Kanten (= 3n - 3 - h), 262 Gabriel-Kanten, 176 RNG-Kanten, 149 MST-Kanten gegen 11 175 Paare. Ebene wählbar in Schritt 3.",
}
# Beobachtete Spannweite des MEDIANS der Elementarschritte von "Delaunay + Prim" über die 5 festen Instanzen (mit Sicherheitsabstand).
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": (2300.0, 2900.0),
    "Kleine Instanz (n = 20)": (600.0, 760.0),
    "Große Instanz (n = 300)": (15000.0, 18500.0),
    "Cluster (n = 120)": (4900.0, 6200.0),
    "Gitter (n = 100)": (3700.0, 4700.0),
    "Geländezuschlag 0,4": (2300.0, 2900.0),
    "Single-Linkage-Chaining": (4900.0, 6200.0),
    "Kandidatengraphen (n = 150)": (6700.0, 8400.0),
}
