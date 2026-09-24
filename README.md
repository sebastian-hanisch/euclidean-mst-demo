# Euklidischer MST – Delaunay statt n² Kanten – Streamlit-Demo

Viertes Stück der **Spannbaum-Reihe** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning". Kruskal, Prim und Borůvka bekamen den Graphen als Kantenliste - bei dichten Graphen war genau sie das Problem (bei 160 Punkten 12 880 Kanten). Sind die Kosten aber **euklidisch** (Kante = Abstand), braucht man den vollständigen Graphen nicht: der minimale Spannbaum liegt schon in der **Delaunay-Triangulierung**, dem Netz aus Dreiecken, deren Umkreis keinen anderen Punkt enthält. Sie hat höchstens **3n − 3 − h** Kanten statt n(n−1)/2 (h = Punkte auf der konvexen Hülle), und es gilt **MST ⊆ RNG ⊆ Gabriel-Graph ⊆ Delaunay** (Shamos & Hoey 1975: EMST in O(n log n)). Die Demo baut die Triangulierung **von Hand** (Bowyer-Watson, numpy-frei im Kern, scipy nur in den Tests als Gegenprobe) und misst, **ab wann sich der Umweg lohnt**, was die drei Kandidatengraphen an Kanten sparen, **wann die Annahme "euklidisch" bricht**, was der MST mit **Single-Linkage-Clustering** zu tun hat und ob die Näherung über nächste Nachbarn in **hohen Dimensionen** trägt. Kruskal und Prim aus [kruskal-demo](../kruskal-demo) und [prim-demo](../prim-demo) laufen als Vergleich und Kontrollrechnung mit.

**Einordnung in die Reihe:** geplant sind elf Stücke, dies ist das vierte:

```
Kruskal (Wurzel)                                                                           [gebaut: kruskal-demo]
 ├─ Prim (Kontrast: wächst von einem Punkt)                                                [gebaut: prim-demo]
 ├─ Borůvka (Kontrast: alle Komponenten parallel)                                          [gebaut: boruvka-demo]
 ├─ Euklidischer MST (keine n²-Kantenliste, Delaunay)                                      [DIESES STÜCK]
 ├─ Gerichteter Spannbaum (Chu-Liu/Edmonds)                                                [nicht gebaut]
 ├─ Bottleneck-/Grad-/Hop-beschränkter Spannbaum → Kapazitierter MST                       [nicht gebaut]
 ├─ Steiner-Baum → Prize-Collecting Steiner-Baum                                           [nicht gebaut]
 ├─ MST-Sensitivität & dynamischer MST                                                     [nicht gebaut]
 └─ Zufällige Spannbäume & Kirchhoff                                                       [nicht gebaut]
```

Ergebnis in Kürze: **Die Triangulierung lohnt sich ab etwa 30 Punkten - danach immer deutlicher (bei 300 Punkten 8-fach billiger als Prim mit Array, 41-fach billiger als Kruskal auf der vollständigen Liste) -, aber die Annahme "euklidisch" bricht schon bei kleinem Geländezuschlag, und die kNN-Näherung ist bei gleichverteilten Punkten nicht in hohen, sondern in niedrigen Dimensionen am schwächsten.** In **Elementarschritten** (Abstände, Vergleiche, Heap-Operationen, Orientierungs- und In-Circle-Tests, neue Dreiecke; ausdrücklich keine Laufzeit) braucht bei 60 Punkten Delaunay + Prim **2591**, Delaunay + Kruskal **3253**, Prim mit Array auf dem impliziten vollständigen Graphen **5251** und Kruskal auf der vollständigen Liste **19 206** Schritte. Der Baum ist immer derselbe, auch auf dem Gitter mit vielen gleichen Abständen.

| Frage | Ergebnis (gleichverteilte Punkte im Quadrat, rein euklidisch, sofern nicht anders angegeben; **Median** über 5 feste Instanzen, Seeds 100000–100004; vollständig deterministisch) |
|---|---|
| **Derselbe Baum?** | ✅ ja, in **100 %** der Instanzen für alle vier Wege (Prim-Array, Kruskal vollständig, Delaunay + Kruskal, Delaunay + Prim), gleichverteilt, Cluster und Gitter; der MST liegt immer in der Delaunay-Triangulierung |
| **Wie viele Kanten?** | n = 10/20/40/80/160/300: Delaunay **22/49/109/227/465/882** (= 3n − 3 − h exakt), Gabriel **14/30/69/143/291/562**, RNG **10/21/47/96/192/365**, MST n − 1; bei n = 300: **2,9 / 1,9 / 1,2** Kanten je Punkt gegen 299 beim vollständigen Graphen. Auf dem Gitter sind Delaunay und Gabriel fast gleich (837 gegen 832) |
| **Elementarschritte** | n = 10/20/40/80/160/300: Prim-Array **126/551/2301/9401/38 001/134 251**, Kruskal vollständig **277/1540/7591/36 986/171 916/684 295**, Delaunay + Kruskal **319/823/1973/4638/10 348/21 111**, Delaunay + Prim **286/681/1576/3647/8072/16 766** |
| **Ab wann lohnt es sich?** | Prim-Array / (Delaunay + Prim) = **0,44/0,81/1,46/2,58/4,71/8,01**; Kreuzungspunkt (Schrittweite 5) bei **n = 30** (Cluster 30, Gitter 25); darunter gewinnt das Array (n = 20: 551 gegen 681). Kruskal vollständig / (Delaunay + Kruskal) = 0,9/1,9/3,9/8,0/16,6/32,4. Kruskal gewinnt nie |
| **Was kostet der Aufbau?** | **19 bis 35** Schritte je Punkt (n = 10 bis 300; Orientierungstests 7,4 bis 10,1, In-Circle-Tests 5,2 bis 10,6, mittlerer Hohlraum 2,8 bis 4,9 Dreiecke) - wächst nur langsam; auf dem Gitter (kozirkular) bis 44 |
| **Wann bricht "euklidisch"?** | Kosten = Abstand mal Geländefaktor in [1, 1 + Zuschlag], n = 60, 50 Instanzen: Anteil mit fehlender MST-Kante bei Zuschlag 0/0,05/0,1/0,2/0,4/0,8: **0/4/4/18/52/92 %** (n = 160: 0/24/86 % bei 0/0,1/0,4). Der Schaden bleibt klein: Kostenaufschlag des MST auf Delaunay im Mittel **0/0,00/0,00/0,02/0,11/0,55 %**, im schlimmsten Fall 2,33 % |
| **Single-Linkage-Chaining** | Schnitt des MST in so viele Cluster wie Wolken, Reinheit gegen die wahren Wolken (n = 60, 50 Instanzen): Median **1,00/0,74/0,77/0,68/0,64** bei 2/3/4/6/8 Wolken; jede Wolke exakt getroffen in **60/34/6/0/0 %** der Instanzen |
| **Hohe Dimensionen (kNN-Näherung)** | 150 gleichverteilte Punkte im Einheitswürfel: mit **k ≥ 5** enthält der kNN-Graph in **jeder** Dimension von 2 bis 50 alle MST-Kanten (Kostenfehler 0,0 %). ⚠️ Erwartung "höhere Dimension = schlechtere Näherung" **widerlegt**: am schwächsten ist **k = 2 in niedriger Dimension** (d = 2: Abdeckung 0,93, Kostenfehler 1,02 %, kNN-Graph nie zusammenhängend; d = 3: 1,58 %; danach sinkt der Fehler: d = 5/10/20/50: 0,66/0,31/0,16/0,05 %). Gleiches Bild bei n = 50 und n = 400 |

## Was die Demo zeigt

1. **Delaunay und MST in Aktion** (Schritt-Slider): **Punkte** → **Delaunay wächst** (Slider über die eingefügten Punkte: Dreiecke grau, der eingefügte Punkt als grüner Stern, sein **Hohlraum** - die Dreiecke, deren Umkreis ihn enthält - rot gefüllt, die neuen Kanten orange; darüber der Text mit Orientierungstests der Punktsuche, Schritten über Nachbardreiecke, In-Circle-Tests und neuen Dreiecken) → **Kandidatenkanten und MST** (Ebene wählbar: nur MST / RNG / Gabriel / Delaunay; die Kantenzahlen der Hierarchie; mit Geländezuschlag die MST-Kante **außerhalb** der Triangulierung rot) → **Single-Linkage-Cluster** (Slider k: die k − 1 teuersten MST-Kanten abgeschnitten, Punktfarbe = Cluster, Dendrogramm-Höhen darunter, bei Clustern die Reinheit gegen die wahren Wolken).
2. **Was kostet welcher Weg?** Gewinner, Delaunay + Prim, Kanten, Aufbau je Punkt; gestapelte Balken der Elementarschritte je Weg (Aufbau / Abstände / MST-Verfahren); Zerlegung des Aufbaus.
3. **🔀 Kreuzungspunkt** (auf Abruf): n = 10 bis 60, vier Wege, Kreuzungspunkt von Delaunay + Prim gegen Prim-Array.
4. **🌄 Gelände-Experiment** (auf Abruf): Anteil der Instanzen mit fehlender MST-Kante und Kostenaufschlag über den Geländezuschlag.
5. **🔗 Chaining-Experiment** (auf Abruf): Reinheit des Single-Linkage-Schnitts über die Zahl der Wolken.
6. **🧭 Hohe Dimensionen** (auf Abruf): Abdeckung, Kostenfehler und Zusammenhang des kNN-Graphen über d = 2 bis 50 und k = 2 bis 20.
7. **📐 Sweeps** über n und Geländezuschlag (Elementarschritte, Verhältnisse, Kanten, Aufbau je Punkt, fehlende MST-Kanten; 5 feste Instanzen, Median, 10.–90. Perzentil-Band).
8. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an".

Regler: Punkte (gleichverteilt / **Cluster** / **Gitter**), n (5–300), Wolken (2–8), **Geländezuschlag** (0 = rein euklidisch, Voreinstellung; sonst Kosten = Abstand mal Faktor - ein echter Regler, der die Annahme bricht), Seed (+ 🎲; beim Gitter ohne Zuschlag entfällt er, weil nichts zufällig ist). Kein Zufall im Kern.

## Messwerte der Presets

| Preset | Instanz | Ergebnis (Elementarschritte Prim-Array / Kruskal vollständig / Delaunay + Kruskal / Delaunay + Prim) |
|---|---|---|
| Standardfall (Voreinstellung) | 60 gleichverteilte Punkte, Seed 35 | 5251 / 19 238 / 3320 / **2634**; Kosten 538,18; 165 Kanten statt 1770 Paaren |
| Kleine Instanz (n = 20) | | **551** / 1578 / 830 / 685 (das Array gewinnt noch) |
| Große Instanz (n = 300) | | 134 251 / 684 020 / 21 687 / **17 130**; Aufbau 35,9 Schritte je Punkt |
| Cluster (n = 120) | 4 Gauß-Wolken | 21 301 / 100 976 / 7685 / **5709**; Reinheit im 4-Cluster-Schnitt 87 von 120 Punkten |
| Gitter (n = 100) | ganzzahlig, gleiche Abstände | 14 751 / 57 977 / 5392 / **4186**; Delaunay und Gabriel je 261 Kanten; derselbe Baum, Kosten 990 |
| Geländezuschlag 0,4 | 60 Punkte | MST-Kante (20, 56) außerhalb der Triangulierung; MST auf Delaunay 633,94 statt 632,77 (+0,19 %) |
| Single-Linkage-Chaining | 6 Wolken, n = 120 | Reinheit im 6-Cluster-Schnitt nur 65 von 120 Punkten |
| Kandidatengraphen (n = 150) | | 436 Delaunay-, 262 Gabriel-, 176 RNG-, 149 MST-Kanten gegen 11 175 Paare |

Die einzelne Instanz weicht von den Medianen ab - die Mediane sind die belastbaren Zahlen; die Presets prüfen sich zusätzlich über die 5 festen Instanzen gegen eine gemessene Spannweite des Medians von Delaunay + Prim (`tests/test_presets.py`).

## Modell und Verfahren

- **Instanz** (`emst_scenario.py`): n Punkte im Quadrat (gleichverteilt), Gauß-Wolken (Standardabweichung 5) oder Gitter mit Abstand 10 (ganzzahlig: gleiche Abstände exakt gleich); Kosten = Abstand mal Geländefaktor in [1, 1 + Zuschlag], der nur vom Punktpaar und Seed abhängt.
- **Delaunay** (`emst_delaunay.py`): Bowyer-Watson mit Kantenspeicher. Punkte einzeln einfügen in räumlicher Reihenfolge (Bänder in Schlangenlinie, Vergleiche mitgezählt), das Dreieck des Punkts per **Walk** über die Nachbarn suchen (Orientierungstests), den **Hohlraum** (Dreiecke mit strikt enthaltendem Umkreis, In-Circle-Test) per Breitensuche bestimmen, neu verbinden. Ein großes Hilfsdreieck umschließt alle Punkte; sein Maßstab wächst mit dem Seitenverhältnis der Punktwolke (bis 10⁶ getestet). Alle Punkte auf einer Geraden: Pfad; doppelte Punkte werden abgelehnt.
- **Vier Wege** (`emst_methods.py`): Prim-Array auf dem impliziten vollständigen Graphen (n(n−1)/2 Abstände, keine Kantenliste), Kruskal auf der vollständigen Liste, Delaunay + Kruskal, Delaunay + Prim (Decrease-Key). Abstände zählen je einen Schritt; Kanten stehen immer nach (u, v) sortiert, der Schlüssel (Kosten, Kantenindex) ordnet strikt.
- **Kandidatengraphen:** Gabriel = Delaunay-Kanten, in deren offenem Durchmesserkreis kein Scheitel der anliegenden Dreiecke liegt; RNG = Delaunay-Kanten mit leerer Linse **gegen alle Punkte** geprüft (eine Prüfung nur unter Delaunay-Nachbarn genügt nicht - so erst beim Testen gegen die Definition aufgefallen).
- **Single-Linkage:** Verschmelzungshöhen = sortierte MST-Kantenlängen; k Cluster = MST ohne die k − 1 teuersten Kanten.
- **kNN-Näherung:** MST auf dem symmetrisierten kNN-Graphen, unverbundene Komponenten über die jeweils billigste echte Verbindung verbunden (der Baum ist immer ein Spannbaum, sein Fehler ≥ 0).
- **Elementarschritte:** Abstände (je 1), Vergleiche und Heap-Operationen (MST-Verfahren), Orientierungs- und In-Circle-Tests, neue Dreiecke, Sortier-Vergleiche (Delaunay). Ein Näherungsmaß für den Vergleich, **keine Laufzeit**.

## Was nicht funktioniert hat / Grenzen

- **Erwartung "höhere Dimension = schlechtere kNN-Näherung" - widerlegt:** bei gleichverteilten Punkten ist die Näherung ab k = 5 überall exakt, die Schwäche liegt bei kleinem k in niedriger Dimension (Unzusammenhang). **Nicht gemessen:** Daten mit echter Struktur; auch der kNN-Graph selbst wird hier per Brute Force (n² Abstände) berechnet, was das eigentliche Ziel der Näherung (subquadratisch) verfehlt. Eine Triangulierung in d > 2 Dimensionen ist **nicht gebaut** (ihre Größe wächst mit d; Theorie, nicht gemessen).
- **Elementarschritte sind keine Laufzeit:** optimierte Bibliotheken (Qhull, CGAL) bauen die Triangulierung viel schneller als diese Python-Umsetzung; gemessen wird die Schrittzahl, nicht die Sekunden.
- **Die Annahme "euklidisch" ist eng:** mit Geländezuschlag fehlen früh MST-Kanten (bei 0,4 in 52 % der Instanzen), der Kostenschaden bleibt aber klein; wer exakte Bäume braucht, muss dann auf einem größeren Kandidatengraphen arbeiten.
- **Single-Linkage-Chaining:** der Schnitt trifft die Wolken nur bei getrennten Wolken (Reinheit 1,00 bei 2, 0,64 bei 8 Wolken); Ausreißer und Brücken verketten. Andere Linkage-Verfahren: agglomerative-demo.
- **Gitter:** die Triangulierung ist nicht eindeutig (kozirkulare Punkte), Gabriel ist nicht in jeder Triangulierung enthalten; der Schlüssel (Kosten, Kantenindex) liefert trotzdem in allen Wegen denselben Baum.
- **Nicht gebaut:** Divide-and-Conquer- oder Sweepline-Triangulierung (O(n log n) im schlechtesten Fall; hier inkrementell mit Walk), symbolische Punkte im Unendlichen (statt des großen Hilfsdreiecks), parallele EMST-Verfahren, FAMST und Verwandte (nur die Grundidee des kNN-Graphen), Euklidischer MST in hohen Dimensionen exakt; ebenso gerichtete Spannbäume, Nebenbedingungen, Steiner-Bäume, Sensitivität, Kirchhoff.

## Verifikation

- **Delaunay:** Kantenmenge und Hülle gleich `scipy.spatial.Delaunay`/`ConvexHull` über 60 Zufalls-, Cluster-, flache, normalverteilte und fast kozirkulare Instanzen sowie flache Punktwolken mit Seitenverhältnissen bis 10⁶; Umkreise leer, Dreiecke gegen den Uhrzeigersinn mit positiver Fläche, **Euler-Zahlen** (2n − 2 − h Dreiecke, 3n − 3 − h Kanten), Flächensumme gleich der Hülle, jede innere Kante in zwei Dreiecken; Gitter (kozirkular) als gültige Triangulierung; Sonderfälle (drei und vier Punkte konvex und konkav, alle Punkte auf einer Geraden, doppelte Punkte); Ereignisse spielen die Triangulierung nach, Zähler-Identitäten.
- **MST:** alle vier Wege derselbe Baum (auch bei gleichen Abständen), Kosten gegen scipy und Brute-Force (6 Punkte); **MST ⊆ Delaunay** auf allen Instanztypen; **MST ⊆ RNG ⊆ Gabriel ⊆ Delaunay** in allgemeiner Lage, Gabriel- und RNG-Filter gleich der Definition über alle Punktpaare; mit Geländezuschlag ein festgeschriebenes Gegenbeispiel (Kante fehlt, MST auf Delaunay teurer), ohne Zuschlag nie; Kostenzerlegung (Aufbau + Abstände + MST).
- **Single-Linkage:** Verschmelzungshöhen und k-Cluster gleich einem naiven O(n³)-Verfahren. **kNN:** k = n − 1 exakt, Näherungsbaum ein Spannbaum mit Fehler ≥ 0, symmetrischer Graph gleich der Definition.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt**, über dieselben Auswertungsfunktionen wie die App selbst (`ev.run_config`/`ev.sweep`/`ev.crossover`/`ev.terrain_miss`/`ev.chaining`/`ev.highdim`), NIE über ein Ad-hoc-Skript; Schrittzahlen sind ganzzahlig und plattformunabhängig (Delaunay in reiner Python-Arithmetik), Kennzahlen aus Matrixprodukten (kNN) nur mit Rändern; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt und jeder Einfüge-Schritt, alle Instanztypen und Ebenen, Extremwerte, Würfel, Permalink-Grenzen, Instanzwechsel, Experimente und Sweeps auf Abruf, Footer).

Literatur: Shamos, M. I., & Hoey, D. (1975). *Closest-point problems.* 16th Annual Symposium on Foundations of Computer Science, 151-162. Bowyer, A. (1981). *Computing Dirichlet tessellations.* The Computer Journal 24(2), 162-166. Watson, D. F. (1981). *Computing the n-dimensional Delaunay tessellation with application to Voronoi polytopes.* The Computer Journal 24(2), 167-172. Gabriel, K. R., & Sokal, R. R. (1969). *A new statistical approach to geographic variation analysis.* Systematic Zoology 18(3), 259-278. Toussaint, G. T. (1980). *The relative neighbourhood graph of a finite planar set.* Pattern Recognition 12(4), 261-268.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Punkte, vier Schritte, Aufwand, Experimente auf Abruf, Sweeps, Grenzen, Mathe |
| `emst_delaunay.py` | Bowyer-Watson von Hand: Walk, Hohlraum, Ereignisse zum Nachspielen, Zähler |
| `emst_methods.py` | Die vier Wege, Gabriel/RNG, Single-Linkage, kNN-Näherung |
| `emst_algorithm.py`, `emst_unionfind.py` | Prim (drei Umsetzungen), Kruskal, Heap, Sortierung, Union-Find (aus prim-demo) |
| `emst_scenario.py` | Instanzen (gleichverteilt, Cluster, Gitter), Geländefaktor, Kantenlisten |
| `emst_constants.py` | Konstanten, Presets, gemessene Werte |
| `emst_evaluation.py` | Kennzahlen, Sweeps, Kreuzungspunkt, Gelände-, Chaining- und Hochdimensions-Experiment |
| `emst_presets.py`, `emst_visualization.py` | Permalink/Presets, Plotly-Figuren |
| `tests/` | Delaunay-Kette, Wege und Hierarchie, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
