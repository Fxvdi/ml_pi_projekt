# MetaOD auf TEP – Ergebnisse & Erkenntnisse

Zusammenfassung des ersten End-to-End-Laufs von MetaOD auf dem
Tennessee-Eastman-Prozess (Notebook `metaod.ipynb`). Konzept und Setup: siehe
[METAOD.md](METAOD.md).

> **Status:** Vorläufig. Zahlen auf einem Subsample (10 Läufe je Fehlertyp,
> ein Seed, Top-8 der MetaOD-Empfehlungen, Fit-Subsample 10 000 Normalpunkte).
> Die **Tendenzen** sind belastbar, die exakten Werte für die Präsentation mit
> hochskaliertem `N_RUNS_TEST` bestätigen.

## Setup

- **Training:** nur Fault-Free (unsupervised), 52 Sensor-Features, standardisiert
- **MetaOD:** labellose Detektor-Empfehlung im isolierten Legacy-Stack
- **Test:** Fault-Free + Faulty, korrekte Per-Sample-Labels (`sample >= 160`)
- **Schwelle:** aus Trainings-Scores zu Ziel-FAR 1 % (kein Testlabel-Leak)
- **Testset-Anomalieanteil:** ~79 % (fehlerlastig, s. Hinweis unten)

## Ergebnisse (Top-8 der MetaOD-Empfehlung)

| Rang | Modell | PR-AUC | ROC-AUC | Detection Rate | FAR | Laufzeit |
|---:|---|---:|---:|---:|---:|---:|
| 1 | ABOD 5 | 0.967 | 0.872 | 0.547 | 0.001 | 42 s |
| 2 | kNN (5, median) | 0.972 | 0.891 | 0.629 | 0.004 | **6 s** |
| 3 | ABOD 25 | **0.973** | **0.891** | **0.630** | 0.005 | 673 s |
| 4 | ABOD 10 | 0.971 | 0.886 | 0.619 | 0.005 | 110 s |
| 5 | ABOD 20 | 0.972 | 0.891 | 0.628 | 0.005 | 391 s |
| 6 | LOF (100, manhattan) | 0.968 | 0.876 | 0.587 | 0.005 | 36 s |
| 7 | ABOD 3 | 0.957 | 0.837 | 0.000 | 0.000 | 17 s |
| 8 | ABOD 15 | 0.972 | 0.889 | 0.627 | 0.005 | 223 s |

Per-Fault-Stichprobe (ABOD 5, Rang 1):

| Fehler | Detection Rate | FAR | Ø Delay | Charakter |
|---:|---:|---:|---:|---|
| 1 | 0.993 | 0.000 | 5.3 | Sprung – leicht erkennbar |
| 4 | 0.381 | 0.001 | 1.0 | Sprung, aber regelkompensiert |
| 13 | 0.942 | 0.001 | 42.5 | Slow Drift – spät erkannt |

## Wichtigste Erkenntnisse

### 1. MetaOD wählt „gut, aber nicht optimal"
MetaODs Top-1 (ABOD 5) wird von seinem eigenen Rang 2/3 geschlagen (Detection
Rate 0.55 vs 0.63). **Aber** alle Top-Empfehlungen liegen eng beieinander
(PR-AUC 0.957–0.973) – MetaOD wählt keinen *schlechten*, nur einen leicht
suboptimalen Detektor. Ehrliche Antwort auf „kann AutoML Labels ersetzen?":
**fast, aber es lässt Performance liegen.**

### 2. Der Hyperparameter zählt mehr als die Detektor-Familie
ABOD 3 (Detection 0.00) vs. ABOD 25 (0.63) – dieselbe Familie, dramatisch
anderes Ergebnis, nur wegen `n_neighbors`. Das rechtfertigt, dass MetaOD
Hyperparameter mitliefert; die Familie allein würde nicht reichen.

### 3. Einfacher schlägt komplex (Effizienz)
kNN erreicht praktisch die beste PR-AUC (0.972 vs 0.973) bei **100× weniger
Rechenzeit** als ABOD 25 (6 s vs 673 s). Für die Praxis die klar bessere Wahl –
ABOD lohnt den Aufwand hier nicht.

### 4. Es gibt eine Decke der punktbasierten Methoden
Alle punktbasierten Detektoren clustern bei ROC-AUC 0.87–0.89; selbst der beste
fängt bei 1 % FAR nur ~63 % der Fehlersamples. Die verpassten ~37 % sind
großteils die dynamischen Fehler (z. B. 4, 13). **Genau diese Lücke motiviert
die Zeitreihen-Modelle in Phase 3.**

### 5. Praktischer Lesson-Learned: Schwellenübertragung kann kippen
ABOD 3 liefert Detection 0.00, weil die aus Trainings-Scores bestimmte Schwelle
auf Testdaten keine Alarme erzeugt. Die schwellen*unabhängigen* Metriken
(PR-AUC/ROC) bleiben gültig. → In der finalen Pipeline prüfen, ob die Schwelle
überhaupt Alarme auslöst, sonst wirkt ein brauchbarer Detektor künstlich tot.

## Methodische Hinweise

- **Testset-Balance:** Der Anomalieanteil (~79 %) ist invertiert zur Realität
  (20 Fehlerklassen vs. 1 Normalklasse). Das **verzerrt PR-AUC und F1** nach
  oben (triviale Baseline ≈ 0.79). **ROC-AUC, FAR und Per-Fault Detection Rate**
  sind balance-robust und daher die aussagekräftigeren Maße. Für realistischere
  Zahlen normal-lastiger samplen (mehr Fault-Free-Läufe).
- **Laufzeit:** ABOD skaliert schlecht mit `n_neighbors` (ABOD 25 ≈ 11 min).
  Für den vollen Top-k-Lauf `FIT_SAMPLE_SIZE` reduzieren oder auf schnelle
  Detektoren (kNN, LOF, HBOS) fokussieren.

## Story für die Präsentation

> „MetaOD wählt ohne Labels einen guten, aber nicht optimalen Detektor;
> punktbasierte Verfahren stoßen bei ~63 % Detection an eine Decke – und genau
> diese Lücke motiviert die Zeitreihen-Modelle."

## Offene Schritte

- Top-k hochskalieren (`N_RUNS_TEST`) für stabile Endzahlen
- **Oracle-Vergleich:** Abstand MetaOD-Wahl ↔ bester Detektor der Liste
  quantifizieren (separate EM/MV-/Oracle-Arbeit, Merge über die CSVs in
  `artifacts/`)
- Phase 3: Zeitreihen-Modelle gegen die punktbasierte Decke
