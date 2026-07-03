# MetaOD & Unsupervised Modellselektion

## Motivation: Praxisnahe Simulation

In der Prozessindustrie sind **gelabelte Anomaliedaten teuer**. Ein Sensor-
ausfall oder ein Fehlertyp muss erst aufgetreten und dokumentiert worden sein,
bevor ein Label existiert. Normaldaten hingegen sind im laufenden Betrieb
jederzeit verfügbar.

Dieses Projekt simuliert genau dieses Szenario:

- **Training:** nur Normaldaten, keine Labels
- **Modellselektion:** ohne Labels — wie würde ein Unternehmen entscheiden?
- **Finale Evaluation:** Labels werden *nur zum Vergleich der Selektionsstrategien*
  verwendet, nicht zur Modellwahl selbst

---

## Das Selektionsproblem

Nach dem Training mehrerer PyOD-Detektoren stellt sich die Frage: **Welches
Modell nehmen wir — ohne Labels?**

Wir vergleichen drei Strategien:

```
Mehrere PyOD-Detektoren trainiert
            │
            ├──► MetaOD          → Meta-Learning-Empfehlung
            ├──► EM / MV         → Intrinsische Metriken ohne Labels
            │
            ▼
    Modell ausgewählt (ohne Labels)
            │
            ▼
    Finale Bewertung mit Labels   ← nur hier tauchen Labels auf
            │
            ▼
    Oracle (Labels zur Selektion) → theoretische Obergrenze
```

---

## Die drei Ansätze

### 1. MetaOD — Meta-Learning-Empfehlung

MetaOD [1] empfiehlt Detektoren per **kollaborativem Filtern** über eine große
Benchmark-Matrix (viele Datensätze × viele Detektoren × gemessene Performance):

1. Meta-Features des Datensatzes extrahieren (Dimensionalität, Schiefe, ...)
2. Ähnlichkeit zu historisch bekannten Datensätzen messen
3. Detektor-Ranking ableiten — ohne einen einzigen Label zu benötigen

**Vorteil:** kein manuelles Tuning, kein Domänenwissen nötig.  
**Nachteil:** Empfehlung basiert auf externen Benchmarks, nicht auf den
eigenen Daten.

### 2. Excess Mass (EM) & Mass Volume (MV) — Intrinsische Metriken

Goix et al. [2] schlagen zwei komplementäre Metriken vor, die **nur die
Daten selbst** brauchen:

**Excess Mass (EM)**
- Misst, ob der Detektor hohe Scores in Regionen *niedriger Datendichte* vergibt
- Intuition: Anomalien liegen in dünn besiedelten Regionen — ein guter Score
  sollte dort hoch sein
- Höher = besser

**Mass Volume (MV)**
- Misst das Volumen der als anomal markierten Region relativ zur abgedeckten
  Datenmasse
- Intuition: ein guter Detektor markiert eine *kleine Region*, die aber *viel
  Masse* der Anomalien enthält
- Niedriger = besser (kleinere Fläche bei gleicher Masse)

Beide Metriken sind schwellenunabhängig und benötigen keine Labels.

### 3. Oracle — theoretische Obergrenze

Der Oracle wählt rückwirkend den Detektor mit dem höchsten **PR-AUC** auf
den gelabelten Testdaten. Er ist in der Praxis unerreichbar, dient aber als
Referenz:

> *"Wie nah kommen MetaOD und EM/MV an den Oracle heran?"*

Das ist die zentrale Forschungsfrage dieses Projekts.

---

## Evaluationsrahmen

| Selektionsstrategie | Labels bei Selektion? | Praxistauglich? |
|---|---|---|
| **MetaOD** | Nein | Ja |
| **EM / MV** | Nein | Ja |
| **Oracle** | Ja | Nein (Obergrenze) |

Nach der Modellauswahl wird jede Strategie mit den echten Labels bewertet:
PR-AUC, Detection Rate, FAR, Detection Delay pro Fehlertyp (siehe
`automl/evaluation/`).

---

## Technische Umsetzung (Legacy-Stack)

MetaODs vortrainierte Modelle (`trained_models/`) wurden mit **scikit-learn
0.22.1** gepickelt. Der moderne Projekt-Stack (Python 3.13, sklearn 1.9,
numpy 2.x) kann sie nicht laden: das C-Struct-Layout der gepickelten
RandomForest-Trees hat sich ab sklearn 1.3 geändert (`missing_go_to_left`),
und der MinMaxScaler aus 0.22.1 kennt das ab 1.0 erwartete `clip`-Attribut nicht.

**Lösung:** MetaOD läuft isoliert in einem alten Stack, statt den ganzen
Projekt-Stack herabzustufen.

- [`scripts/run_metaod_selection.py`](../scripts/run_metaod_selection.py) deklariert
  seinen Legacy-Stack (Python 3.10, sklearn 1.0.2, numpy 1.23) über
  **PEP-723-Inline-Metadaten**; `uv run` baut die isolierte Umgebung automatisch.
- Die MetaOD-Quellen sind unter [`external/metaod/`](../external/metaod) **vendored**
  (reines Python, BSD), damit der isolierte Lauf nicht das moderne sklearn zieht.
- Ablauf: Notebook speichert den skalierten Meta-Sample als `.npy` →
  Subprozess-Lauf → Ranking als JSON zurück ins Notebook.

**`trained_models/` beschaffen** (gitignored, ~14 MB) — einmalig aus dem
MetaOD-Repo laden und ins Projekt-Root entpacken:

```bash
curl -L -o trained_models.zip \
  https://raw.githubusercontent.com/yzhao062/MetaOD/master/saved_models/trained_models.zip
unzip trained_models.zip && rm trained_models.zip
```

> **Hinweis Reproduzierbarkeit:** MetaODs Meta-Features nutzen intern
> randomisierte Detektoren (IForest, LODA); die Ranking-Reihenfolge kann leicht
> schwanken. Die Top-Empfehlungen (auf TEP: ABOD, kNN, LOF) sind stabil.

---

## Einordnung in den Phasenplan

| Phase | Inhalt |
|---|---|
| Phase 1 ✅ | Baselines: PCA-T²/SPE als Literatur-Referenz |
| Phase 2 | PyOD-Detektoren trainieren + Selektion via MetaOD / EM / MV / Oracle |
| Phase 3 | Zeitreihen-Modelle (LSTM-AE, DPCA) + Gesamtvergleich |

---

## Quellen

[1] Y. Zhao, R. A. Rossi, L. Akoglu: *Automatic Unsupervised Outlier Model
Selection.* NeurIPS, 2021.

[2] N. Goix, N. Drougard, R. Briot, M. Chiapino: *How to Evaluate the Quality
of Unsupervised Anomaly Detection Algorithms?* ICML Anomaly Detection Workshop,
2016.
