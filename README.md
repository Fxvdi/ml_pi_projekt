- [Google Dokumentation (Docs & PowerPoint)](https://docs.google.com/document/d/1xCzNuHb4aJFeL6dh9fqxzVupTzS7ex81C6WqlYu80aE/edit?usp=sharing)
- [Daten](https://drive.google.com/drive/folders/1p-McaULO5VP9qN01V3a_34H8y4cIHShN?usp=sharing)

# Konzept

Ziel des Projekts ist ein modulares AutoML-System für Anomalieerkennung auf dem TEP-Datensatz. Die Pipeline soll so aufgebaut sein, dass Daten, Modelle, Suchstrategien und Bewertung klar voneinander getrennt sind. Dadurch lassen sich einzelne Bausteine später austauschen, ohne das gesamte System umzubauen.

## Grundidee

Die erste Version startet im unsupervised Setting:

1. Die TEP-Parquet-Dateien werden aus dem `automl/data/`-Ordner geladen.
2. Als Training werden nur die fehlerfreien Trainingsdaten verwendet.
3. Eine Suchstrategie erzeugt Kandidaten für Modelle und Parameter.
4. Ein Detektor wird aus einer Registry erzeugt und auf den Trainingsdaten gelernt.
5. Der Detektor gibt Anomalie-Scores für die Testdaten aus.
6. Die Evaluation berechnet Metriken wie PR-AUC, F1 und Laufzeit.
7. Das beste Setup wird ausgewählt und kann im nächsten Schritt weiterverwendet werden.

Die fehlerhaften Trainingsdaten werden in der ersten Version nicht für das Training genutzt. Sie bleiben aber für spätere semi-supervised Ansätze oder zusätzliche Experimente verfügbar.

## Austauschbare Bausteine

### Methoden

Methoden beschreiben den allgemeinen Ansatz der Anomalieerkennung.

Beispiele:
- nearest-neighbor-based
- probabilistic / linear-based
- ensemble / isolation-based
- neural network-based
- meta-learning-based

Ein Methodenwechsel bedeutet zum Beispiel:
- zuerst ein Distanzverfahren
- später ein Ensemble-Verfahren
- später ein neuronales Verfahren

### Modelle

Modelle sind die konkreten Implementierungen innerhalb einer Methode.

Beispiele:
- LOF für nearest-neighbor-based detection
- Isolation Forest für ensemble / isolation-based detection
- One-Class SVM als klassisches Anomalie-Modell
- Autoencoder als neuronales Modell
- AnoGAN als deep-learning-basierter Ansatz

Ein Modellwechsel bedeutet zum Beispiel:
- von Isolation Forest zu LOF
- von LOF zu One-Class SVM
- von einem klassischen Modell zu einem Autoencoder

### Strategien

Strategien entscheiden, wie Modelle und Parameter ausgewählt werden.

Beispiele:
- Random Search als einfacher Einstieg
- SMAC für modellbasierte Suche
- Irace für iteratives Aussortieren schlechter Kandidaten
- ParamILS für lokale Optimierung
- Meta-Learning für vorgeschlagene Startkonfigurationen

Ein Strategiewechsel bedeutet zum Beispiel:
- zuerst Random Search
- später SMAC
- später Meta-Learning als Initialisierung

## Beispiel für Austauschbarkeit

Die Pipeline bleibt gleich, auch wenn einzelne Teile ersetzt werden:

- Methode ändern: nearest neighbor -> ensemble
- Modell ändern: LOF -> Isolation Forest
- Strategie ändern: Random Search -> SMAC
- Metrik ändern: ROC-AUC -> PR-AUC

Dadurch kann das System verschiedene Kombinationen systematisch vergleichen, ohne dass der restliche Code angepasst werden muss.

## Ordneridee

- `automl/data/` lädt und organisiert die TEP-Daten
- `automl/detectors/` enthält austauschbare Detektoren und Adapter
- `automl/search/` enthält Suchstrategien und Kandidatenerzeugung
- `automl/evaluation/` berechnet Metriken und Laufzeiten
- `automl/registry.py` verbindet Namen mit konkreten Modellen
- `automl/pipeline.py` verbindet alles zu einem Ablauf

## Erste Version

Die erste Version des Projekts soll:
- nur Batch-Daten verarbeiten
- unüberwacht trainieren
- auf dem fehlerfreien Training lernen
- auf den Testdaten bewerten
- PyOD erst ganz am Ende als Referenz einbauen

## Aktueller Startpunkt

Der erste lauffähige Workflow liegt in `automl.pipeline.run_minimal_workflow(data_dir)`.

Am einfachsten startest du ihn jetzt so:

```powershell
python run_automl.py
```

Mit einem anderen Detektor:

```powershell
python run_automl.py --detector local_outlier_factor
```

Mit Random Search:

```powershell
python run_automl.py --strategy search
```

Beispiel:

```python
from automl.pipeline import run_minimal_workflow

result = run_minimal_workflow("automl/data")
print(result)
```

Im aktuellen Stand werden zwei austauschbare Detektoren verglichen:
- `isolation_forest`
- `local_outlier_factor`
- `one_class_svm`

Wenn du gezielt vergleichen willst:

```powershell
python run_automl.py --compare isolation_forest local_outlier_factor one_class_svm
```

Random Search über ausgewählte Detektoren:

```powershell
python run_automl.py --strategy search --compare isolation_forest local_outlier_factor one_class_svm
```

Die Registry in `automl/registry.py` erzeugt diese Modelle. Der Minimal-Workflow nutzt standardmäßig `isolation_forest`, damit der Start schnell bleibt. Wenn du `--compare` ohne weitere Namen aufrufst, werden automatisch alle registrierten Detektoren verglichen. Wenn du nur eine Teilmenge willst, kannst du die Namen direkt angeben.

Beispiel für einen Vergleich in Python:

```python
from automl.pipeline import run_comparison_workflow

result = run_comparison_workflow(
	"automl/data",
	["isolation_forest", "local_outlier_factor", "one_class_svm"],
)
print(result)
```

Beispiel für alle Detektoren im Terminal:

```powershell
python run_automl.py --compare
```

Beispiel für einen einzelnen Detektor im Terminal:

```powershell
python run_automl.py --detector local_outlier_factor
```

## Spätere Erweiterungen

Später kann das System erweitert werden um:
- semi-supervised learning
- Streaming oder Online-Detection
- Meta-Learning
- weitere Search-Strategien
- weitere Frameworks wie PyOD oder AutoOD

## Vergleichstabelle

Eine erste Übersicht über Methoden und externe Frameworks liegt in [Markdown/Comparison_Table.md](Markdown/Comparison_Table.md).


