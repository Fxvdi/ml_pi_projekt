- [Google Dokumentation (Docs & PowerPoint)](https://docs.google.com/document/d/1xCzNuHb4aJFeL6dh9fqxzVupTzS7ex81C6WqlYu80aE/edit?usp=sharing)
- [Daten](https://drive.google.com/drive/folders/1p-McaULO5VP9qN01V3a_34H8y4cIHShN?usp=sharing)

# Konzept

Ziel des Projekts ist ein modulares AutoML-System für Anomalieerkennung auf dem TEP-Datensatz. Die Pipeline trennt Daten, Modelle, Suchstrategien, Evaluation und Berichtserzeugung klar voneinander. Dadurch lassen sich Baselines, Suchverfahren und externe Referenzframeworks austauschen, ohne den restlichen Ablauf umzubauen.

Am Ende soll ein fairer, reproduzierbarer Vergleich entstehen, der zeigt:
- welche Detektoren auf TEP am besten funktionieren
- wie sich Suchstrategien wie Random Search, Successive Halving und Hyperband unterscheiden
- wie sich die eigene Pipeline gegen Referenzmodelle wie PyOD verhält
- welche Konfiguration für das finale Ergebnis ausgewählt wird

Das Ergebnis eines Laufs ist deshalb nicht nur ein einzelnes Modell, sondern ein nachvollziehbarer Benchmark mit Metriken, Parametern, Seed, Split-Informationen und einem lesbaren Vergleichsbericht.

Ein zweites Hauptziel ist der faire Vergleich mit PyOD. Dazu wird die eigene AutoML-Pipeline unter denselben Split-Bedingungen, denselben Ressourcen- und Seed-Vorgaben und mit derselben Auswertungslogik gegen ausgewählte PyOD-Modelle getestet. Da es sich um ein unsupervised Setting handelt, werden Modelle nicht über Labels im Training verglichen, sondern über ihre Anomalie-Scores und die daraus berechneten Kennzahlen auf einem getrennten, gelabelten Testsplit. Validation dient dabei nur der Modell- und Strategiewahl, der Testsplit ausschließlich der finalen Berichterstattung.

## Grundidee

Die aktuelle Version startet im unsupervised Setting und läuft in vier Schritten:

1. Die TEP-Parquet-Dateien werden aus dem `automl/data/`-Ordner geladen.
2. Aus `train_fault_free` wird ein reproduzierbarer Train/Validation-Split gebildet.
3. Eine Registry erzeugt den gewünschten Detektor oder eine ganze Vergleichsmenge.
4. Die gewählte Strategie trainiert auf dem Trainsplit, bewertet auf Validation und refittet das beste Setup für die finale Testbewertung.

Der Testsplit wird nur für die finale Berichterstattung verwendet. Validation dient ausschließlich der Modellauswahl und Strategieauswahl.

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
- auf Validation auswählen und auf Test final bewerten
- Ergebnisse, Seed und Split-Informationen persistent speichern
- PyOD als optionale Referenz direkt mitvergleichbar machen

## Was der Run liefert

Ein Lauf erzeugt am Ende:
- ein ausgewähltes Modell mit Parametern
- Kennzahlen wie PR-AUC, ROC-AUC, F1 und Laufzeit
- Metadaten zu Seed, Datenpfad, Split-Plan und Strategie
- auf Wunsch eine JSON- oder JSONL-Datei für die Historie
- auf Wunsch einen Markdown-Vergleichsbericht oder einen aggregierten Bericht über mehrere Seeds

## Aktueller Startpunkt

Der erste lauffähige Workflow liegt in `automl.pipeline.run_minimal_workflow(data_dir)`. Für vollständige Vergleiche gibt es außerdem `BenchmarkRunner` und die Berichtsfunktionen in `automl.reporting`.

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
- `elliptic_envelope`
- `hbos`
- `copod`
- `ecod`
- optional zusätzlich PyOD-Varianten über `--registry pyod` oder `--registry all`

Wenn du gezielt vergleichen willst:

```powershell
python run_automl.py --compare isolation_forest local_outlier_factor one_class_svm
```

Random Search über ausgewählte Detektoren:

```powershell
python run_automl.py --strategy search --compare isolation_forest local_outlier_factor one_class_svm
```

Die Registry in `automl/registry.py` erzeugt diese Modelle. Der Minimal-Workflow nutzt standardmäßig `isolation_forest`, damit der Start schnell bleibt. Wenn du `--compare` ohne weitere Namen aufrufst, werden automatisch alle registrierten Detektoren verglichen. Wenn du nur eine Teilmenge willst, kannst du die Namen direkt angeben.

Für wiederholte Experimente mit mehreren Seeds und Auswertungen auf mehreren Registries ist `automl.benchmark.BenchmarkRunner` der zentrale Einstiegspunkt. Daraus entstehen dann ein einzelnes JSONL-Protokoll und ein Markdown-Bericht mit Einzel- und Aggregatwerten.

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


