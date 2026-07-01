# AutoML für Anomalieerkennung

## Ziel
Ein modulares AutoML-System für Anomalieerkennung auf TEP-Daten, das später mehrere Detektor-Familien, Suchstrategien und Betriebsmodi austauschbar unterstützt.

## Aktueller Fokus
- Start im Batch-Modus
- Primär unüberwachte Anomalieerkennung
- Labels zunächst nur für Evaluation und Benchmarking
- Später erweiterbar auf semi-supervised und Streaming

## Rolle des data-Ordners
- Enthält die vier TEP-Parquet-Splits
- Stellt nur Laden, Zusammenführen und leichte Normalisierung bereit
- Macht keine Modellierung, Suche oder Bewertung
- Liefert für Training die fehlerfreien Trainingsdaten
- Liefert für Evaluation die zusammengeführten Testdaten mit Labeln

## Zentrale Herausforderungen
- search space: Menge aller Modell- und Hyperparameter-Kombinationen
- cold-start: kein Vorwissen für neue Datensätze
- running time: Laufzeitbudget für Suche und Training
- high-dimensional data: viele Merkmale pro Beispiel
- scalability: Verhalten bei mehr Daten, Features oder Modellen
- evaluation metrics: geeignete Metriken für stark unausgewogene Daten
- data streams: späterer Betrieb auf fortlaufenden Daten

## Such- und Optimierungsverfahren
- ParamILS: Konfigurationssuche über lokale Verbesserungen
- SMAC: modellbasierte Suche mit Surrogatmodell
- GGA: genetischer Suchansatz für Konfigurationen
- Irace: iteratives Selektionsverfahren für Kandidaten
- region-based method: Suche in vielversprechenden Teilräumen
- Meta-Learning: gute Startpunkte aus früheren Datasets ableiten

## Detektor-Familien für die Baseline

### Unsupervised Anomaly Detection
1. proximity/nearest neighbor-based methods: Anomalien liegen weit von Nachbarn entfernt
2. probabilistic/linear-based methods: Anomalien passen schlecht zu einer angenommenen Verteilung oder linearen Struktur
3. ensemble/isolation-based methods: Anomalien werden durch Ensembles oder Isolationsprinzipien gefunden

### Neuronale Verfahren
- neural network-based anomaly detection: Deep-Learning-Ansätze für komplexe Muster
- AnoGAN: GAN-basiertes Verfahren zur Anomalieerkennung

### Meta-Learning-basierte Ansätze
- MetaOD: lernt, welche Detektoren für neue Daten gut passen könnten
- Meta-AAD: meta-gestützte Auswahl oder Anpassung von Anomalie-Detektoren

## Semi-supervised AutoML
- auto-SSL: AutoML für semi-supervised learning
- SVM: klassischer Klassifikator als Baustein für semi-supervised Varianten
- auto-sklearn: allgemeines AutoML-Framework, nützlich als Referenz
- Class Mass Normalization (CMN): Korrektur von Klassenverteilungen
- Transductive SVM: nutzt unbeschriftete Daten direkt in der Optimierung

## Vergleichs- und Baseline-Frameworks
- PyOD: zentrale Baseline und Vergleichsframework für Outlier Detection
- LSCP: Ensemble-Verfahren mit lokal passender Detektorauswahl
- TODS: Framework für Time-Series-Outlier-Detection
- AutoOD: AutoML-Ansatz speziell für Outlier Detection

## Vorschlag für die Implementierung
1. Den data-Ordner als reine Daten-Schnittstelle verwenden
2. Ein einheitliches Dataset-Objekt für TEP-Daten bauen
3. Eine gemeinsame Detektor-Schnittstelle definieren
4. Eine Registry für austauschbare Detektoren anlegen
5. Eine separate Search-API für SMAC, Irace, Random Search oder Regeln bauen
6. Ein Evaluationsmodul mit PR-AUC, F1, ROC-AUC und Laufzeit ergänzen
7. Die erste Baseline mit PyOD umsetzen
8. Danach weitere Detektoren und Suchstrategien einstecken

## Ordnerstruktur
- `automl/` - Kernpaket
- `automl/data/` - Parquet-Lader, Split-Handling und Datencontainer
- `automl/detectors/` - austauschbare Detektoren und Wrapper
- `automl/search/` - Suchstrategien und Konfigurationslogik
- `automl/evaluation/` - Metriken und Laufzeitmessung

## Nächste Fragestellungen
- Soll der erste Suchraum nur Hyperparameter enthalten oder auch ganze Modellfamilien?
- Soll die erste Version nur Batch unterstützen oder direkt schon Streaming-Hooks enthalten?