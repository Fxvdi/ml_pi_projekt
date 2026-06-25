# Checklist

## Ideen aus Idea.md übernehmen

- [ ] Neue Themen aus [Idea.md](Idea.md) eintragen
- [ ] Relevante Punkte als implementiert / offen markieren
- [ ] Neue Modelle, Strategien oder Frameworks in die passenden Abschnitte einsortieren

## Aktueller Stand

- [x] Ziel des Projekts festgelegt: AutoML für Anomalieerkennung auf dem TEP-Datensatz
- [x] Fokus auf modularen Aufbau mit austauschbaren Bausteinen definiert
- [x] Erste Version auf unsupervised learning ausgerichtet
- [x] Batch-Verarbeitung als Startpunkt festgelegt
- [x] Parquet-Dateien nach `automl/data/` verschoben
- [x] TEP-Splits als Ladefunktion implementiert
- [x] Ein erster Minimal-Workflow für unsupervised learning läuft
- [x] Registry für austauschbare Detektoren eingebaut
- [x] Vergleichsworkflow für mehrere Detektoren eingebaut
- [x] Suchstrategien über CLI auswählbar gemacht
- [x] Ergebnisdarstellung für Runs lesbar formatiert
- [x] Vergleichsbericht für Benchmark-Suiten erzeugt
- [x] Externe Aufbereitung für PyOD und TODS liegt bereits vor
- [x] Vergleichstabelle für Methoden und Frameworks erstellt
- [x] README um Ziel, Ablauf und Ergebnis ergänzt

## Implementiert

- [x] Isolation Forest als erste Baseline implementiert
- [x] Local Outlier Factor als zweites Vergleichsmodell implementiert
- [x] One-Class SVM als dritte Baseline implementiert
- [x] EllipticEnvelope als vierte Baseline implementiert
- [x] HBOS als fünfte Baseline implementiert
- [x] COPOD als sechste Baseline implementiert
- [x] ECOD als siebte Baseline implementiert
- [x] Random Search als erste Parametersuche angebunden
- [x] TEP-Daten werden als vier Splits geladen: Training / Testing, fault-free / faulty
- [x] Feasible unsupervised training auf `train_fault_free`
- [x] Evaluation auf kombiniertem Testsplit mit Labels
- [x] Metriken: PR-AUC, ROC-AUC und F1
- [x] Laufzeitmessung im Evaluationslauf
- [x] Registry-basierte Modellauswahl
- [x] Default-Workflow bleibt schnell und nutzt nur einen Detektor
- [x] Vergleichsworkflow kann mehrere Detektoren gezielt ausführen
- [x] Random Search als Suchstrategie für Parameter und Detektoren eingebaut
- [x] Successive Halving als Budget-Strategie implementiert
- [x] Hyperband als Budget-Strategie implementiert
- [x] Parameter-Budget pro Suchstufe ergänzt
- [x] Vollsuchmodus über alle registrierten Detektoren mit Strategieauswahl möglich
- [x] Ein kleines CLI für wiederholbare Runs gebaut
- [x] PyOD-Adapter für Benchmarking entworfen

## Noch nicht implementiert

- [x] PyOD als Referenz ganz am Ende integrieren (extern aufbereitet)
- [ ] Autoencoder als neuronale Baseline prüfen
- [ ] LODA als schnelle Baseline prüfen
- [ ] Extended Isolation Forest als Ensemble-Variante prüfen
- [ ] SMAC als Suchstrategie implementieren
- [ ] Irace als Suchstrategie implementieren
- [ ] Bayesian Optimization als Suchstrategie ergänzen
- [ ] Optuna als Suchstrategie für Hyperparameter-Optimierung einbauen
- [ ] NSGA-II als multiobjective Suchstrategie prüfen
- [ ] MOGA-FS als Feature-Selection-Ansatz prüfen
- [ ] RLNAS als neural architecture search Ansatz prüfen
- [ ] ASAD als Auto-Selective Anomaly Detection Ansatz prüfen
- [ ] Meta-Learning als Initialisierung einsetzen
- [ ] Semi-supervised learning vorbereiten
- [ ] Streaming / Online-Detection vorbereiten
- [x] Ergebnisse systematisch speichern
- [x] Ein kleines CLI oder Skript für wiederholbare Runs gebaut

## Forschungsstand, noch offen in der Implementierung

- [x] ParamILS als mögliche Suchstrategie bekannt
- [x] SMAC als mögliche Suchstrategie bekannt
- [x] GGA als möglicher Suchansatz bekannt
- [x] Successive Halving als Budget-Strategie bekannt
- [x] Hyperband als Budget-Strategie bekannt
- [x] Irace als mögliche Suchstrategie bekannt
- [x] Bayesian Optimization als mögliche Suchstrategie bekannt
- [x] NSGA-II als mögliche multiobjective Suchstrategie bekannt
- [x] MOGA-FS als möglicher Feature-Selection-Ansatz bekannt
- [x] RLNAS als möglicher neural architecture search Ansatz bekannt
- [x] ASAD als möglicher Auto-Selective Anomaly Detection Ansatz bekannt
- [x] Meta-Learning als möglicher Ansatz bekannt
- [x] AnoGAN als mögliche neuronale Methode bekannt
- [x] PyOD als Referenzframework bekannt
- [x] LSCP als Vergleichsframework bekannt
- [x] TODS als Vergleichsframework bekannt (extern aufbereitet)
- [x] AutoOD als Vergleichsframework bekannt
- [x] LODA als mögliche Baseline bekannt
- [x] HBOS als mögliche Baseline bekannt
- [x] COPOD als mögliche Baseline bekannt
- [x] ECOD als mögliche Baseline bekannt
- [x] Extended Isolation Forest als mögliche Baseline bekannt

## Priorisierung

### Priorität 1: Vergleichsgrundlage sauber machen

- [ ] Saubere Train/Validation/Test-Trennung für alle Vergleiche einziehen
- [x] PyOD-Adapter oder PyOD-Benchmark-Harness direkt in die Pipeline hängen
- [ ] Einheitliche Vorverarbeitung und Threshold-Logik für alle Modelle festziehen
- [x] Vergleichsläufe mit mehreren Seeds und persistierten Ergebnissen etablieren

### Benchmark-Regeln

- [ ] Hauptmetrik festlegen, aktuell PR-AUC
- [ ] Sekundärmetriken festlegen: ROC-AUC, F1 und Laufzeit
- [x] Gleiche Splits für AutoML und PyOD verwenden
- [ ] Validation nur für Modell- und Strategieauswahl nutzen
- [ ] Testsplit nur für die finale Berichterstattung verwenden
- [ ] Gleiche Vorverarbeitung und gleiche Threshold-Logik für alle Modelle anwenden
- [x] Mehrere Seeds pro Experiment fahren und Mittelwert plus Streuung berichten
- [x] Alle Läufe mit Modellname, Parametern, Seed und Metriken persistieren
- [ ] Gleiche Budget- und Laufzeitgrenzen für AutoML und PyOD festlegen
- [x] Einheitliche Evaluation definieren

### Split-Plan für TEP

- [x] Train nur auf `train_fault_free`
- [x] Validation aus einem Anteil von `train_fault_free` bilden
- [ ] `train_faulty` nur für spätere explorative Analysen oder zusätzliche Experimente nutzen
- [ ] Test aus `test_fault_free` und `test_faulty` zusammensetzen
- [x] Validation für Suchstrategien und Modellwahl nutzen
- [x] Test nur für die finale Berichterstattung verwenden
- [x] Split-Ziehung reproduzierbar machen über festen Seed
- [x] Split-Informationen zusammen mit den Ergebnissen speichern
- [x] Gleiche Split-Definition für AutoML und PyOD verwenden

### Persistenz-Plan

- [x] Ergebnisse als JSON und JSONL speicherbar machen
- [x] Benchmark-Metadaten mit Zeitstempel, Datenpfad und Strategie mit speichern
- [x] Ergebnisdarstellung und Persistenz auf dieselbe AutoMLResult-Struktur stützen
- [x] Split-Informationen in gespeicherten Ergebnissen mit ablegen
- [x] Seed-Informationen in gespeicherten Ergebnissen mit ablegen
- [x] Mehrere Runs in einer gemeinsamen History-Datei sammelbar machen

### Priorität 2: Baseline-Abdeckung vervollständigen

- [ ] Autoencoder als neuronale Baseline prüfen
- [ ] LODA als schnelle Baseline prüfen
- [ ] Extended Isolation Forest als Ensemble-Variante prüfen

### Priorität 3: Suchstrategien für den AutoML-Vergleich

- [ ] Optuna als nächstes Suchframework evaluieren
- [ ] Bayesian Optimization als Suchstrategie ergänzen
- [ ] SMAC als Suchstrategie implementieren
- [ ] Irace als Suchstrategie implementieren
- [ ] NSGA-II als multiobjective Suchstrategie prüfen

### Priorität 4: Erweiterte Forschungsoptionen

- [ ] MOGA-FS als Feature-Selection-Ansatz prüfen
- [ ] RLNAS als neural architecture search Ansatz prüfen
- [ ] ASAD als Auto-Selective Anomaly Detection Ansatz prüfen
- [ ] Meta-Learning als Initialisierung einsetzen
- [ ] Semi-supervised learning vorbereiten
- [ ] Streaming / Online-Detection vorbereiten

## Nächste Arbeitspakete

- [ ] AnoGAN als späteres Deep-Learning-Modell prüfen
- [x] TODS als Vergleichsframework extern aufbereitet
- [ ] Vergleichstabelle für Methoden und Frameworks erstellen