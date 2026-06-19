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
- [x] README um Konzept und Workflow ergänzt

## Implementiert

- [x] Isolation Forest als erste Baseline implementiert
- [x] Local Outlier Factor als zweites Vergleichsmodell implementiert
- [x] One-Class SVM als dritte Baseline implementiert
- [x] Random Search als erste Parametersuche angebunden
- [x] TEP-Daten werden als vier Splits geladen: Training / Testing, fault-free / faulty
- [x] Feasible unsupervised training auf `train_fault_free`
- [x] Evaluation auf kombiniertem Testsplit mit Labels
- [x] Metriken: PR-AUC, ROC-AUC und F1
- [x] Laufzeitmessung im Evaluationslauf
- [x] Registry-basierte Modellauswahl
- [x] Default-Workflow bleibt schnell und nutzt nur einen Detektor
- [x] Vergleichsworkflow kann mehrere Detektoren gezielt ausführen
- [x] Ein kleines CLI für wiederholbare Runs gebaut

## Noch nicht implementiert

- [ ] PyOD als Referenz ganz am Ende integrieren
- [ ] Einen dritten Detektor ergänzen
- [ ] EllipticEnvelope als leichtere weitere Baseline prüfen
- [ ] Autoencoder als neuronale Baseline prüfen
- [ ] LODA als schnelle Baseline prüfen
- [ ] HBOS als schnelle Baseline prüfen
- [ ] COPOD als robuste Baseline aus dem PyOD-Umfeld prüfen
- [ ] ECOD als robuste Baseline aus dem PyOD-Umfeld prüfen
- [ ] Extended Isolation Forest als Ensemble-Variante prüfen
- [ ] Successive Halving als Budget-Strategie implementieren
- [ ] Hyperband als Budget-Strategie ergänzen
- [ ] Vollsuchmodus für alle Detektoren mit Strategie implementieren
- [ ] SMAC als Suchstrategie implementieren
- [ ] Irace als Suchstrategie implementieren
- [ ] Bayesian Optimization als Suchstrategie ergänzen
- [ ] NSGA-II als multiobjective Suchstrategie prüfen
- [ ] MOGA-FS als Feature-Selection-Ansatz prüfen
- [ ] Meta-Learning als Initialisierung einsetzen
- [ ] Semi-supervised learning vorbereiten
- [ ] Streaming / Online-Detection vorbereiten
- [ ] Ergebnisse systematisch speichern
- [ ] Ein kleines CLI oder Skript für wiederholbare Runs bauen
- [ ] EllipticEnvelope als leichtere weitere Baseline prüfen
- [ ] Autoencoder als neuronale Baseline prüfen

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
- [x] Meta-Learning als möglicher Ansatz bekannt
- [x] AnoGAN als mögliche neuronale Methode bekannt
- [x] PyOD als Referenzframework bekannt
- [x] LSCP als Vergleichsframework bekannt
- [x] TODS als Vergleichsframework bekannt
- [x] AutoOD als Vergleichsframework bekannt
- [x] LODA als mögliche Baseline bekannt
- [x] HBOS als mögliche Baseline bekannt
- [x] COPOD als mögliche Baseline bekannt
- [x] ECOD als mögliche Baseline bekannt
- [x] Extended Isolation Forest als mögliche Baseline bekannt

## Nächste Arbeitspakete

- [ ] AnoGAN als späteres Deep-Learning-Modell prüfen
- [ ] TODS als Vergleichsframework genauer untersuchen
- [ ] Vergleichstabelle für Methoden und Frameworks erstellen