# Vergleichstabelle für Methoden und Frameworks

## Methoden

| Methode | Charakter | Stärken | Grenzen | Rolle im Projekt |
| --- | --- | --- | --- | --- |
| nearest-neighbor-based | Distanz- und Nachbarschaftsmodelle | Einfach, oft interpretierbar, gute Baselines | Sensitiv gegenüber Skalierung und Dimensionalität | LOF passt hier hinein |
| ensemble / isolation-based | Isolation oder Teilmengen-basierte Verfahren | Robust, schnell, gute Standardbaseline | Nicht immer sehr fein in lokalen Strukturen | Isolation Forest, Extended Isolation Forest |
| probabilistic / linear-based | Verteilungs- oder Kovarianzmodellierung | Statistisch sauber, leicht testbar | Nimmt oft stärkere Annahmen an | EllipticEnvelope als Vertreter |
| histogram / density-based | Dichte oder Histogramme pro Feature | Schnell und oft stabil auf tabellarischen Daten | Abhängig von Binning und Skalierung | HBOS, COPOD, ECOD |
| kernel-based | Kerneltrick oder nichtlineare Grenzen | Flexibel, kann komplexe Trennungen lernen | Teurer und tuning-empfindlicher | One-Class SVM |
| neural network-based | Neuronale Rekonstruktion oder Generierung | Potenzial für komplexe Muster | Höherer Aufwand, mehr Datenbedarf | Autoencoder, später AnoGAN |
| meta-learning-based | Lernt aus vorherigen Aufgaben | Kann Suche beschleunigen | Braucht gute Meta-Datenbasis | Spätere Erweiterung |

## Frameworks

| Framework | Fokus | Stärke | Wann sinnvoll | Status im Projekt |
| --- | --- | --- | --- | --- |
| PyOD | Breite Sammlung klassischer Anomalie-Detektoren | Viele Baselines in einem konsistenten API-Design | Wenn du schnell viele Verfahren vergleichen willst | Extern aufbereitet, als Referenz verfügbar |
| LSCP | Ensemble- und Kompositionsansatz für Outlier Detection | Kombination vieler Detektoren | Wenn du Ensemble-Ideen analysieren willst | Extern als Referenz vorbereitet |
| TODS | End-to-End Data Science / AutoML für Anomalieerkennung | Breiter Workflow mit Pipeline-Gedanken | Wenn du ein größeres AutoML-Referenzsystem brauchst | Extern aufbereitet, als Vergleichsframework markiert |
| AutoOD | AutoML-orientierte Anomalieerkennung | Vergleich auf AutoML-Ebene | Wenn du einen stärker automatisierten Referenzpunkt willst | Als Referenzframework geführt |

## Einordnung für das Projekt

- Die Methoden-Tabelle hilft dir bei der Modellwahl innerhalb der Registry.
- Die Framework-Tabelle hilft dir beim Vergleich mit externen Ansätzen.
- PyOD und TODS sind bereits extern vorbereitet und müssen im Repo nicht noch einmal neu aufgebaut werden.