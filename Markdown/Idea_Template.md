# Idea

Nutze diese Datei, wenn du mir neue Themen in Stichpunkten geben willst.
Ich lese die Punkte direkt und überführe sie in [Checklist.md](Checklist.md).

## So kannst du mir Themen schicken

Schreibe pro Thema einfach kurze Stichpunkte in diesem Format:

```md
- Thema: Local Outlier Factor
- Typ: Modell
- Status: recherchiert, relevant, implementiert
- Kurz: Dichtebasierter Outlier-Detektor
- Passt zu: unsupervised, Baseline
- Vergleich: Isolation Forest
- Metrik: PR-AUC, F1
- Nächster Schritt: in Checklist.md aufnehmen
```

## Erlaubte Typen

- Methode
- Modell
- Strategie
- Framework
- Metrik
- Daten
- Preprocessing
- Sonstiges

## Erlaubte Statuswerte

- recherchiert
- relevant
- in Checklist.md aufnehmen
- implementiert
- offen

## Minimalformat für neue Einträge

Wenn du es noch knapper willst, reichen diese vier Zeilen:

```md
- Thema: <Name>
- Typ: <Typ>
- Status: <Status>
- Kurz: <1 kurzer Satz>
```

## Wie ich daraus die Checkliste erweitere

- Wenn Status = recherchiert, kommt es in den Forschungsstand.
- Wenn Status = relevant oder in Checklist.md aufnehmen, kommt es in die offenen oder nächsten Aufgaben.
- Wenn Status = implementiert, markiere ich es in der Implementiert-Sektion als erledigt.
- Wenn mehrere Themen kommen, sortiere ich sie nach Typ in die passende Sektion ein.

## Beispiel für mehrere Stichpunkte

```md
- Thema: SMAC
- Typ: Strategie
- Status: recherchiert, relevant
- Kurz: Modellbasierte Suchstrategie für AutoML

- Thema: One-Class SVM
- Typ: Modell
- Status: offen
- Kurz: Klassisches Anomalie-Modell für unsupervised learning

- Thema: PyOD
- Typ: Framework
- Status: offen
- Kurz: Vergleichsframework für Outlier Detection
```