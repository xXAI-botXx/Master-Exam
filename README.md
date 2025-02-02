# Master-Exam
This project contains my Master Exam (planning, writing, ...)

- [Master-Exam](#master-exam)
    - [Projekt Details](#projekt-details)
    - [Plan](#plan)
    - [Schriftliches](#schriftliches)
    - [Source](#source)
    - [Fragen](#fragen)



---
### Projekt Details

**Titel:** Können generative Modelle komplexe Zusammenhänge lernen?

**Aufgabe:** (Problemstellung)<br>
Es soll untersucht werden, ob generative Modelle physikalisch korrekte Daten erzeugen können. Spezifisch wird die Generierung der Schallausbreitung in städtischen Gebieten untersucht.

- Input: Satellitenbild -> mit Gebäuden? Kein Input?
- Output: Schallausbreitung (Schallfrequenz für jeden Pixel)

**Weitere Bedingungen:**
- Python + Pytroch?
- ...

**Ziele:**
- Analyse und Evaluation von aktuellen generativen Ansätzen physikalische Zusammenhänge abbilden zu können
  - Erstellung eines Trainingdatensatzes (genug Daten? Evtl. mit synthetischen Daten erweitern?)
  - Entwicklung eines BenchmarksMetriken zur Bewertung von generativen Modellen
  - Aktuelle generative Ansätze herauserarbeiten
  - Aktuelle generative Ansätze implementieren
  - Aktuelle generative Ansätze trainieren oder vortrainiert?
- Mitarbeit im Forschungsprojekt (?)
- Schreiben der Masterarbeit



**Thesen:**
1. Generative Modelle können die Wahrscheinlichkeiten komplexer physikalischer Vorgänge anhand eines Bildes vorhersagen
2. Generative Modelle sind nicht für deterministische Aufgaben geeignet und folgerichtig, müssen die Modelle angepasst werden, um deterministisch zu sein -> jenachdem wie man es einsetz bzw. was der input ist

=> Hängt von der eigentlich und ganz genauen Aufgabenstellung ab!!!


**Koorperation:**<br>
Das Projekt wird in Kooperation mit der Herrenknecht Vertikal GmbH und dem Institute for Machine Learning and Analytics im Rahmen des BMBF geförderten Forschungsprojekts KI-Bohrer durchgeführt.
- https://www.ki-bohrer.de/
- https://imla.hs-offenburg.de/




---
### Plan

1. Thema deines Masters finden
2. Recherche -> Rausschreiben in md file
3. Thema aktualisieren + These(n) definieren
4. Experimentenplan aufstellen (wie messbar?)
5. Experimente aufsetzen (evtl. architekturen erstellen)
6. Experiment durchführen (trainieren + ergebnis)
7. anfangen zu schreiben / schreibplan
8. ...




---
### Schriftliches

Der Latex Code ist auf Overleaf und hier wird der Code ebenfalls ohne die Bilder geupdated.

https://www.overleaf.com/project/679f743ec9a1ac78e5db574b




---
### Source

Here all important resources (most likely papers) are listed with the probably most helpful statements.

Themen: Generative Modelle, Abbildung von Physikalischen Prozessen mit NN, Abbildung von Physikalischen Prozessen mit generativen Modellen 


**Template-Paper-Name**<br>
```
@article{,
	title={}, 
	author={},
	year={},
	url={},
	doi={}
}
```
- template statement 1
- ...


...



---
### Fragen

- Analyse und Evaluation von aktuellen generativen Ansätzen physikalische Zusammenhänge abzubilden -> **Wird hier nur die Schallausbreitung oder mehr erwartet?**
- Sollen die generativen Modelle selbst trainiert oder vortrainiert getestet werden?
  - Wahscheinlich eher selbst trainieren
- Was ist mit "- Mitarbeit im Forschungsprojekt" gemeint? Eigentlich ja nur indirekt oder? Kein direktes Ziel, oder?
- Wie kann folgendes Ziel realistisch sein? "Praktische Anwendung der
Ergebnisse auf realen
Baustellen". Das Projekt kann doch nur indirekt helfen und nicht direkt auf der baustelle einsatz finden....oder?
- Was soll das model am Ende als Input bekommen? Ein Satellitenbild -> mit oder ohne Gebäuden eingezeichnet? Oder bekommt es gar keinen Input und generiert einfach so Bilder mit Schallausbreitung und generiert auch noch die Städtische Umgebung?


- Gibt es schon generative modelle? Sollen diese irgendwie auch verwendet werden?
- Wie sehen die Daten aus?




