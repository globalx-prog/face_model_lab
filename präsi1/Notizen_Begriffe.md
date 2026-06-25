# Notizen und Begriffserklärungen

Diese Notizen erklären die wichtigsten Begriffe der Präsentation zum Face Model Lab. Sie sind als Vorbereitung für die Präsentation gedacht und müssen nicht vollständig auf die Folien.

## Projektidee

Das Projekt behandelt Face Detection für Video-Anonymisierung. Das Ziel ist nicht nur, ein Modell mit guten Laborwerten zu finden, sondern ein Modell, das in einer Video-Pipeline praktisch nutzbar ist. Ein Gesicht, das nicht erkannt wird, bleibt im Video sichtbar und ist deshalb für Datenschutz kritischer als eine zusätzliche Blur-Box an einer falschen Stelle.

## Zentrale Begriffe

**Face Detection**  
Aufgabe, Gesichter in einem Bild zu finden und mit Bounding Boxes zu markieren. Es geht hier nicht darum, Personen zu identifizieren, sondern nur darum, Gesichtbereiche zuverlässig zu lokalisieren.

**Object Detection**  
Allgemeiner Oberbegriff für Verfahren, die Objekte in Bildern finden und klassifizieren. Face Detection wird in diesem Projekt als spezieller Object-Detection-Fall formuliert: Die Objektklasse ist im Kern "Gesicht".

**Bounding Box**  
Rechteck um ein erkanntes Objekt. Für jedes erkannte Gesicht gibt das Modell typischerweise Koordinaten und einen Confidence-Wert aus.

**Ground Truth / GT-Faces**  
Manuell oder datensatzseitig vorgegebene Referenzboxen. Sie gelten bei der Evaluation als korrekte Lösung. `GT-Faces` bezeichnet die Anzahl der tatsächlichen Gesichter im Validierungsdatensatz.

**IoU (Intersection over Union)**  
Maß für die Überlappung zwischen Vorhersagebox und Ground-Truth-Box. Eine Vorhersage zählt als Treffer, wenn die IoU über einer gesetzten Schwelle liegt, hier zum Beispiel `0,4`.

**Confidence / Score**  
Sicherheitswert des Modells für eine erkannte Box. Ein Threshold entscheidet, ab welchem Score eine Box akzeptiert wird.

**Threshold**  
Grenzwert für die Annahme einer Vorhersage. Ein niedriger Threshold findet tendenziell mehr Gesichter, erzeugt aber mehr Fehlalarme. Ein hoher Threshold reduziert Fehlalarme, kann aber Gesichter übersehen.

**Recall**  
Anteil der gefundenen Gesichter an allen vorhandenen Gesichtern. Für Datenschutz ist Recall besonders wichtig, weil verpasste Gesichter sichtbar bleiben.

**Precision**  
Anteil der korrekten Vorhersagen an allen Vorhersagen. Precision ist wichtig, wenn unnötige Blur-Boxen vermieden werden sollen.

**F1-Score**  
Kombiniert Precision und Recall zu einem Wert. Nützlich, wenn ein ausgewogener Threshold gesucht wird.

**mAP50**  
Mean Average Precision bei IoU-Schwelle `0,5`. Häufige Object-Detection-Metrik; für Datenschutz allein aber weniger anschaulich als Recall.

**ms/Bild**  
Inferenzzeit pro Bild. Diese Metrik entscheidet, ob ein Modell für Videoverarbeitung praktikabel ist.

**Latenz**  
Zeit, die ein Modell für eine Vorhersage braucht. Bei Videos addiert sich diese Zeit über sehr viele Frames.

**WIDER FACE**  
Datensatz für Face Detection mit vielen schwierigen Situationen: kleine Gesichter, verdeckte Gesichter, Gruppen, unterschiedliche Beleuchtung und Perspektiven.

**COCO-Baseline**  
Vergleich mit Modellen, die auf dem allgemeinen COCO-Datensatz vortrainiert sind. COCO enthält viele Objektklassen, ist aber nicht speziell auf kleine Gesichtsboxen optimiert. Darum fallen COCO-Baselines bei Face Detection deutlich ab.

**Face-Finetuning**  
Nachtraining eines Modells auf Gesichtsdaten. Dadurch lernt das Modell die Zielklasse und die typischen Größen/Positionen von Gesichtern besser.

## Was heißt Red20?

`Red20` bezeichnet einen reduzierten Trainings- oder Evaluationslauf mit ungefähr 20 Prozent beziehungsweise einem bewusst verkleinerten Ausschnitt der Daten. In der Präsentation ist `Red20-Smoke` der breite Smoke-Test: mehrere Modellfamilien werden schnell gegeneinander getestet, aber nur kurz trainiert. Der Zweck ist eine erste Orientierung, nicht das finale Modellranking.

## Was heißt Red6/10?

`Red6` bezeichnet den kleineren Kandidatenlauf mit einem reduzierten Datenausschnitt. `ep10` bedeutet zehn Trainingsepochen. `Red6/10` steht in der Präsentation also für den Kandidatenvergleich auf reduziertem Datensatz mit zehn Epochen, insbesondere zwischen YOLOv8m und FCOS.

## Was heißt Smoke-Test?

Ein Smoke-Test ist ein kurzer Prüflauf. Er soll zeigen, ob die Pipeline grundsätzlich funktioniert und welche Modelle grob vielversprechend sind. Er ersetzt keine finale Evaluation.

## Modellfamilien

**Faster R-CNN**  
Two-Stage-Detector. Das Modell erzeugt zuerst Kandidatenregionen und klassifiziert/verfeinert diese danach. Vorteil: oft robuste Lokalisierung. Nachteil: meist langsamer.

**RetinaNet**  
One-Stage-Detector mit Focal Loss. Das Modell sagt Boxen und Klassen direkt auf Feature Maps voraus. Focal Loss hilft gegen das Ungleichgewicht zwischen vielen Hintergrundbereichen und wenigen echten Objekten.

**FCOS**  
Anchor-free One-Stage-Detector. Statt vordefinierter Anchor-Boxen sagt das Modell direkt pro Feature-Map-Position Objektinformationen voraus. Vorteil im Projekt: starker Recall.

**YOLOv8m**  
Praxisorientierter One-Stage-Detector. YOLO ist auf schnelle Inferenz und einfache Nutzung ausgelegt. Vorteil im Projekt: gute Video-Pipeline-Anbindung und viele direkt nutzbare Run-Artefakte.

## Größerer Abschnitt: Wie funktionieren die Modelle?

Alle getesteten Modelle verarbeiten ein Bild in mehreren Schritten. Zuerst extrahiert ein Backbone visuelle Merkmale. Diese Merkmale sind nicht mehr einzelne Pixel, sondern abstraktere Informationen wie Kanten, Texturen, Formen und Objektteile. Aus diesen Merkmalen entstehen Feature Maps, auf denen das Modell Gesichter in unterschiedlichen Größen finden kann.

Danach kommt der Detection Head. Dieser Teil des Modells entscheidet, wo im Bild ein Gesicht liegen könnte und wie sicher diese Vorhersage ist. Das Ergebnis besteht typischerweise aus Bounding Boxes, Confidence-Werten und manchmal Klassenwahrscheinlichkeiten. Da dieses Projekt im Kern nur Gesichter sucht, ist die Klassenfrage einfacher als bei allgemeinen Object-Detection-Datensätzen.

Two-Stage-Modelle wie Faster R-CNN arbeiten in zwei Schritten. Zuerst schlägt ein Region Proposal Network mögliche Objektbereiche vor. Danach prüft ein zweiter Modellteil diese Bereiche genauer und verfeinert die Boxen. Das kann sehr genau sein, kostet aber mehr Rechenzeit.

One-Stage-Modelle wie RetinaNet und YOLO sagen Boxen und Scores direkter voraus. Sie überspringen den separaten Kandidatenschritt und sind deshalb oft schneller. Für Videos ist das attraktiv, weil viele Frames verarbeitet werden müssen.

Anchor-basierte Modelle arbeiten mit vordefinierten Boxformen. Diese Anchor-Boxen dienen als Ausgangspunkt für die Vorhersage. Das Modell passt sie dann an die tatsächliche Objektposition an. Der Nachteil ist, dass Anchors zur Objektgröße und zum Datensatz passen müssen.

Anchor-free-Modelle wie FCOS verzichten auf solche vordefinierten Boxen. Stattdessen sagen sie direkt von Feature-Map-Positionen aus voraus, ob dort ein Objekt liegt und wie die Box aussieht. Das kann besonders bei variierenden Objektgrößen nützlich sein.

Nach der Vorhersage gibt es oft mehrere überlappende Boxen für dasselbe Gesicht. Non-Maximum Suppression (NMS) entfernt doppelte Treffer und behält typischerweise die Box mit dem höchsten Score. Der Confidence-Threshold entscheidet zusätzlich, welche Boxen überhaupt berücksichtigt werden.

Für die Präsentation ist die wichtigste Schlussfolgerung: Die Modelle unterscheiden sich nicht darin, dass sie völlig andere Ausgaben liefern. Am Ende liefern alle Boxen und Scores. Sie unterscheiden sich vor allem darin, wie diese Boxen entstehen, wie schnell sie berechnet werden und wie zuverlässig kleine oder schwierige Gesichter gefunden werden.

## Faster R-CNN im Detail

Faster R-CNN ist ein Two-Stage-Detector. Das bedeutet: Das Modell löst die Detektion nicht in einem einzigen direkten Schritt, sondern trennt sie in Kandidatensuche und genaue Prüfung.

Im ersten Schritt läuft das Bild durch ein Backbone. Dieses Backbone extrahiert Merkmale aus dem Eingabebild. Es arbeitet nicht mehr mit dem Originalbild als bloßer Pixelmatrix, sondern erzeugt Feature Maps, in denen relevante visuelle Muster enthalten sind.

Auf diesen Feature Maps arbeitet das Region Proposal Network (RPN). Das RPN schlägt Regionen vor, in denen wahrscheinlich ein Objekt liegt. Im ursprünglichen Object-Detection-Kontext können das viele Objektarten sein; in unserem Projekt interessiert praktisch die Klasse Gesicht. Für Face Detection heißt das: Das RPN erzeugt Kandidatenbereiche, die möglicherweise Gesichter enthalten.

Danach folgt der zweite Schritt: Die vorgeschlagenen Regionen werden genauer geprüft. Der ROI Head klassifiziert die Region und verfeinert die Bounding Box. Dadurch kann Faster R-CNN oft präzise lokalisieren, braucht aber mehr Rechenzeit als direkte One-Stage-Modelle.

Der Ablauf im Projekt war:

1. WIDER FACE als Datengrundlage verwenden.
2. Annotationen so aufbereiten, dass sie für das Torchvision-Training nutzbar sind.
3. Faster R-CNN im Red20-Smoke-Test trainieren beziehungsweise feinjustieren.
4. Auf Validierungsbildern Vorhersagen erzeugen.
5. Vorhersagen mit Ground Truth über IoU-Schwelle und Confidence-Threshold bewerten.
6. Recall, Trefferzahl und ms/Bild mit FCOS, RetinaNet und YOLOv8m vergleichen.

In den Ergebnissen liegt Faster R-CNN im Red20-Smoke-Vergleich beim Recall hinter FCOS, aber vor YOLOv8m und RetinaNet. Gleichzeitig ist die Inferenzzeit höher als bei den schnelleren Ansätzen. Für das Projekt bedeutet das: Faster R-CNN ist eine starke Qualitätsreferenz, aber nicht automatisch die beste Wahl für eine Video-Pipeline.

Der wichtigste Unterschied zu den anderen Modellen ist der zweistufige Ablauf. YOLOv8m und RetinaNet arbeiten als One-Stage-Detectoren: Sie sagen Klassen, Scores und Boxen direkter auf Feature Maps voraus. FCOS ist ebenfalls One-Stage, verzichtet aber zusätzlich auf vordefinierte Anchor-Boxen. Faster R-CNN investiert mehr Arbeit in die Kandidatenprüfung. Das kann bei schwierigen Bildern helfen, kostet aber Zeit pro Bild.

## Interpretation der Ergebnisse

FCOS ist in den gezeigten Ergebnissen der Qualitätsfavorit, weil der Recall höher ist. Das heißt: Es findet mehr der vorhandenen Gesichter. Für Datenschutz ist das stark.

YOLOv8m ist der Praxisfavorit, weil es schneller ist, gute Run-Artefakte liefert und einfacher in die Video-Pipeline passt. Es ist daher für eine operative Lösung attraktiv, auch wenn der Recall niedriger ist.

Die Modellentscheidung ist deshalb zweigeteilt: Für maximale Abdeckung spricht FCOS. Für einfache und schnelle Videoverarbeitung spricht YOLOv8m.

## TODO für fehlende Rohdaten

Für neu gezeichnete Trainings- und Metrikgrafiken fehlen im Projekt aktuell CSV/JSON-Rohdaten mit den epochengenauen Verläufen für FCOS und YOLO. Solange diese Daten fehlen, sollten keine neuen Vergleichsgrafiken aus abgelesenen Bildwerten erstellt werden. Sinnvoll wäre:

- YOLO `results.csv` aus dem Run-Ordner ergänzen.
- FCOS Trainingshistorie als CSV oder JSON exportieren.
- Beide Verläufe mit denselben Achsen und klarer Legende neu plotten.
- In der Folie deutlich unterscheiden zwischen Trainingskurven, Validierungsmetriken und finalen Evaluationswerten.
