# LaTeX-Präsentation ausführen und anschauen

## 1. In den Präsentationsordner wechseln

```bash
cd "/home/clemens/OneDrive/Dokumente/Master MIM/Fächer/Muserterkennung/Projekt/face_mode_lab/präsi"
```

## 2. PDF bauen

Schneller Testlauf:

```bash
xelatex main.tex
xelatex -output-directory=main_out main.tex

```

Robuster Lauf mit Literaturverzeichnis:

```bash
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

Danach entsteht:

```text
main.pdf
```

## 3. Vorschau/PDF öffnen

Unter Linux:

```bash
xdg-open main.pdf
```

Alternativ mit einem konkreten PDF-Viewer:

```bash
evince main.pdf
```

oder:

```bash
okular main.pdf
```

## 4. Direkt bauen und öffnen

```bash
pdflatex main.tex && xdg-open main.pdf
```

## 5. Falls `pdflatex` fehlt

Ubuntu/Debian:

```bash
sudo apt install texlive-latex-extra texlive-lang-german texlive-bibtex-extra biber
```

Danach erneut bauen:

```bash
pdflatex main.tex
```

## 6. Vorschau in VS Code

Empfohlen ist die Extension **LaTeX Workshop**.

Typischer Ablauf:

1. `main.tex` öffnen.
2. Links in VS Code auf das LaTeX-Workshop-Symbol gehen.
3. `Build LaTeX project` ausführen.
4. `View LaTeX PDF` öffnen.

Wenn die PDF nicht automatisch aktualisiert wird, einmal manuell bauen:

```bash
pdflatex main.tex
```
