# LaTeX-Praesentation ausfuehren und anschauen

## 1. In den Praesentationsordner wechseln

```bash
cd "/home/clemens/OneDrive/Dokumente/Master MIM/Fächer/Muserterkennung/Projekt/HTWK___Mustererkennung_präsi"
```

## 2. PDF bauen

Schneller Testlauf:

```bash
xelatex main.tex

lualatex main.tex
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

## 3. Vorschau/PDF oeffnen

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

## 4. Direkt bauen und oeffnen

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

1. `main.tex` oeffnen.
2. Links in VS Code auf das LaTeX-Workshop-Symbol gehen.
3. `Build LaTeX project` ausfuehren.
4. `View LaTeX PDF` oeffnen.

Wenn die PDF nicht automatisch aktualisiert wird, einmal manuell bauen:

```bash
pdflatex main.tex
```
