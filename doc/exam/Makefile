TEXC?=lualatex
TEXCFLAGS?=--halt-on-error
OUTDIR?=.build
OUTDIR_AUX?=$(OUTDIR)/aux

LATEXMKFLAGS=-pdf -dvi- -ps- -bibtex \
	-pdflatex='$(TEXC) %O -synctex=1 $(TEXCFLAGS) %S' \
	-outdir=$(OUTDIR) -emulate-aux-dir -auxdir=$(OUTDIR_AUX)

.PHONY: thesis.pdf
TARGET=thesis.pdf

.PHONY: all
all: $(TARGET)

$(TARGET): thesis.tex
	@mkdir -p $(OUTDIR)
	@mkdir -p $(OUTDIR_AUX)
	latexmk $(LATEXMKFLAGS) thesis.tex

.PHONY: watch
watch:
	while inotifywait -r \
		-e close_write .; \
		do $(MAKE); done

RM?=rm -f

.PHONY: clean
clean:
	$(RM) -r "$(OUTDIR)"
