# Makefile for compiling main.tex with xelatex
# and cleaning auxiliary files

# source
TEX = main.tex
CLS = IEEEtran.cls
BIB = references.bib
# temp
BBL = main.bbl
BBL2 = references.bib.bbl
BLG = main.blg
BLG2 = references.bib.blg
PDF = main.pdf
AUX = main.aux
DVI = main.dvi
LOG = main.log
LOG2 = missfont.log
OUT = main.out
BCF = main.bcf
XML = main.run.xml
FLS = main.fls
XMK = main.fdb_latexmk
XDV = main.xdv
TEMP_FILES = $(AUX) $(DVI) $(LOG) $(LOG2) $(OUT) $(BCF) $(XML) $(BBL) $(BLG) $(BBL2) $(BLG2) $(FLS) $(XMK) $(XDV)
TEMP_DIRS = svg-inkscape

# Default target
all: $(PDF)

# Compile target
$(PDF): $(TEX) $(CLS) $(BIB)
	rm -f $(TEMP_FILES)
	rm -rf $(TEMP_DIRS)
	# /usr/local/texlive/2024/bin/x86_64-linux/latexmk -xelatex -shell-escape main.tex
	
	/usr/local/texlive/2024/bin/x86_64-linux/pdflatex $(TEX)
	/usr/local/texlive/2024/bin/x86_64-linux/bibtex main
	/usr/local/texlive/2024/bin/x86_64-linux/pdflatex $(TEX)
	/usr/local/texlive/2024/bin/x86_64-linux/pdflatex $(TEX)
	# chmod a+rw $(TEMP_FILES)
	find . -maxdepth 1 -type f \( -name "$(AUX)" -o -name "$(DVI)" -o -name "$(LOG)" -o -name "$(PDF)" -o -name "$(LOG2)" \) -exec chmod a+rw {} \;

# Clean target
clean:
	rm -f $(TEMP_FILES)
	rm -f $(PDF)
	rm -rf $(TEMP_DIRS)

# Clean shortcut target
c: clean

# Phony targets
.PHONY: all clean c
