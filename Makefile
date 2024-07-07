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
TEMP_FILES = $(AUX) $(DVI) $(LOG) $(PDF) $(LOG2) $(OUT) $(BCF) $(XML) $(BBL) $(BLG) $(BBL2) $(BLG2)

# Default target
all: $(PDF)

# Compile target
$(PDF): $(TEX) $(CLS) $(BIB)
	/usr/local/texlive/2024/bin/x86_64-linux/xelatex $(TEX)
	/usr/local/texlive/2024/bin/x86_64-linux/biber main
	/usr/local/texlive/2024/bin/x86_64-linux/xelatex $(TEX)
	/usr/local/texlive/2024/bin/x86_64-linux/xelatex $(TEX)
	# chmod a+rw $(TEMP_FILES)
	find . -maxdepth 1 -type f \( -name "$(AUX)" -o -name "$(DVI)" -o -name "$(LOG)" -o -name "$(PDF)" -o -name "$(LOG2)" \) -exec chmod a+rw {} \;

# Clean target
clean:
	rm -f $(TEMP_FILES)

# Clean shortcut target
c: clean

# Phony targets
.PHONY: all clean c
