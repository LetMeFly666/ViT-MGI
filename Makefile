# Makefile for compiling main.tex with xelatex
# and cleaning auxiliary files

# Variables
TEX = main.tex
CLS = IEEEtran.cls
PDF = main.pdf
AUX = main.aux
DVI = main.dvi
LOG = main.log
LOG2 = missfont.log

# Default target
all: $(PDF)

# Compile target
$(PDF): $(TEX) $(CLS)
	/usr/local/texlive/2024/bin/x86_64-linux/xelatex $(TEX)

# Clean target
clean:
	rm -f $(AUX) $(DVI) $(LOG) $(PDF) $(LOG2)

# Clean shortcut target
c: clean

# Phony targets
.PHONY: all clean c
