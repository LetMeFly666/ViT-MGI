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
TEMP_FILES = $(AUX) $(DVI) $(LOG) $(PDF) $(LOG2)

# Default target
all: $(PDF)

# Compile target
$(PDF): $(TEX) $(CLS)
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
