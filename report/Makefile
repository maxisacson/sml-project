LTX=$(shell which lualatex || which pdflatex)
BTX=$(shell which bibtex)

all:
	$(LTX) report.tex
	$(BTX) report.aux
	$(LTX) report.tex
	$(LTX) report.tex
