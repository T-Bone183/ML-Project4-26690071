MAIN := 26690071RW441assignment4

.PHONY: all clean cleanall view

all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	latexmk -c $(MAIN).tex

cleanall:
	latexmk -C $(MAIN).tex

view: all
	xdg-open $(MAIN).pdf
