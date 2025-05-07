NAME = basics
IN = ${NAME}.tex
OUT = ${NAME}.pdf 

ifeq (${LTEX},True)
	COMPILER=lualatex
else
	COMPILER=pdflatex
endif

# TEST := $(if $(TEST),$(TEST),$(something else))

${OUT}: ${IN}
	$(COMPILER) $(IN) 
	make clean
    
clean:
	rm *.xml *.aux *.blg *.out *.log *.bib *.fls *.gz
