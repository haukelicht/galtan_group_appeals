SECS := secs
APXS := apxs
# NOTEBOOKS := $(wildcard $(SECS)/*.ipynb)
NOTEBOOKS := $(filter-out $(SECS)/%BACKUP%.ipynb, $(wildcard $(SECS)/0*.ipynb))
QMDS := $(NOTEBOOKS:.ipynb=.qmd)

secs: $(QMDS)



.PHONY: FORCE #%.qmd
FORCE:

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

%.qmd: %.ipynb FORCE
	echo "quarto convert $< -> $@ (strip YAML; split at '# APPENDIX')"
	quarto convert "$<" -o "$@.tmp"
	# Remove first 4 lines (your fixed-size YAML header)
	tail -n +5 "$@.tmp" > "$@.body"
	# Remove Colab metadata (id, colab, outputId)
	sed -i '' -E '/^#\| (id|colab|outputId):/d' "$@.body"

	# Derive appendix path: <dir>/apx-<basename>
	if csplit -f "$@.part" -n 1 -s "$@.body" '/^\# APPENDIX/'; \
	then \
		tail -n +2 "$@.part1" > "$(APXS)/$(@F)"; \
		rm "$@.part1"; \
		mv "$@.part0" "$@"; \
	else \
		mv "$@.body" "$@"; \
	fi
	# Remove stale appendix if it wasn't created / is empty
	rm -f "$@.tmp" "$@.body"


preview:
	quarto preview _main.qmd --port 4444 --no-browser --no-watch-inputs

render: 
	quarto render _main.qmd --execute --wrap=preserve


clean:
	rm _main.tex _main.log _main.fls _main.fdb_latexmk _main.aux 
	rm -r _main.synctex.gz
	rm -r _main_files/

all: secs render

# version:
# 	cp _main.pdf "_versions/licht_2025_multimodal_emo_mllm_$$(date +%Y-%m-%d).pdf"

wordcount:
	@/opt/homebrew/bin/pdftotext -nopgbrk -f 3 _main.pdf tmp.txt
	@if csplit -s tmp.txt '/^\\end{CSLReferences}$$/'; then mv xx00 tmp.txt; rm -f xx0*; fi
	@sed -i '' '/^[0-9]+$$/d' tmp.txt
	@WORDCOUNT=$$(wc -w < tmp.txt | tr -d ' '); \
	sed -i '' "s/^wordcount: .*/wordcount: $$WORDCOUNT/" _main.qmd; \
	echo "number of words: $$WORDCOUNT"
	# @rm -f tmp.txt

extract-content:
	@echo "Extracting content from \\section{Introduction} to \\section{References}..."
	@sed -n '/\\\section{Introduction}/,/\\section{References}/{ /\\section{References}/!p; }' _main.tex > main_content.tex
	@echo "✅ Content extracted to main_content.tex"


separate:
	@if [ -z "$(PAGE)" ]; then \
		echo "Usage: make separate PAGE=<page_number>"; \
		exit 1; \
	fi
	/opt/homebrew/bin/pdfseparate -f 1 -l 1 _main.pdf titlepage.pdf
	@TOTAL=$$(/opt/homebrew/bin/pdfinfo _main.pdf | grep Pages | awk '{print $$2}'); \
	/opt/homebrew/bin/pdfseparate -f 1 -l $(PAGE) _main.pdf manuscript-%02d.pdf; \
	/opt/homebrew/bin/pdfseparate -f $$(($(PAGE) + 1)) -l $$TOTAL _main.pdf sm-%02d.pdf
	@/opt/homebrew/bin/pdfunite manuscript-*.pdf manuscript.pdf
	@/opt/homebrew/bin/pdfunite sm-*.pdf sm.pdf
	@rm -f manuscript-*.pdf sm-*.pdf
	@mv titlepage.pdf manuscript.pdf sm.pdf submissions/

# # arXiv preparation and testing
# arxiv-prepare:
# 	python3 ./prepare_arxiv.py --overwrite

# arxiv-compile: arxiv-prepare
# 	@echo "Compiling arXiv version..."
# 	cd _arxiv && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
# 	@echo "✅ arXiv PDF compiled successfully: _arxiv/main.pdf"

# arxiv-clean:
# 	@echo "Cleaning arXiv auxiliary files..."
# 	cd _arxiv && rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.fls *.ptc *.fdb_latexmk *.synctex.gz
# 	@echo "✅ arXiv auxiliary files cleaned"

# arxiv: arxiv-compile arxiv-clean

# arxiv-zip: arxiv-prepare
# 	@echo "Creating arXiv submission zip file..."
# 	cd _arxiv && zip -r ../arxiv-submission.zip . -x "*.aux" "*.log" "*.bbl" "*.blg" "*.out" "*.toc" "*.ptc" "*.lof" "*.lot" "*.fls" "*.fdb_latexmk" "*.synctex.gz" "main.pdf"
# 	@echo "✅ arXiv submission zip created: arxiv-submission.zip"
