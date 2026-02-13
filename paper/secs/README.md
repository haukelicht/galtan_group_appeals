# Section files

- The **markdown** files (`0*.md`) in this folder contain the text of the no-code paper sections.

- The **Jupyter Notebook** files (`0*.ipynb`) in this folder contain the text an code 


## Quarto files for sections of the paper

calling the make rule `make secs` from the [parent directory](../) will create the quarto files in this folder from the Jupyter Notebooks.

<span style="color: red;">Do NOT manually edit these files.</span>

The `.qmd` files are then included in the main paper via `include` statements in the main [`_main_.qmd`](../_main_.qmd) file.

