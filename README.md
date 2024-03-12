# Group appeals of populist radical-right and green parties 

This folder contains the materials for a project on the group appeals of populist radical-right and green parties in Europe.
The project is lead by 

- Leonce Röth (leonce.roeth@gsi.uni-muenchen.de)
- Hauke Licht (hauke.licht@wiso.uni-koeln.de)

## Folder and file structure


```
|- code/  <== all python and R code
|- docu/
|   | - coding_scheme/
|- data/
|   |- manifestos/
|      |- annotated/  <== .docx files with annotations
|      |- parsed/     <== data (e.g., JSON files) with text and annotations parsed from .docx files
|      |- raw/        <== original .txt files
|      |- sentences/  <== sentence-segmented .txt files
|- setup/  <== setup info
```

## Setup and reproducibility

### R 

Please use the `renv` package to manage dependicies.

To initialize the project

```r
renv::init() # then select option 1
```

## Documentation

Please state the author and date at the beginning of your R and pyhton scripts and a script's porpuse or title.
In R, you can use the following template:

```r
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  
#' @author 
#' @date   1999-12-31
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #
```