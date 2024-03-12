# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Prepare data folder structure
#' @author Hauke Licht
#' @date   
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

# load required packages
library(readr)
library(dplyr)
library(purrr)

data_path <- file.path("data")
output_path <- file.path(data_path, "manifestos", "raw")

# create folders ---- 

cases <- read_tsv(file.path(data_path, "dataset_population.tsv"))

table(is.na(cases$manifesto_id))
paths <- cases |> 
  filter(!is.na(country_iso3c)) |> 
  distinct(country_iso3c, election_date) |> 
  with(
    file.path(output_path, tolower(country_iso3c), format(election_date, "%Y%m"))
  )

res = lapply(paths, dir.create, showWarnings = FALSE, recursive = TRUE)
