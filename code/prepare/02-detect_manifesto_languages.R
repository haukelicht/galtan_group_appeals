# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Detect languages of raw texts
#' @author Hauke Licht
#' @date   2023-05-31
#' @update 2024-03-20
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(cld2)

data_path <- file.path("data", "manifestos")
rawdata_path <- file.path(data_path, "raw")

files <- list.files(rawdata_path, pattern = "-unknown.txt", include.dirs = FALSE, recursive = TRUE, full.names = TRUE)

detect_languages <- function(fp, n.heighest = 1) {
  lines <- read_lines(fp)
  langs = cld2::detect_language(lines)
  tab <- sort(prop.table(table(langs)), decreasing = TRUE)
  idxs <- 1:n.heighest
  if (length(tab) < n.heighest)
    idxs <- seq_along(tab)
  out <- data.frame(lang = names(tab[idxs]), prob = as.vector(tab[idxs]))
  
  return(out)
}

res <- tibble(fp = files) |> 
  mutate(
    country_iso3c = toupper(sub(".+/([a-z]{3})/.+", "\\1", fp, perl = TRUE)),
    manifesto_id = sub("-.+$", "", basename(fp)),
    langs = map(fp, detect_languages)
  ) |> 
  unnest(langs)


# inspect  
count(res, country_iso3c, lang)
# note: this makes sense

distinct(res, lang)

lang_codes <- read_csv("data/manifestos/language_codes.csv")
codes2names <- with(lang_codes, setNames(language, code))


res |> 
  transmute(
    source = fp,
    dest = file.path(dirname(fp), sprintf("%s-%s.txt", manifesto_id, codes2names[lang]))
  ) |> 
  with(map2(source, dest, file.rename))
