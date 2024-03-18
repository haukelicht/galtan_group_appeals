# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Detect languages of raw texts
#' @author Hauke Licht
#' @date   2023-05-31
#' @update 2024-03-07
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(cld2)

data_path <- file.path("data")
rawdata_path <- file.path(data_path, "manifestos", "raw")


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
res


codes2names <- c(
  "bg" = "bulgarian",
  "cs" = "czech",
  "da" = "danish",
  "de" = "german",
  "el" = "greek",
  "en" = "english",
  "es" = "spanish",
  "et" = "estonian",
  "fi" = "finnish",
  "fr" = "french",
  "hr" = "croatian",
  "hu" = "hungarian",
  "is" = "icelandic",
  "is" = "islandic",
  "it" = "italian",
  "ja" = "japanese",
  "nl" = "dutch",
  "pl" = "polish",
  "pt" = "portuguese",
  "ro" = "romanian",
  "sk" = "slovak",
  "sl" = "slovenian",
  NULL
)

res |> 
  transmute(
    source = fp,
    dest = file.path(dirname(fp), sprintf("%s-%s.txt", manifesto_id, codes2names[lang]))
  ) |> 
  with(map2(source, dest, file.rename))
