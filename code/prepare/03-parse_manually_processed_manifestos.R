# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Parse Word document manifestos manually prepared by us 
#' @author Hauke Licht
#' @date   2024-03-15
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

# load pkgs 
library(readr)
library(dplyr)
library(purrr)
library(stringr)
library(officer)

data_path <- file.path("data", "manifestos", "raw")

# read data ----

docx_files <- list.files(data_path, pattern = "^[^~].+\\.docx$", recursive = TRUE, full.names = TRUE)

# names(docx_files) <- sub(".docx", "", basename(docx_files), fixed = TRUE)

docs_df <- map_dfr(set_names(docx_files), compose(as_tibble, docx_summary, read_docx), .id = "path")

table(docs_df$content_type)
table(docs_df$style_name, useNA = "ifany")

# I double-checked them, these are all relevant text passages
filter(docs_df, style_name == "Bildbeschriftung")
filter(docs_df, style_name == "Andere")
filter(docs_df, style_name == "Inhaltsverzeichnis") 
filter(docs_df, is.na(style_name), nchar(text) > 0)


styles_regex <- "^(Fließtext|Überschrift|List|Inhaltsverzeichnis|Bildbeschriftung|Andere)"

out <- docs_df |> 
  filter(
    (
      is.na(style_name) & nchar(text) > 0
      |
      grepl(styles_regex, style_name)
    )
  ) |> 
  select(path, text)

imap(
  split(out$text, out$path),
  function(texts, fp) {
    write_lines(
      trimws(texts),
      sub("\\.docx$", "-unknown.txt", fp)
    )
    invisible()
  }
)
