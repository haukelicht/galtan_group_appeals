library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(stringr)
library(ggplot2)

data_path <- file.path("data", "manifestos")
input_path <- file.path(data_path, "sentences")

# read sentence-level data ----
files <- list.files(input_path, pattern = ".txt", include.dirs = FALSE, recursive = TRUE, full.names = TRUE)

df <- tibble(fp = files) |> 
  mutate(nm = sub(paste0(input_path, .Platform$file.sep), "", fp)) |> 
  separate(nm, c("country_iso3c", "date", "nm"), sep = .Platform$file.sep) |> 
  extract(nm, c("manifesto_id", "language"), regex = "^([^-]+)-([a-z]+)\\.txt$") |> 
  mutate(text = map(fp, read_lines)) |> 
  unnest(text)

sort(table(df$language), decreasing = TRUE)

# identify patterns to clean ----

df <- df |> 
  mutate(
    n_chars = nchar(text),
    n_special_chars = str_count(text, "[^\\p{L}0-9.!?,;\\p{Pi}\\p{Pf} ]"),
    special_char_ratio = n_special_chars/n_chars
  ) 

df |> 
  filter(special_char_ratio > 0) |> 
  # pull(special_char_ratio) |> 
  # qplot() + scale_x_log10()
  ggplot(aes(x = n_chars, y = special_char_ratio)) +
    geom_point() +
    scale_x_log10()
# note: anything with ≥25% special characters is likely spam 
    
# df |> 
#   filter(n_chars > 5) |>
#   filter(special_char_ratio > .25) |> 
#   View()


df <- df |> 
  # remove lines that are enpty or only white spaces
  filter(!grepl("^\\s*$", text, perl = TRUE)) |> 
  # remove all lines that contain not a single letter
  filter(!grepl("^\\P{L}+$", text, perl = TRUE)) |>
  # remove annotations
  filter(!grepl("^<No title information>$", text, perl = TRUE)) |>
  # remove lines with 5 or less characters
  filter(nchar(text) > 5) |> 
  # remove lines with ≥25% special character ratio
  filter(special_char_ratio < 0.25) |> 
  # remove any lines that start with seven or mor digits (IBAN etc.)
  filter(!grepl("^\\d{7,}", text)) |> 
  select(-n_chars, -n_special_chars, -special_char_ratio)

# identify bullet point characters
tab <- df$text |> str_extract("^[^\\p{L}0-9\\s]+(?=\\s*\\p{L})") |> table() |> sort(decreasing = TRUE)
length(tab)
# note: I inspected all with 2 or more occurrences
sum(tab > 1)
tab[(1:10)+40]

# df |>
#   filter(grepl("....", text, fixe = TRUE)) |>
#   View()

bullet_point_chars <- c(
  "•", "•", "・", "■", "-", 
  "[\u2776-\u277F]",  
  "…", "\\.{3,4}", "\\.", 
  "\\+", "\\*", "~", "=",
  "¡", "\\^-\\^", ";'V;", "' -", "\\^ -", 
  "0\\b", "O\\b", "o\\b"
)

# identify special characters to replace
tab <- df$text |> 
  str_extract_all("[^\\p{L}0-9\\p{Sc} ]{2,4}") |> 
  unlist() |> 
  table() |> 
  sort(decreasing = TRUE)

# tab[(1:10)+200]

split_at <- c("(?<=:)(?=•)")
replace <- c(
  ",," = "„", "<<" = "«", ">>" = "»",
  "_:(?=\\s|$)" = ":", ":_(?=\\s|$)" = ":", 
  "_!(?=\\s|$)" = "!", "!_(?=\\s|$)" = "!", 
  "_\\?(?=\\s|$)" = "?", "\\?_(?=\\s|$)" = "?", 
  "\\._(?=\\s|$)" = ".", "\\.�(?=\\s|$)" = ".", ";;(?=\\s|$)" = ";" ,
  "CO~2~" = "CO2",  "\\(--> " = "(",
  "\\^'\\^" = "'",
  "(?<=^|\\b)https?://\\S+" = "URL"
)
remove <- c(
  "^�\\s*", "^\\[\\*\\]\\s*", "^:",
  "~\\.~", 
  "//\\|",
  "^\\|", "^/", "^:", "^\\!", "^<\\b",
  "^\\[[^\\]]+\\]$"
)

df <- df |> 
  separate_rows(text, sep = split_at[1]) |> 
  mutate(
    text = str_replace(text, sprintf("^(%s)(?=\\s*\\p{L})", paste(bullet_point_chars, collapse = "|")), "- "),
    text = str_replace_all(text, replace),
    text = str_remove_all(text, sprintf("(%s)", paste(remove, collapse = "|"))),
    text = str_replace_all(text, "\\s+", " ")
  ) |> 
  filter(!grepl("^\\s*$", text))

sort(unique(df$language))

set.seed(1234)
df |> 
  sample_n(20) |> 
  select(language, text)

language_codes <- read_csv(file.path(data_path, "language_codes.csv"))
name2code <- with(language_codes, setNames(code, language))

df <- df |> 
  select(-fp) |>
  mutate(
    country_iso3c = toupper(country_iso3c),
    lang = name2code[language],
  ) |> 
  # count(manifesto_id) |> arrange(desc(n))
  group_by() |> 
  mutate(sentence_id = sprintf("%s-%05d", manifesto_id, row_number())) |> 
  ungroup()

# compute translation cost in dollars
df |> 
  filter(lang != "en") |> 
  summarise(cost = sum(nchar(text))/1e6*20)

# write to disk ----
fp <- file.path(data_path, "all_manifesto_sentences.tsv")
df |> 
  select(country_iso3c, sentence_id, lang, text) |> 
  write_tsv(fp)

