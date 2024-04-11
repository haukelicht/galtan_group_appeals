# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Download manifesto raw texts
#' @author Hauke Licht
#' @date   2023-05-31
#' @update 2024-03-20
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

library(manifestoR)
library(dataverse)
library(httr)
library(readr)
library(dplyr)
library(purrr)
library(stringr)
library(lubridate)
source(file.path("R", "manifestoR_utils.R"))

data_path <- file.path("data", "manifestos")
rawdata_path <- file.path(data_path, "raw")
dir.create(rawdata_path, showWarnings = FALSE, recursive = TRUE)
sentence_data_path <- file.path(data_path, "sentences")
dir.create(sentence_data_path, showWarnings = FALSE, recursive = TRUE)

# determine download sources ----

manifesto_sources <- read_tsv(file.path(data_path, "manifesto_sources.tsv"))

# download from manifestoVault V1.0 ----

Sys.setenv("DATAVERSE_SERVER" = "dataverse.nl")
Sys.setenv("DATAVERSE_ID" = "FSW")

# URL: https://doi.org/10.34894/VKQSPO
repo_doi <- "doi:10.34894/VKQSPO"

files <- dataset_files(repo_doi, version = "1.1") # 2024-01-30
(file_names <- map_chr(files, "label"))

cases <- manifesto_sources |> 
  filter(country_iso3c %in% c("DEU", "IRL", "GBR")) |> 
  distinct(country_iso3c, manifesto_id)

df <- map_dfr(
  file_names[grep("csv$", file_names)],
  ~get_dataframe_by_name(., dataset = repo_doi, .f = read_csv)
)

df <- df |> 
  mutate(manifesto_id = sprintf("%d_%d", party, date)) |> 
  inner_join(cases)

# discard from list of manifestos to get data for
manifesto_sources <- manifesto_sources |> anti_join(distinct(df, country_iso3c, manifesto_id))

df <- df |> 
  transmute(
    path = file.path(
      sentence_data_path, 
      tolower(country_iso3c), 
      date,
      sprintf("%d_%d-%s.txt", party, date, ifelse(country_iso3c == "DEU", "german", "english"))
    ),
    text
  ) 
  
res <- imap(
  split(df$text, df$path), 
  function(texts, fp) {
    dir.create(dirname(fp), showWarnings = FALSE, recursive = TRUE)
    write_lines(texts, fp)
  }
)
rm(df); gc()

# download from CMP ----

cases <- manifesto_sources |> 
  filter(raw_text_exists == "yes") |> 
  group_by(manifesto_id) |> 
  filter("Manifesto Project" %in% source) |> 
  ungroup() |> 
  filter(source == "Manifesto Project") |> 
  distinct(country_iso3c, election_date, party_id, manifesto_id)

## 0. setup ----

# set in CMP_API_KEY=<API key> in .Renviron file (maybe use `usethis::edit_r_environ(scope = "project")`)
api_key <- Sys.getenv("CMP_API_KEY")
mp_setapikey(key = api_key)

# define CMP dataset version to use
this_version <- "MPDS2023a" # !!! DO NOT CHANGE THIS !!!

## 1. get metadata ----

# get dataset (meta data only)
cmp_dataset <- mp_maindataset(version = this_version, south_america = TRUE)

# keep selected indicators
cmp_dataset_metadata <- cmp_dataset |> 
  transmute(
    country, countryname
    , edate, date
    , party, partyname, partyabbrev
    , manifesto_id = paste0(party, "_", date)
    , manual
    , progtype
    , datasetorigin
    , datasetversion
    , id_perm
  )

## 2. identify the manifestos we want ----

these <- cmp_dataset_metadata |>
  transmute(manifesto_id, available_ = TRUE) |>
  right_join(cases) |>
  # # verify
  # with(table(is.na(available_)))
  # with(table(is.na(election_date)))
  select(manifesto_id) |> 
  separate(manifesto_id, c("party", "date"), sep = "_") |> 
  mutate_all(as.double) |> 
  as.data.frame()
  
## 2. get the annotated text data ----

cmp_data <- mp_corpus(these)
class(cmp_data)

# convert into data frame
cmp_data_df <- as_tibble(cmp_data)
nrow(cmp_data_df)

# for how many have we quasi-sentence level data (i.e., "annotations")
prop.table(table(cmp_data_df$annotations))
# notes: 
#  - only ~60% have annotations at quasi-sentence level
#  - the rest is just one long text string per manifesto
#  - below, we'll process these differently 

## 3 process manifestos ----

### 3.1 process manifestos with annotations ----

# keep only data for one manifesto as an example
cmp_data_df_long <- cmp_data_df %>%
  filter(annotations) |> 
  select(manifesto_id, language, annotations, data) %>%
  unnest(data) %>%
  select(-eu_code)

#' @note when `cmp_code` is NA or "H", this is because 
#'  (i) the recorded text comes from a title page, preamble, table of contents, etc.
#'  (ii) the recorded text belongs to a header
table(is.na(cmp_data_df_long$cmp_code))
table(cmp_data_df_long$cmp_code == 'H')

# # note: the commented-out code below helps identifying characters that identify bullet points
# tab <- cmp_data_df_long |>
#   filter(grepl("^\\s?\\W", text)) |>
#   pull(text) |>
#   str_extract("^[^\\p{L}\\s]*(?=\\s*\\p{L})") |>
#   table() |>
#   sort(decreasing = TRUE)
# 
# chars <- names(tab)
# tab <- tab[-grep("^//", chars)]
# chars <- names(tab)
# 
# chars[(1:10)+110]
# # length(tab)
# i = 113
# cat(chars[i])
# # grepl("\\p{Pd}", chars[i], perl = TRUE)
# # grepl("\\p{Pc}", chars[i], perl = TRUE)
# sprintf("%X", utf8ToInt(chars[i]))
# 
# cmp_data_df_long |>
#   mutate(r_ = row_number()) |>
#   filter(grepl("^\\s?\\W", text)) |>
#   filter(grepl(paste0("^", chars[i]), text)) |>
#   # filter(grepl("^\\.{3}", text)) |>
#   # sample_n(20) |>
#   View()

bullet_point_chars <- c(
  "\u2022+", ">+ ", "» ", "\\*", "√", "·", "→", "o\\s+→", "• …", "·\\s+•", "¾", "► ", "§ ", "o ",
  # private use
  "§?\uF0A7 ", "\uF0B7", "\uF020", "\uF076", "\uf0b7", "\u25BA", "\uF09F", "\\p{Co}",
  "\\p{Pd}+", "\\p{Pc}+", "−+",
  "\\(\\p{Ll}\\) ",
  "\\(\\d{1,2}\\) ",
  "\\d{1,2}\\) ",
  "\\(?\\d{1,2}(\\.\\d{1,2}){1,2}\\) ",
  "\\d{1,2}((\\.|/)\\d{1,2}){1,2}\\.? ",
  "\\d{1,2}\\. ",
  NULL
)

bullet_point_regex <- sprintf("^\\s*(%s)\\s*", paste(bullet_point_chars, collapse = "|"))

remove <- c(
  intToUtf8("0xA0"), 
  intToUtf8("0x7F"), 
  "^//\\s*", 
  "^\\s*\\*+\\s*$",
  "^…(?!\\s)",
  "^\uFFFD",
  "^\uF02F",
  "^\\|",
  "^‖",
  "\\(…\\)",
  "<>"
)

bullet_point_replace <- " <BP> "

replace <- c(
  "(?<=[:!?.;,])\\s*\\?\\s+" = bullet_point_replace,
  "« " = "«",
  " »" = "»",
  "\\h+" = " ",
  NULL
)

cmp_data_df_long <- cmp_data_df_long |>
  group_by(manifesto_id) |> 
  mutate(
    # create counter that indicates when a text has not been annotated with a CMP code
    uncoded_ = (is.na(cmp_code) | cmp_code == 'H')
    , tmp_ = cumsum(!uncoded_)
    # use this info to compute a content block indicator
    , bloc_nr_ = cumsum(tmp_ == lag(tmp_, default = 0))
    # drop the temporary columns
    , tmp_ = NULL
  ) |> 
  # discard preamble
  filter(row_number() >= min(which(!uncoded_))) |> 
  ungroup() |> 
  mutate(
    # remove bullet points
    text = if_else(
      grepl(bullet_point_regex, text, perl = TRUE),
      sub(bullet_point_regex, bullet_point_replace, trimws(text), perl = TRUE),
      trimws(text)
    ),
    text = str_replace_all(text, setNames(rep("", length(remove)), remove)),
    text = str_replace_all(text, replace),
    text = gsub(bullet_point_replace, "\n ", text, fixed = TRUE),
    # wrap header rows
    text = ifelse(is.na(cmp_code) | cmp_code == 'H', paste("\n", text, "\n"), text),
    text = sub("^(\n )+", "\n ", text)
  ) |> 
  select(-cmp_code) |> 
  filter(
    !(uncoded_ & grepl("^\n \\p{L}+ \n$", text, perl = TRUE)),
    !(uncoded_ & grepl("^\n \\S+ \n$", text))
  ) |> 
  group_by(manifesto_id, bloc_nr_) |>   
  mutate(
    prev_ends_with_letter = TRUE, #grepl("\\p{L}\\s*$", lag(text, default = ""), perl = TRUE),
    text = if_else(
      prev_ends_with_letter & grepl("^\\p{L}", text, perl = TRUE),
      paste0(" ", text),
      text
    )
  ) %>%
  summarise(
    language = first(language),
    text = paste(text, collapse = "")
  ) |> 
  ungroup() |> 
  group_by(manifesto_id) |> 
  summarise(
    language = first(language),
    text = paste(text, collapse = "\n\n")
  ) |> 
  mutate(text = gsub("(\n\\s+)+", "\n", text) |> trimws())

tmp <- cmp_data_df_long |> 
  separate(manifesto_id, c("party", "date"), sep = "_", remove = FALSE) |> 
  left_join(cases) |> 
  mutate(path = file.path(rawdata_path, tolower(country_iso3c), date, sprintf("%s-%s.txt", manifesto_id, language)))

# write to disk 
res <- map2(tmp$text, tmp$path, function(x, fp) {
  dir.create(dirname(fp), recursive = TRUE, showWarnings = FALSE)
  write_lines(x, fp)
})

# remove the processed manifestos from the corpus data frame
cmp_data_df <- anti_join(cmp_data_df, select(tmp, manifesto_id))
manifesto_sources <- anti_join(manifesto_sources, select(tmp, manifesto_id))

### 3.2 process manifestos without annotations ----

cmp_data_df_long <- cmp_data_df %>%
  filter(!annotations) |> 
  select(manifesto_id, language, annotations, data) %>%
  # inner_join(distinct(manifesto_sources, manifesto_id))
  unnest(data) %>%
  select(-cmp_code, -eu_code)

# write_lines(sample(cmp_data_df_long$text, 1), "~/Downloads/manifesto.txt")
# idx <- which(cmp_data_df_long$manifesto_id == "13951_198401")
# write_lines(cmp_data_df_long$text[idx], "~/Downloads/manifesto.txt")

bullet_point_regex <- sprintf("(^|\\s)(%s)\\h*", paste(bullet_point_chars, collapse = "|"))

# set.seed(42)
tmp <- cmp_data_df_long |> 
  # sample_n(1) |> 
  mutate(
    text = str_replace_all(text, bullet_point_regex, "\n"),
    text = str_replace_all(text, "(?<![\\p{L} ])\\h{2,}(?!\\p{Ll})", "\n")
  ) |> 
  separate_rows(text, sep = "\n") |> 
  # View()
  mutate(
    # trim white spaces
    text = trimws(text),
    # next: remove bullet point chars
    text = if_else(
      grepl(bullet_point_regex, text, perl = TRUE),
      sub(bullet_point_regex, "\n ", text, perl = TRUE),
      text
    ),
    text = str_replace_all(
      text, 
      c(
        # "\\h{2,}(?=\\p{Lu}+ )" = "\n",
        "(?<=\n)(\\h*[^\\p{Ll}\n]{2,}[.!?])(?=\\h+\\p{L}\\p{Ll})" = "\\1\n",
        "(?<=\n)(\\h*[^\\p{Ll}\n]{2,})(?=\\h{2,}\\p{L}\\p{Ll})" = "\\1\n",
        "\\h+" = " ",
        "(\n\\h)+" = "\n",
        NULL
      )
    )
  ) |> 
  separate_rows(text, sep = "\n") |> 
  filter(
    !grepl("^\\h*$", text, perl = TRUE),
    !grepl("^\\P{L}*$", text, perl = TRUE),
  )
  
tmp <- tmp |> 
  group_by(manifesto_id, language) |> 
  summarise(text = paste(text, collapse = "\n")) |> 
  ungroup() |> 
  left_join(distinct(manifesto_sources, manifesto_id, country_iso3c)) |> 
  separate(manifesto_id, c("party", "date"), sep = "_", remove = FALSE) |> 
  mutate(
    path = file.path(rawdata_path, tolower(country_iso3c), date, sprintf("%s-%s.txt", manifesto_id, language))
  )

# write to disk 
res <- map2(tmp$text, tmp$path, function(x, fp) {
  dir.create(dirname(fp), recursive = TRUE, showWarnings = FALSE)
  write_lines(x, fp)
})

# remove the processed manifestos from the corpus data frame
manifesto_sources <- anti_join(manifesto_sources, select(tmp, manifesto_id))

# download from PoliDoc Archive ----

cases <- manifesto_sources |> 
  filter(raw_text_exists == "yes") |> 
  group_by(manifesto_id) |> 
  filter("PoliDoc" %in% source) |> 
  ungroup() |> 
  filter(source == "PoliDoc") |> 
  distinct(country_iso3c, country_name, election_date, party_id, party_name, manifesto_id)


tmp <- cases |> 
  transmute(
    country_name,
    polidoc_id = sprintf("%d.000.%d.1.1.txt", party_id, year(election_date)),
    path = file.path(
      rawdata_path, 
      tolower(country_iso3c), 
      sprintf("%04d%02d", year(election_date), month(election_date)), 
      sprintf("%s-unknown.txt", manifesto_id))
  )

tmp <- split(tmp, tmp$path)

endpoint <- "https://www.mzes.uni-mannheim.de/projekte/polidoc_net/index_new.php"
res <- imap(tmp, function(x, fp) {
  resp <- GET(
    url = endpoint,
    query = list(
      "download_type" = "txt",
      "download_file" = with(x, sprintf("files/%s/txt_controlled/%s", country_name, polidoc_id)),
      "sitzungskennung" = "b1ig1kgonjbg3kfca3dvvekscr"
    ),
    content_type("text/plain"),
    user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36")
  )
  write_lines(
    x = content(resp, as = "text"),
    file = fp
  )
})


# determine missing manifestos ----

missing_manifestos <- anti_join(manifesto_sources, distinct(cases, manifesto_id))

missing_manifestos |> 
  select(country_name, party_abbrev, election_date, manifesto_id)

View(missing_manifestos)

write_tsv(missing_manifestos, file.path(data_path, "cases", "missing_manifesto_raw_texts_2024-03-20.tsv"))

