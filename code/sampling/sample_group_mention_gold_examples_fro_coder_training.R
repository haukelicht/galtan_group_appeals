# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Sample gold example instances for coder training
#' @author Hauke Licht
#' @date   2024-04-29
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

# load required packages
library(readr)
library(dplyr)
library(tidyr)
library(purrr)

# load the data ----

data_path <- file.path("data", "annotations")

fp <- file.path(data_path, "group-menion-gold-examples", "sample.tsv")
gold_examples <- read_tsv(fp)

fp <- file.path(data_path, "group-menion-gold-examples", "worker_responses.rds")
approved_gold_examples <- read_rds(fp)
# approved_gold_examples$label[[17]]


fp <- file.path(data_path, "group-menion-gold-examples", "agreement_cases.tsv")
agreement_cases <- read_tsv(fp)

fp <- file.path(data_path, "group-menion-gold-examples_review", "consolidated_cases.tsv")
consolidated_cases <- read_tsv(fp)

used_in_coding_instructions <- c(
  "11110_201809-399610",
  "11320_197009-389755",
  "11320_197009-389755",
  "11320_200209-393252",
  "11620_198509-390516",
  "11620_198509-390516",
  "11620_201009-395461",
  "12951_198509-328910",
  "13230_201109-193060",
  "13720_200711-191719",
  "22110_201006-323094",
  "31720_199303-208620",
  "34710_200709-253646",
  "41113_201709-164517",
  "41320_201709-166938",
  "53110_200205-277092"
)
# aggregate annotations and classify into sets ----


fitler_labels <- function(annotation, .keep) {
  idxs <- map_lgl(annotation, ~ .x[[3]] %in% .keep)
  return(annotation[idxs])
}

recode_labels <- function(annotation, .map) {
  map(annotation, function(a) c(a[1:2], unname(.map[a[[3]]])))
}

social_label_cats <- c("social group", "universal social group reference", "social organization")
recode_map <- c(
  "social group" = "social group", 
  "universal social group reference" = "social group", 
  "social organization" = "social group"
)

tmp <- approved_gold_examples |> 
  mutate(
    # discard non-social label class annotations
    label = map(label, fitler_labels, .keep = social_label_cats),
    label = map(label, recode_labels, .map = recode_map),
    any_social = lengths(label) > 0
  ) |>
  # head(17) |>
  # last() |> as.list() |> 
  group_by(id, text) |>
  # keep only doubly annotated gold examples
  filter(n_distinct(annotator) == 2) |> 
  summarise(
    unanimous = do.call("identical", label),
    any_social = any(any_social),
    metadata = metadata[1],
    label = ifelse(
      unanimous,
      label[1],
      list()
    )
  ) |> 
  ungroup()

tmp <- tmp |> 
  # discard examples used in coding instructions
  filter(!id %in% used_in_coding_instructions) |>
  mutate(
    set = case_when(
      !any_social & unanimous ~ "negative",
      id %in% agreement_cases$id ~ "agreement",
      id %in% consolidated_cases$id ~ "disagreement"
    )
  ) |> 
  filter(!is.na(set))

with(tmp, table(any_social, set))

gold_labels <- consolidated_cases |> 
  distinct() |> 
  group_by(id) |>
  select(-text, -mention) |>
  nest(gold = s:l) |> 
  ungroup() |> 
  mutate(
    gold = map(gold, ~map(transpose(as.list(.)), unname)),
    gold = map(gold, recode_labels, .map = recode_map)
  )

tmp <- tmp |> 
  left_join(gold_labels, by = "id") |>
  mutate(
    label = ifelse(set == "disagreement", gold, label),
    gold = NULL
  )

set.seed(1234)
tmp <- tmp |> 
  group_by(set) |>
  # reshuffle within set
  sample_frac(1.0) |> 
  mutate(
    # get first half in first round, second in second round
    test_round = ifelse(row_number() <= ceiling(n()/2), 1, 2)
  ) |> 
  # reshuffle "globally"
  ungroup() |> 
  sample_frac(1.0) 
  
with(tmp, table(test_round, set))

# save to disk ----

JOB_NAME = "group-menion-coder-training"
data_path <- file.path("data", "annotations", JOB_NAME)
dir.create(data_path, recursive = TRUE, showWarnings = FALSE)

jsonify <- function(x) {
  jsonlite::toJSON(x, dataframe = "values", auto_unbox = TRUE)
}

OVERWRITE = FALSE
for (r in 1:2) {
  json_lines <- tmp |>
    filter(test_round == r) |>
    select(id, text, metadata) |>
    mutate(
      r_ = row_number(),
      label = list(list())
    ) |> 
    group_by(r_) |> 
    group_split(.keep = FALSE) |>
    map(flatten) |> 
    map(as.list) |> 
    map_chr(jsonify)
  
  # write to disk
  fn <- sprintf("sample_round%d.manifest", r)
  fp <- file.path(data_path, fn)
  if (!file.exists(fp) | OVERWRITE) {
    write_lines(json_lines, fp)
  }
  
  fn <- sprintf("sample_round%d.tsv", r)
  fp <- file.path(data_path, fn)
  if (!file.exists(fp) | OVERWRITE) {
    tmp |>
      filter(test_round == r) |>
      select(id, text, metadata) |> 
      unnest_wider(metadata) |> 
      write_tsv(fp)
  }
}
  
json_lines <- tmp |>
  select(id, text, metadata, label, test_round) |>
  group_by(row_number()) |> 
  group_split(.keep = FALSE) |>
  # head(17) |> first() |> as.list()
  map(flatten) |> 
  map(as.list) |> 
  map_chr(jsonify)

# write to disk
fn <- "gold_annotations.manifest"
fp <- file.path(data_path, fn)
if (!file.exists(fp) | OVERWRITE) {
  write_lines(json_lines, fp)
}

fn <- "gold_annotations.rds"
fp <- file.path(data_path, fn)
if (!file.exists(fp) | OVERWRITE) {
  tmp |>
    select(id, text, metadata, label, test_round) |> 
    write_rds(fp)
}




