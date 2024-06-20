# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Select samples for review from experts' gold examples annotations
#' @author Hauke Licht
#' @date   2024-04-16
#' @update 2024-04-21
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup  ----

library(readr)
library(jsonlite)
library(dplyr)
library(tidyr)
library(purrr)
library(ggplot2)

source(file.path("code", "utils.R"))

JOB_NAME = "group-menion-gold-examples"

label_map <- map_chr(read_json(file.path("data", "annotations", "label_config.json")), "text")
label_map <- setNames(seq_along(label_map), label_map)

data_path <- file.path("data", "annotations", JOB_NAME)

# read worker responses ----

worker_responses <- read_rds(file.path(data_path, "worker_responses.rds"))

# compute agreement estimates ---

tmp <- worker_responses |> 
  group_by(id, text) |> 
  summarise(
    tokens = list(tokens[[1]])
    , n_annotators =  n_distinct(annotator)
    , annotators = list(c(annotator))
    , annotations = list(do.call(rbind, annotations))
  ) |> 
  ungroup()

# compute inter-coder agreement metrics
  
# note: to select sentences for review, we take the vanilla (not chance-adjusted)
#  agreement because 
#   (a) we still need to figure out how to compute chance-adjusted 
#        agreement for sequence labeling, and
#   (b) we don't believe our RAs' choose labels at random if unsure anyways

social_label_cats <- label_map[c("social group", "unsure", "universal social group reference", "social organization")]

irr_metrics <- tmp |> 
  filter(n_annotators > 1) |> 
  mutate(
    # how many tokens highlighted (cross-annotator average)
    prop_unlabeled = map_dbl(annotations, ~mean(. == 0))
    # none highlighted 
    , all_unlabeled = prop_unlabeled == 1.0
    # any "social group" annotations 
    , any_social = map_lgl(annotations, ~any(. %in% social_label_cats))
    # agreement in sentences computed at token level
    , agreement = map(annotations, compute_metrics, focal.cats = social_label_cats)
  ) |> 
  unnest_wider(agreement, names_sep = "_")


tab <- with(irr_metrics, table(all_unlabeled, any_social))
prop.table(tab) 
# note: prevalence of (universal) social group/organization mentions in stratified selective samples at ~36%

summary(select(irr_metrics, starts_with("agreement_"))) |> t()
map(select(irr_metrics, starts_with("agreement_")), quantile, p = .1)

irr_metrics |> 
  select(id, starts_with("agreement")) |> 
  pivot_longer(starts_with("agreement")) |> 
  ggplot(aes(x = value, fill = name)) + 
    geom_density(color = NA, alpha = .5) + 
    scale_x_continuous(breaks = seq(0, 100, 10))+ 
    scale_fill_grey()

# inspect
irr_metrics |> 
  filter(!all_unlabeled) |> 
  # group_by(job, any_social_groups) |> 
  group_by(any_social) |> 
  summarise(n = n(), across(starts_with("agreement"), compose(list, summary))) |> 
  pivot_longer(starts_with("agreement")) |> 
  unnest_wider(value) |> 
  filter(!(!any_social & name == "agreement_binary")) |> 
  mutate(across(-c(any_social:name), as.vector)) |> 
  arrange(desc(name), !any_social)

# select agreement cases for drawing positive examples ----

# extract disagreement cases
set.seed(1234)
these <- irr_metrics |> 
  # subset to sentences with any annotations in social categories
  filter(any_social) |>
  # subset to sentences with 100 % agreement 
  filter(agreement_focal == 100) |> 
  # reshuffle
  sample_frac(1.0) |> 
  # extract sentences' IDs
  pull(id)

length(these)

tmp <- worker_responses |> 
  filter(id %in% these) |> 
  group_by(id, text) |> 
  slice(1) |> 
  select(id, text, label) |>
  unnest_longer(label) |>
  mutate(label = map(label, setNames, nm = c("s", "e", "l"))) |> 
  unnest_wider(label) |> 
  filter(l %in% names(social_label_cats)) |>
  mutate(mention = trimws(substr(text, s, e)))

write_tsv(tmp, file.path(data_path, "agreement_cases.tsv"))

# select disagreement cases for review ----

# OPEN QUESTIONS: 
#  - Focus review on disagreement on 'social groups' category? No.
#  - Discard 'unsure' codings first? No

# extract disagreement cases
these <- irr_metrics |> 
  # subset to sentences with any annotations
  filter(!all_unlabeled) |>
  # subset to sentences with less than 90% agreement
  filter(agreement_all < 100) |> 
  # # stratify by labeling job
  # group_by(job) |> 
  # # sample 50% of sentences
  # sample_frac(.50) |> 
  # ungroup() |> 
  # # reshuffle
  sample_frac(1.0) |> 
  # extract sentences' IDs
  pull(id)

length(these)

label_map <- c(
  "social group" = "SocG",
  "universal social group reference" = "UniSocG",
  "social organization" = "SocO",
  "political group" = "PolG",
  "political institution" = "PolIn",
  "public institution" = "PubIn",
  "unsure" = "unsure"
)

# note: to blind te reviewer towards annotators, I use random letters
annotators_map <- c("rleonce" = "a", "lhauke" = "z")

tmp <- worker_responses |> 
  filter(id %in% these) |> 
  select(id, text, annotator, label) |> 
  unnest_longer(label) |> 
  filter(lengths(label) > 0) |> 
  mutate(label = map(label, setNames, nm = c("s", "e", "l"))) |> 
  unnest_wider(label) |>
  # with(table(l))
  mutate(l = sprintf("%s-%s", label_map[l], annotators_map[annotator])) |> 
  select(-annotator) |> 
  nest(label = c(s, e, l))

# inspect
table(unlist(map(tmp$label, "l")))

jsonify <- function(x) {
  jsonlite::toJSON(x, dataframe = "values", auto_unbox = TRUE)
}

json_lines <- tmp |> 
  group_split(id) |>
  map(flatten) |> 
  map(as.list) |> 
  map_chr(jsonify)

# inspect
length(json_lines)
i_ <- sample(seq_along(json_lines), 1)
json_lines[[i_]]

# write to disk
JOB_NAME = "group-menion-gold-examples_review"

data_path <- file.path("data", "annotations", JOB_NAME)
dir.create(data_path, recursive = TRUE, showWarnings = FALSE)

fn <- "sample.manifest"
write_lines(json_lines, file.path(data_path, fn))

foo <- read_lines(file.path(data_path, "sample_2024-04-16.manifest"))
ids <- foo |> map(fromJSON) |> map_chr("id")

json_lines <- tmp |> 
  filter(!id %in% ids) |>
  group_split(id) |>
  map(flatten) |> 
  map(as.list) |> 
  map_chr(jsonify)

json_lines[1]

write_lines(json_lines, file.path(data_path, "update.manifest"))


