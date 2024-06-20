# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Parse worker responses for annotation job "grp_gm-gold-examples_review"
#' @author Hauke Licht
#' @date   2024-04-21
#' @update 2024-04-24
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(jsonlite)

source(file.path("code", "utils.R"))

# parse worker response files into data frame ----

JOB_NAME = "group-menion-gold-examples_review"

data_path <- file.path("data", "annotations", JOB_NAME)

# parse ----

# label map
label_map <- map_chr(read_json(file.path("data", "annotations", "label_config_review.json")), "text")
label_map <- setNames(seq_along(label_map), label_map)

responses <- parse_worker_responses(
  fp = file.path(data_path, "annotations.jsonl"),
  worker.id = "consolidated",
  label.map = label_map
)

responses <- arrange(responses, id)

# sanity checks ----

# any non-integer?
table(map_lgl(responses$annotations, function(a) any(a %% 1 !=0)))

# re-label ----

# note: to consolidate disagreeing annotations, we used a different label scheme

label_map <- c(
  "SocG"    = "social group",
  "UniSocG" = "universal social group reference",
  "SocO"    = "social organization",
  "PolG"    = "political group",
  "PolIn"   = "political institution",
  "PubIn"   = "public institution",
  "unsure"  = "unsure"
)

responses$label <- map_depth(responses$label, 2, function(x) {x[[3]] <- unname(label_map[sub("-[az]$", "", x[[3]])]); x} )

responses$annotations <- map(responses$annotations, ~ceiling(./2))

# export as TSV ----

social_label_cats <- c("social group", "universal social group reference", "social organization")

tmp <- responses |> 
  select(id, text, label) |>
  unnest_longer(label) |>
  mutate(label = map(label, setNames, nm = c("s", "e", "l"))) |> 
  unnest_wider(label) |> 
  filter(l %in% social_label_cats) |>
  mutate(mention = trimws(substr(text, s, e)))

write_tsv(tmp, file.path(data_path, "consolidated_cases.tsv"))

# write to disk ----
write_rds(responses, file.path(data_path, "worker_responses.rds"))
