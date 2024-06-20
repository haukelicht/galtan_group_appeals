# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Parse worker responses for annotation job
#' @author Hauke Licht
#' @date   2024-04-16
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

JOB_NAME = "group-menion-gold-examples"

data_path <- file.path("data", "annotations", JOB_NAME)

json_files <- list.files(
  file.path(data_path, "annotations")
  , pattern = "\\.jsonl$"
  , recursive = TRUE
  , full.names = TRUE
)

# parse ----

# label map
label_map <- map_chr(read_json(file.path("data", "annotations", "label_config.json")), "text")
label_map <- setNames(seq_along(label_map), label_map)

# fp <- json_files[1]
# worker.id = "lhauke"
worker_responses <- parse_workers_responses(
  # fp = json_files[1], 
  # fp = json_files[2], 
  json_files,
  sub("\\.jsonl$", "", basename(json_files)), 
  label.map = label_map
)

worker_responses <- arrange(worker_responses, id)

# keep only multiply annotated
worker_responses |> count(id) |> count(n)

worker_responses <- worker_responses |> group_by(id) |> filter(n_distinct(annotator) > 1) |> ungroup()

# sanity checks ----

# any non-integer?
table(map_lgl(worker_responses$annotations, function(a) any(a %% 1 !=0)))

# write to disk ----

dir.create(file.path(data_path, "parsed"), showWarnings = FALSE)
write_rds(worker_responses, file.path(data_path, "worker_responses.rds"))
