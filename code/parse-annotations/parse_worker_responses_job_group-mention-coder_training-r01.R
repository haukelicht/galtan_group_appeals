# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Parse worker responses for annotation 
#'          job "grp_group-mention-coder-training-r01"
#' @author Hauke Licht
#' @date   2024-05-01
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

JOB_NAME = "group-menion-coder-training"

data_path <- file.path("data", "annotations", JOB_NAME)

# label map
label_map <- map_chr(read_json(file.path("data", "annotations", "label_config.json")), "text")
label_map <- setNames(seq_along(label_map), label_map)

# parse coder annotations and gold annotations ----

responses <- parse_workers_responses(
  files = c(
    file.path(data_path, "annotations", "sample_round1_emarie.jsonl"),
    file.path(data_path, "gold_annotations.manifest")
  ),
  worker.ids = c("emarie", "gold"),
  label.map = label_map
)

with(responses, table(annotator, test_round, useNA = "ifany"))

# compute agreement ----

social_label_cats <- label_map[c("social group", "social organization")]

tmp <- responses |> 
  mutate(
    # true labels
    any_social = map_lgl(annotations, ~any(.)),
    any_social = ifelse(annotator == "gold", any_social, NA),
    # annotations
    any_coded = map_lgl(annotations, ~any(. %in% label_map)),
    any_coded = ifelse(annotator != "gold", any_coded, NA),
    any_social_coded = map_lgl(annotations, ~any(. %in% social_label_cats)),
    any_social_coded = ifelse(annotator != "gold", any_social_coded, NA),
  ) |> 
  group_by(id, text) |> 
  summarise(
    n_annotators =  n_distinct(annotator)
    , annotators = list(c(annotator))
    , annotations = list(do.call(rbind, annotations))
    , any_social = all(any_social, na.rm = TRUE)
    , any_coded = all(any_coded, na.rm = TRUE)
    , any_social_coded = all(any_social_coded, na.rm = TRUE)
  ) |> 
  ungroup() |> 
  filter(n_annotators == 2)

nrow(tmp)

# compute inter-coder agreement metrics

irr_metrics <- tmp |> 
  mutate(
    # agreement in sentences computed at token level
    agreement = map(annotations, compute_metrics, focal.cats = social_label_cats)
  ) |> 
  unnest_wider(agreement, names_sep = "_")

# sentence level agreemnent (diagonal: TN and TP, off-diagonal: FN and FP)
tab <- with(irr_metrics, table(gold = any_social, coder = any_coded))
prop.table(tab, 1) |> round(2)

summary(select(irr_metrics, starts_with("agreement_"))) |> t()
map(select(irr_metrics, starts_with("agreement_")), quantile, p = .1)

# inspect
irr_metrics |> 
  # focus on sentences with any annotations by coder
  filter(any_coded) |>
  group_by(any_social) |> 
  summarise(n = n(), across(starts_with("agreement"), compose(list, summary))) |> 
  pivot_longer(starts_with("agreement")) |> 
  unnest_wider(value) |> 
  filter(!(!any_social & name == "agreement_binary")) |> 
  mutate(across(-c(any_social:name), as.vector)) |> 
  arrange(desc(name), !any_social)


# sentences with 100 % agreement 
irr_metrics |> 
  filter(agreement_all == 100) |> 
  count(any_social)

# sentences with some disagreement 
irr_metrics |> 
  filter(agreement_all < 100) |> 
  count(any_social)

# export for coder review
tbd <- irr_metrics |> 
  filter(agreement_all < 100) |> 
  select(id, text, agreement_all) |>
  left_join(select(responses, id, label, annotator), multiple = "all") |> 
  mutate(
    mentions = map2_chr(
      label, 
      text, 
      function(l, t) {
        paste(
          map_chr(
            l, 
            function(.l) {
              sprintf(
                "%s [%s]", 
                sQuote(trimws(substr(t, .l[[1]], .l[[2]]))), 
                .l[[3]]
              )
            }
          ), 
          collapse = "; "
        )
      }
    ) 
  ) |> 
  select(id, text, annotator, mentions, agreement_all) |>
  pivot_wider(names_from = "annotator", values_from = "mentions") 

comments <- read_tsv(file.path(data_path, "review", "round1_coder_comments.tsv"), col_select = 1:2)

tbd <- tbd |> 
  left_join(comments, by = "id") |> 
  select(id, text, emarie, gold, agreement_all, comment)
  
View(tbd)

# export as TSV
fp <- file.path(data_path, "review", "round1_to_be_discussed.tsv")
write_tsv(tbd, fp)



