# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Parse worker responses for annotation 
#'          job "grp-group-mention-anno-b3"
#' @author Hauke Licht
#' @date   2024-08-27
#' @update 2024-09-02
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

JOB_NAME = "group-mention-annotation-batch-03"

data_path <- file.path("data", "annotations", JOB_NAME)

# label map
label_map <- map_chr(read_json(file.path("data", "annotations", "label_config.json")), "text")
label_map <- setNames(seq_along(label_map), label_map)

# parse coder annotations and gold annotations ----

responses <- parse_workers_responses(
  files = c(
    file.path(data_path, "annotations", "emarie.jsonl")
  ),
  worker.ids = c("annotator"),
  label.map = label_map
)

responses

# create separate job to review unsure instances ----

any_unsure <- Vectorize(function(x) {
  any(map_lgl(map_chr(x, 3), ~.x == "unsure"))
})

# review_cases <- responses

review_cases <- tribble(
  ~sentence_id, ~issue,
  '41953_201309-162301',	'unsure about "the private major creditors" of banks --> social or organizatonal',
  '11320_197009-389783',	'"social insitutions" --> public or organizational groups?',
  '12110_201709-346252',	'word: company missing?',
  '21914_198712-30776',	'small start ups?',
  '41320_201309-157942',	'board of directors as group? (social or organizational)',
  '22711_198909-311952',	'word: company missing?',
  '92436_201510-367533',	'"companies in which state treasury holds miniority shares" --> as organizational or public',
  '92436_200109-360138',	'"The management or radio" --> organizational? (wording suggest that a group of people is meant rather than the action, but not sure)',
  '41320_201309-159335',	'probably international state organizations (EU..) but not completly sure',
  '51620_200505-235219',	'unsure about "the further education sector" --> tend not to highlight',
  '11320_201409-397145',	'"smith companies"?',
  '41521_200909-150058',	'word: company missing?, unsure about crafts',
  '21914_198510-27769',	'"start up small entroprreneurs"? ',
  '13951_197501-181770',	'unsure if the schools are private or public with independence',
  '42420_201710-19640',	'word: company missing?',
  '82110_201005-115869',	'word: company missing?',
  '87071_201010-301763',	'word: company missing?',
  '31110_199705-208695',	'word: company missing?',
  '41320_197610-120045',	'unsure what our partners relates to',
  '11620_197909-390209',	'unsure about "the moderates"',
  '55110_201605-111381',	'unsure if those international organizations are public or organizational',
  '61620_200011-421713',	'unsure because organizational and social in one',
  '41521_200209-139847',	'research institutions = public or organizational? (not universities)',
  '51620_197006-219676',	'unsure what cost rent groups are',
  '11110_200209-393030',	'word: companies missing?',
  '51320_197905-222215',	'unsure about committees',
  '12951_200109-335192',	'market participants could include both types of group',
  '41113_200909-145013',	'could include both types of groups',
  '51320_199705-229351',	'unsure if it should be highlighted',
  '97710_201112-388835',	'unsure if it should be highlighted',
  '22711_199405-314706',	'word: companies missing?',
  '41521_198010-121858',	'unsure if other groups should be social or organizational',
  '21111_200305-53692',	'unsure if pararegional companies are public',
  '23113_200406-298591',	'unsure about committees',
  '21914_200706-66776',	'unsure about pressure groups',
  '64110_200509-352474',	'spam',
  '41111_198701-124011',	'unsure what kind emancipatory forces are meant (social or organizational)',
  '23113_201310-300669',	'social?',
)


jsonify <- function(x) {
  x <- as.list(x)
  x$label <- x$label[[1]]
  x$metadata <- x$metadata[[1]]
  jsonlite::toJSON(x, auto_unbox = TRUE)
}

unsure_cases <- responses |> 
  filter(any_unsure(label)) |> 
  anti_join(review_cases) |> 
  mutate(issue = "<has unsure>")

lines <- review_cases |> 
  left_join(responses) |> 
  bind_rows(unsure_cases) |> 
  arrange(id) |> 
  select(id = sentence_id, text, label, issue) |> 
  rowwise() |> 
  mutate(metadata = list(list("sentence_id" = id, "issue" = issue)), issue = NULL) |> 
  group_split() |>
  map_chr(jsonify)

lines[[1]]


fp <- file.path(data_path, "review_cases.jsonl")
if (!file.exists(fp)) {
  write_lines(lines, fp)
}

# after manual review

fp <- file.path(data_path, "reviewed.jsonl")
reviewed <- parse_worker_responses(fp = fp, worker.id = 'expert', label.map = label_map)

  
lines <- responses |> 
  filter(!sentence_id %in% review_cases$sentence_id) |> 
  bind_rows(
    responses |> 
      subset(sentence_id %in% review_cases$sentence_id, c(sentence_id, text_nr)) |> 
      inner_join(
        reviewed |> select(-text_nr) |> rename(sentence_id = id)
        , by = "sentence_id"
      )
  ) |> 
  arrange(text_nr) |> 
  select(id = sentence_id, text, label, metadata) |> 
  rowwise() |> 
  group_split() |>
  map_chr(jsonify)

# x <- fromJSON(lines[1], simplifyVector = FALSE)
# do.call(substr, c(x$text, x$label[[1]][1:2]))

fp <- file.path(data_path, "review_annotations.jsonl")
write_lines(lines, fp)


