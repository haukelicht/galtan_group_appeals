# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Parse worker responses for annotation 
#'          job "grp-group-mention-anno-b2"
#' @author Hauke Licht
#' @date   2024-07-22
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

JOB_NAME = "group-mention-annotation-batch-02"

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
responses |> 
  filter(any_unsure(label))

review_cases <- tribble(
  ~sentence_id, ~issue,
  "41111_198701-124548",	'In another example healthcare providers was seen as public institutions. In this case it is further explaineed that this includes medical associations and chambers (which  I would rather have seen as public) and the pharmaeutical industry (organizational group). Should I highlight healthcare providers as organizational group or is it rather a public institution?',
  "32720_199403-283334",	'Family and mass media as organizational group?',
  "21914_200706-69452",	'the modern industries = too implicit or organizational group?',
  "61320_198411-410648",	'financial institutions and other concerned parties = organizational group?',
  "31720_200206-209296",	'highlight seperatley or as a whole? If as whole as which type of group? (explically those airlines/ship owners and carriers are meant which...)',
  "55110_201605-111780",	'highlight non-commercial consumers or not because only description of the beforementioned groups?',
  "41320_199410-130456",	'organizaitonal or social group? Plus: is a word missing?',
  "42710_200809-15093",	'unsure if this translation is correct + highlight "murderers of ...." or "our children"?',
  "87071_201109-301812",	'local media = organizational group?',
  "13951_199012-186518",	'the ugly men = generic singular?',
  "11320_200209-393314",	'a Swedish company = generic singular?',
  "22720_200205-319972",	'generic singular?',
  "61320_199611-418359",	'could be social as well as organizational group',
  "22730_201703-328499",	'unsure because sentence seems to have mistakes in it',
  "14110_201104-200091",	'generic singular?',
  "61320_198411-409837",	'would tend to organizational group (companies as consumers), but might also be social group (individual consumers)',
  "42110_199910-09220",	'Elderly, care, disability homes = public or organizational?',
  "32110_199204-281631",	'without context can\'t be sure what "all" refers to',
  "14110_200303-198330",	'organizational or social group?',
  "32720_199604-284074",	'generic singular?',
  "41320_200209-138652",	'farmers --> social or organizational group?',
  "21112_200305-54793",	'unsure about "the social society"',
  "92436_201110-363775",	'do i highlight "companies that are..." in addition to "strategic companies" or not because it is just a description of those strategic companies?',
  "95100_201604-377550",	'includes social and organizational group --> cannot highlight seperately',
  "41113_200509-140794",	'would tend to organizational group but could also be social',
  "51620_201505-241618",	'unsure about organizational and if it should be annotated at all',
  "13229_201109-192355",	'could relate to politicians or companies or trade unions, can\'t be sure without context',
  "51620_197905-222771",	'unsure if social or organizational + unsure if "the volunatry movement" should be highlighted --> is it an actual movement or just a description of volunteers/voluntary organizations?',
  "31720_198603-206828",	'or include "leaders [...] who renew....."',
  "31720_200706-213028",	'would like to highlight as one but not possible as I would tend to highlight airline as organizational, but ship owners and carriers as social group',
  "51620_199705-230186",	'organizational or social group?',
  "21112_201405-79315",	'unsure about span and if "pioneers.." should be organizational or social',
  "80710_201305-94615",	'public or organizational group?',
  "22110_198909-311479",	'organizational or social group?',
  "82710_199206-112644",	'not completely sure what the the communist nomenclature cadres --> would tend to social group',
  "35110_200502-372525",	'farmers and producers as social or organizational group?',
  "171101_201807-310629",	'convalescent homes = public or organizational?',
  "86421_200604-260891",	'word missing? (companies)',
  "32720_199604-284573",	'unsure about span and type of group --> would tend to social group ',
  "21112_198111-24038",	'farmers = social or organizational?',
  "31110_200706-211033",	'all actors includes social and organizational groups --> what should I do?',
  "13951_199409-187496",	'the privat business society = social group??',
  "34730_201206-254869",	'domestic creditors = organizational group?',
  "82721_201710-117783",	'unsure if I should highlgiht "parasites" in this case but it is a description for immigrants (or at least I assume that from the context:))',
  "71110_201212-289890",	'generic singular?',
  "80710_200506-92731",	'ethnic parties = organizational?',
  "41320_197610-119413",	'word missing? (companies)',
  "23111_198906-292143",	'organizational or social group?',
  "12951_200109-336374",	'generic singular?',
  "21112_201006-77610",	'unsure about span, individuals is social group but if highlighted seperately it does not become clear that association who advocate for... are meant',
  "21112_199505-36673",	'farmers = social or organizational?',
  "83110_201103-196808",	'social or organizational?',
  "12110_201709-346458",	'word missing? ',
  "86710_201004-268319",	'organizational or social group?',
  "51620_199705-230263",	'organizational or social group?',
  "64110_199911-352167",	'unsure about span, individuals is social group but if highlighted seperately it does not become clear that communities affected by... are meant',
  "11620_199809-392642",	'words missing?',
  "92713_200109-360618",	'social or organizational? or political?',
  "63110_201905-03860",	'those who make.... =social or organizational?',
  "14110_201904-203800",	'involves both social and organizational groups',
  "21112_198712-29815",	'social or organizational?',
  "21112_198510-26951",	'social or organizational?',
  "41113_202109-174058",	'social or organizational? --> churches as actors might indicate organizational group, community is social',
  "41113_199410-129324",	'social or organizational?',
  "43810_201110-104911",	'public or organizational group? (international organizations)',
)


jsonify <- function(x) {
  x <- as.list(x)
  x$label <- x$label[[1]]
  x$metadata <- x$metadata[[1]]
  jsonlite::toJSON(x, auto_unbox = TRUE)
}


lines <- review_cases |> 
  left_join(responses) |> 
  select(id = sentence_id, text, label, issue) |> 
  rowwise() |> 
  mutate(metadata = list(list("sentence_id" = id, "issue" = issue)), issue = NULL) |> 
  group_split() |>
  map_chr(jsonify)

lines[[1]]

fp <- file.path(data_path, "review_cases.jsonl")
write_lines(lines, fp)

# after manual review

fp <- file.path(data_path, "reviewed.jsonl")
reviewed <- parse_worker_responses(fp = fp, worker.id = 'expert', label.map = label_map)

lines <- bind_rows(
  filter(responses, !sentence_id %in% review_cases$sentence_id),
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


fp <- file.path(data_path, "review_annotations.jsonl")
write_lines(lines, fp)


