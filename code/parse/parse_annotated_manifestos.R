# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #  
#
#' @title  Parse group appeals annotations from annotated docx files
#' @author Hauke Licht
#' @date   2023-05-31
#' @update 2024-03-20
#
# +~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~+~ #

# setup ----

# load pkgs 
library(dplyr)
library(tidyr)
library(purrr)
library(stringi)
library(stringr)
library(officer)
library(ggplot2)

data_path <- file.path("data",  "manifestos")
manifestos_path <- file.path(data_path, "annotated")

# read data ----

docx_files <- list.files(manifestos_path, pattern = "^[^~].+\\.docx$", full.names = TRUE)

names(docx_files) <- sub("\\.docx$", "", basename(docx_files))
docs <- map(docx_files, read_docx)

length(docs)
i <- 7
names(docs)[i]
doc <- docs[[i]]

cont <- as_tibble(docx_summary(doc))

# table(unlist(str_extract_all(cont$text, "\\[[^\\]]+")))
table(unlist(str_extract_all(cont$text, "\\[Group[^\\]]+\\]")))

#' issues
#' - not a single group appeal 
#'    - Manifesto 1985 Progress Party Norway_RECODED
#'    - Manifesto 1989 Progress Party Norway_RECODED
#'    - Manifesto 1991 New Democracy Sweden_RECODED
#'    - Manifesto 1994 National Alliance Italy_RECODED
#'    - Manifesto 1994 Northern League Italy_RECODED
#'    - Manifesto 1999 Swiss Peoples Party_CODED
#'    - Manifesto 2001 Danish Peoples Party
#'    - Manifesto 2002 List Pim Fortuyn Netherlands_Leonce
#'    - Manifesto 2016 Republicans USA_RECODED

s <- grep("^\\[Here does the program start", cont$text)

if (length(s) == 0)
  s <- 0L

cont <- cont[(s+1):nrow(cont), ]

cont <- cont[!grepl("^\\h*$", cont$text), ]

# (par <- cont$text[149])
# 
# sents <- stri_split_boundaries(par, type = "sentence", locale = "en_EN")[[1]]
# 
# (sent <- sents[1])

parse_sentence <- function(sent) {
  # # test
  # sent <- "abcd [s]def [label]ghij [s]klmn [label]nop"
  # # idx:   1       4 6        7       11 14
  
  if (!grepl("[s]", sent, fixed = TRUE))
    return(tibble(text = sent, annotations = list(NULL)))
  
  # identify starts of spans
  s <- str_locate_all(sent, "\\[s\\]")[[1]][ ,"start"]
  
  # segment sentence into subsets with spans at beginning
  chars <- strsplit(sent, "")[[1]]
  segs <- map2(c(1, s), c(s-1, nchar(sent)), ~paste(chars[.x:.y], collapse = ""))
  if (segs[[1]] == "[")
    segs[[1]] <- NULL
  
  # iterate over segments
  segs <- map_dfr(segs, function(seg) { # seg <- segs[[2]]
    
    # remove the span start special token
    seg <- str_remove(seg, "^\\[s\\]\\h*")
    
    # extract the annotation note
    label <- str_extract(seg, "\\[[^\\]]+\\]") # test: should be "[label]"
    
    # get the extend of the annotation note in the text
    ext <- str_locate(seg, "\\h+\\[[^\\]]+\\]")
    
    # compute the span end
    e <- unname(ext[1, "start"]-1) # test: should be 3
    (span <- substr(seg, 1, e))
    
    # remove the annotation note
    seg <- str_remove(seg, "\\h*\\[[^\\]]+\\]")
    
    # return the relevant data
    tibble(text = seg, s = 0, e, label, span)
  })
  
  # compute the number of characters in segments
  segs$nchar <- c(0, nchar(segs$text[-nrow(segs)]))
  
  # increment span starts by previous segment lengths
  segs$s <- segs$s + segs$nchar
  segs$s <- cumsum(segs$s)+1L
  # increment span ends by previous segment lengths
  segs$e <- segs$s+segs$e-1L
  
  # extract annotations
  annotations <- segs[!is.na(segs$label), c("s", "e", "label", "span")]
  # test: should be
  # # A tibble: 2 × 3
  #       s     e label     
  #   <dbl> <dbl> <chr>     
  # 1     4     6 " [label]"
  # 2    11    14 " [label]"
  
  # return
  return(
    tibble(
      text = paste(segs$text, collapse = "") # test: should be "abcdefghijklmnop"
      , annotations = list(annotations)
    )
  )
}

parse_paragraph <- function(par) {
  sents <- stri_split_boundaries(par, type = "sentence", locale = "en_EN")[[1]]
  out <- map_dfr(sents, parse_sentence, .id = "sentence_nr")
  out$sentence_nr <- as.integer(out$sentence_nr)
  return(out)
}

# View(map_dfr(cont$text[1:10], parse_paragraph, .id = "par_nr"))

nrow(cont)
out <- map_dfr(cont$text, parse_paragraph, .id = "paragraph_nr")
View(out)

tmp <- out |> 
  filter(!map_lgl(annotations, is.null)) |> 
  unnest(annotations) |> 
  filter(grepl("^\\[Group", label)) |> 
  mutate(
    group_types = strsplit(gsub("^\\[Group appeal:\\s*|\\s*\\]$", "", label), "[;,]\\s*"),
    polarity = group_types |> 
      map(str_extract, pattern = "_[pnt]{1,2}$") |> 
      map(unique),
    n_polarities = lengths(polarity),
    group_types = group_types |> 
      map(str_remove, pattern = "_[pnt]{1,2}$") |> 
      map(tolower)
  ) |> 
  filter(n_polarities == 1) |> 
  # with(table(n_polarities))
  mutate(
    polarity = sub("^_", "", map_chr(polarity, first)),
    n_polarities = NULL
  ) #|> 
  
with(tmp, table(polarity)) |> prop.table()
# note: strong class imbalance!

# tmp |> 
#   filter(polarity == "n") |> 
#   with(table(label))

tmp |> 
  select(group_types) |> 
  unnest(group_types) |> 
  count(group_types) |> 
  arrange(desc(n)) |> 
  ggplot(aes(y = reorder(group_types, n), x = n)) + 
    geom_col() + 
    labs(y = NULL)


