require(jsonline, quietly = TRUE)
require(purrr, quietly = TRUE)
require(irr, quietly = TRUE)

parse_sequence_annotations <- function(x, .i = -1, label.map) {
  x$text_nr <- .i
  x$tokens <- strsplit(x$text, " ")
  n_ <- nchar(x$text)
  
  tmp <- rep(0L, nchar(x$text))
  for (l in x$label) # add one to indexes because 0-indexing
    tmp[ (l[[1]]+1):l[[2]] ] <- label.map[ l[[3]] ]
  
  # get word indexes
  word_idxs <- cumsum(strsplit(x$text, "")[[1]] == " ")
  # e.g., strsplit(x$text, "")[[1]][word_idxs == 1]
  # # split into words
  # chars <- unname(split(strsplit(x$text, "")[[1]], word_idxs))
  # split by words
  tmp <- unname(split(tmp, word_idxs))
  # ignore leading white spaces
  tmp[-1] <- map(tmp[-1], `[`, -1)
  # get median annotation
  x$annotations <- lapply(tmp, unique)
  if (any(idxs <- lengths(x$annotations) > 1)) {
    for (i in which(idxs)) {
      tab <- sort(table(tmp[[i]]), decreasing = TRUE)
      if ( tab[1] > tab[2] ) {
        x$annotations[[i]] <- as.integer(names(tab[1]))
      } else if ( i == length(tmp) ) {
        x$annotations[[i]] <- tmp[[i]][1]
      } else if ( x$annotations[[i-1]] == tmp[[i]][1] ) {
        x$annotations[[i]] <- tmp[[i]][1]
      } else if ( x$annotations[[i+1]] == tmp[[i]][length(tmp[[i]])] ) {
        # note: not sure about this!
        x$annotations[[i]] <- tmp[[i]][length(tmp[[i]])]
      } else {
        x$annotations[[i]] <- mean(tmp[[i]])
      }
    }
  }
  x$annotations <- list(unlist(x$annotations))
  if ("metadata" %in% names(x))
    x$metadata <- list(x$metadata)
  x$annotations <- list(unlist(x$annotations))
  x$label <- list(x$label)
  return(as_tibble(x))
}

# helper reading functions
parse_worker_responses <- function(fp, worker.id, label.map) {
  
  lines <- read_lines(fp)
  cont <- map(lines, fromJSON, simplifyVector = FALSE)
  out <- imap_dfr(cont, parse_sequence_annotations, label.map=label.map)
  out$annotator <- worker.id
  
  return(out)
}

parse_workers_responses <- function(files, worker.ids, label.map) map2_dfr(files, worker.ids, parse_worker_responses, label.map = label.map)


compute_metrics <- function(x, focal.cats) {
  out <- list()
  out$all <- irr::agree(t(x))$value
  # using only focal label categories
  x[!x %in% focal.cats] <- 0
  out$focal <- irr::agree(t(x))$value
  return(out)
}
