
source_path <- "data/manifestos/missing"

overview <- read_tsv(file.path(source_path, "Overview MM.txt"))

tmp <- overview |>
  transmute(country = tolower(country_iso3c), manifesto_id) |> 
  separate(manifesto_id, c("party", "date"), sep = "_", remove = FALSE) |> 
  mutate(file_name = sprintf("%s_%s", party, substr(date, 1, 4)))

dest <- "data/manifestos/raw"
src <- file.path(source_path, paste0(tmp$file_name, ".pdf"))
tgt <- with(tmp, file.path(dest, country, date, paste0(manifesto_id, ".pdf")))
map2(src, tgt, file.copy, overwrite = TRUE)

src <- file.path(source_path, paste0(tmp$file_name, ".docx"))
tgt <- with(tmp, file.path(dest, country, date, paste0(manifesto_id, ".docx")))
map2(src, tgt, file.copy, overwrite = TRUE)
