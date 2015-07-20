library(XML)
library(RCurl)
library(rpubchem)

url <- "http://en.wikipedia.org/wiki/Dictionary_of_chemical_formulas"
tables <- readHTMLTable(url)
# Remove first entry (table of contents)
tables[[1]] <- NULL

CASids <- lapply(tables, function (x) {
  if(dim(x)[2] == 3) {
    return(as.character(x[,3]))
  }
  })
# Combine into one list
CASids <- unlist(CASids, use.names = FALSE)

# Convert CAS to pubchem CID
casToCids <- function(query) {
  baseUrl <- "http://www.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pccompound&retmax=100&term="
  queryUrl <- sprintf("%s%s", baseUrl, query)
  xmlResponse <- xmlParse(getURL(queryUrl))
  cids <- sapply(xpathSApply(xmlResponse, "//Id"), function(x) {xmlValue(x)})
  return(cids)
}

cids <- lapply(CASids, casToCids)

# Extract only the ids for which a single match was found
cids_clean <- lapply(cids, function(x) {if(length(x) == 1) return(x)})
cids_clean <- unlist(cids_clean)

# This can now be used to get the compound information from pubchem
