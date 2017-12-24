library(tidyverse)
library(magrittr)

config = yaml::yaml.load_file("config-1m.yaml")
m = readr::read_csv(config[["MovieID_and_row_file"]])
svd_u = scan(config[["factorised_movies_file"]]) %>%
  matrix(ncol = config[["NMF"]][["n_components"]])
svd_s = scan(config[["factorised_diag_file"]])

m$x = svd_u[,1]
m$y = svd_u[,2]

m[1:100,] %>% 
  ggplot(aes(x=x, y=y, label=Title)) + 
  geom_point() + 
  geom_text()
