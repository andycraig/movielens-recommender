library(magrittr)

# Load config.
config = yaml::yaml.load_file('config-1m.yaml')

# Load ratings data.
ratings_df = readr::read_csv(config[["ratings_csv"]],
                 col_names=c("UserID", "MovieID", "Rating", "Timestamp"))

# Convert to matrix.
ratings_mat = ratings_df %$% 
  Matrix::sparseMatrix(i = UserID, j = MovieID, x = Rating) %>% # i: row; j: col
  as.matrix()

# Factorise matrix.
svd_object = softImpute::softImpute(ratings_mat, rank.max = 4)
