library(magrittr)

args = commandArgs()

# Load config.
config = yaml::yaml.load_file(args[[1]])

# Load ratings data.
ratings_df = readr::read_csv(config[["ratings_csv"]],
                             col_names=c("UserID", "MovieID", "Rating", "Timestamp"))
n_movies = length(unique(ratings_df$MovieID)) 
n_users = length(unique(ratings_df$UserID))
# Check that there are no skipped UserIDs.
assertthat::are_equal(n_users, max(ratings_df$UserID))
# Not all MovieIDs are present.
# So that there are not all-missing rows/columns in the ratings matrix, 
# create a second ID for movies: MovieRow.
if (n_movies < max(ratings_df$MovieID)) {
  # Create equivalence between MovieID and MovieRow.
  MovieID_and_row = unique(ratings_df$MovieID) %>%
    tibble::tibble(MovieID = .,
                   MovieRow = 1:length(.))
  # Save this.
  MovieID_and_row %>% readr::write_csv(config[["MovieID_and_row_file"]])
  ratings_df = ratings_df %>% dplyr::left_join(MovieID_and_row)
}

# Convert to matrix.
ratings_mat = ratings_df %$% 
  Matrix::sparseMatrix(i = MovieRow, j = UserID, x = Rating) %>% # i: row; j: col
  as.matrix()

# Factorise matrix.
svd_object = softImpute::softImpute(ratings_mat, 
                                    rank.max = config[["NMF"]][["n_components"]])

# Check outputs are the expected size.
assertthat::are_equal(n_movies, nrow(svd_object$u))
assertthat::are_equal(n_users, nrow(svd_object$v))

# Save output.
write(svd_object$u, config[["factorised_movies_file"]])
write(svd_object$v, config[["factorised_users_file"]])
write(t(svd_object$d), config[["factorised_diag_file"]])
