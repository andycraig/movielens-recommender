library(magrittr)

args = commandArgs(trailingOnly = TRUE)

message("Loading files...")

# Load config.
config = yaml::yaml.load_file(args[[1]])

# Load movies data.
movies_df = readr::read_csv(config[["movies_csv"]])

# Load ratings data.
ratings_df = readr::read_csv(config[["ratings_csv"]],
                             col_names=c("UserID", "MovieID", "Rating", "Timestamp"))

message("Preprocessing...")
n_movies = length(unique(ratings_df$MovieID))
n_users = length(unique(ratings_df$UserID))
# Check that there are no skipped UserIDs.
assertthat::are_equal(n_users, max(ratings_df$UserID))

# Not all MovieIDs are present in the ratings CSV. 
# When we create a matrix from the ratings data,
# if we create a row for every MovieID that could exist, we will have
# rows with no ratings. To avoid this, create a second 'ID' for movies that will
# be the row of the ratings matrix corresponding to that movie.
# Then, drop the movies that are not in the ratings CSV.
# Also, not all MovieIDs in the ratings CSV are in the movies CSV,
# but we don't need to do anything about these - we leave them in the ratings
# matrix because they can add information to the factorisation.

# Create equivalence between MovieID and MovieRow.
MovieID_and_row = unique(ratings_df$MovieID) %>%
	sort() %>%
  tibble::tibble(MovieID = .,
                 MovieRow = 1:length(.)) 

# Add MovieRow to title and save.
MovieID_and_row %>%
  dplyr::inner_join(movies_df, by = "MovieID") %>%  # Add details of movies that are in the ratings file.
  readr::write_csv(config[["MovieID_and_row_file"]])

# Helper function for conversion to matrix.
matrix_by_indices = function(i, j, x, bg, nrow=length(unique(i)), ncol=length(unique(j)))
  matrix(bg, nrow = nrow, ncol = ncol) %>%
  `[<-`(cbind(i, j), x)

# Convert ratings to matrix.
ratings_mat = ratings_df %>% 
  dplyr::left_join(MovieID_and_row, by = "MovieID") %$% # Add in MovieRow.
  matrix_by_indices(MovieRow, UserID, Rating,
                    bg=NA, nrow = n_movies, ncol = n_users)

# Factorise matrix.
message("Factorising ratings matrix...")
svd_object = softImpute::softImpute(ratings_mat,
                                    rank.max = config[["NMF"]][["n_components"]])

# Check outputs are the expected size.
assertthat::are_equal(n_movies, nrow(svd_object$u))
assertthat::are_equal(n_users, nrow(svd_object$v))

# Save output.
message("Saving results...")
svd_object$u %>% write(config[["factorised_movies_file"]], ncolumns = ncol(.))
svd_object$v %>% write(config[["factorised_users_file"]], ncolumns = ncol(.))
svd_object$d %>% t() %>% write(config[["factorised_diag_file"]], ncolumns = ncol(.))

message("Done.")