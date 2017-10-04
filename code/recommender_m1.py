from recommender import Recommender
import numpy as np

# Setup.
# Load config.
#TODO
# Set up recommender.
factorised_users = np.loadtxt(config['factorised_users_file'])
factorised_movies = np.loadtxt(config['factorised_movies_file'])
factorised_diag = np.loadtxt(config['factorised_diag_file'])
recommender = Recommender(U = factorised_movies,
							V = factorised_users,
							S = factorised_diag)

# Make a prediction.
#TODO new_user_ratings =
predictions = recommender.predict_for_new_v(new_user_ratings)
#TODO return or print.
