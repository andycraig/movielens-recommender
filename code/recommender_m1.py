from recommender import Recommender
import numpy as np
import pandas as pd

class MovieLensRecommender:

	def __init__(self, config):
		factorised_users = np.loadtxt(config['factorised_users_file'])
		factorised_movies = np.loadtxt(config['factorised_movies_file'])
		factorised_diag = np.loadtxt(config['factorised_diag_file'])
		self.movie_df = pd.read_csv(config['MovieID_and_row_file'])
		self.recommender = Recommender(U = factorised_movies,
									V = factorised_users,
									S = factorised_diag)

	def ratings_dict_to_column(self, x):
		"""
			Args:
				x: dict. Keys are movie titles, values are ratings.
			Return: Vector with same length as self.recomender.U.
		"""
		# Convert new_user_ratings, which is by MovieID, to by MovieRow.
		x_df = pd.DataFrame.from_dict(x, orient='index')
				.assign(title=x_df.index, rating=0)
				.join(self.movie_df, by='title')
		ratings_col = np.empty(len(self.movie_df))
		ratings_col[:] = np.nan
		ratings_col[x_df['MovieRow']] = x_df['rating']

	def get_predictions(self, new_user_ratings):
		"""
			Args:
				new_new_ratings: dict. Keys are titles, values are ratings.
		"""
		# Make a prediction.
		predictions = self.recommender.predict_for_new_v(ratings_dict_to_column(new_user_ratings))
		# predictions is a vector corresponding to MovieRow.
		# Convert to title.
		#TODO Make sure this returns a copy.
		
