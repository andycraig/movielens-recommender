from recommender import Recommender
import numpy as np
import pandas as pd

class MovieLensRecommender:

	def __init__(self, config):
		factorised_users = np.loadtxt(config['factorised_users_file'])
		factorised_movies = np.loadtxt(config['factorised_movies_file'])
		factorised_diag = np.loadtxt(config['factorised_diag_file'])
		self.MovieID_df = pd.read_csv(config['MovieID_and_row_file'])
		self.recommender = Recommender(U = factorised_movies,
									V = factorised_users,
									S = factorised_diag)

	def get_predictions(new_user_ratings):
		# Make a prediction.
		predictions = self.recommender.predict_for_new_v(new_user_ratings)
		# predictions are by MovieRow, so need to convert to MovieID.
		self.MovieID_df = self.MovieID_df.assign(prediction = predictions)
		#TODO return 
