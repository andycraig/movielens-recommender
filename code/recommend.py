import numpy as np

class recommender:
	slim_matrix_users = None
	slim_matrix_movies = None
	diag = None

	def __init__(self, factorised_users_file, factorised_movies_file, factorised_diag_file):
		self.slim_matrix_users = np.readtxt(factorised_users_file)
		self.slim_matrix_movies = np.readtxt(factorised_movies_file)
		self.diag = np.readtxt(factorised_diag_file)

	def predict_for_new_user(self, new_users_ratings):
		# Do rank-one update.
		#TODO

	def _rank_one_update(self, new_users_ratings):
		"""Using method from Brand.
		"""
		#TODO
