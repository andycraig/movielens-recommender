import numpy as np

class recommender:

	def __init__(self, U, V, S):
		self.U = U
		self.V = V
		self.S = S

	def predict_for_new_user(self, new_users_ratings):
		# Do rank-one update.
		#TODO

	def _rank_one_update(self, new_users_ratings):
		"""Using method from Brand.
		"""
		m = U.transpose() * a
		K = [[]]
