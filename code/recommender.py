import numpy as np

class Recommender:

	def __init__(self, U, V, S):
		"""
			Args:
				U: Matrix.
				V: Matrix. Updates are made to this.
				S: Vector.
		"""
		self.U = U
		self.V = V
		self.S = S # Vector. Matrix is np.diag(self.S).
		self.new_v = None
		# Precompute U_transpose.
		self.U_transpose = self.U.transpose()

	def predict_for_new_v(self, new_v):
		"""
			Args:
				new_v: Column matrix representing ratings for new column of ratings matrix.
		"""
		# If haven't already done so for this new_v, do rank-one update.
		if self.new_v == None or not (self.new_v == new_v).all():
			self._rank_one_update(a=new_v)
		# Dot product of new_v's projection into latent V space
		# with latent U space.
		#TODO
		return predictions

	def _rank_one_update(self, a):
		"""Using method from Brand 2006.
		Update operation, from Table 1.

			Args:
				a: Column matrix to be appended.
		"""
		m = self.U_transpose * a
		p_norm = np.linalg.norm(a - self.U * m)
		K = [[np.diag(self.S), m], [0, p_norm]]
		# Rediagonalise K to get U'^T K V' = S'.
		self.U_prime, self.S_prime, self.V_prime  = np.linalg.svd(K)
		# Now [X a] = U' S' V'^T = ([U P] U') S' ([V Q] V')^T
