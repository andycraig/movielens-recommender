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
		self.diag_S = np.diag(S)
		self.new_v = None
		# Precompute U_transpose.
		self.U_transpose = self.U.transpose()

	def predict_for_new_v(self, new_v):
		"""
			Args:
				new_v: Vector representing ratings for new column of ratings matrix.
		"""
		if len(new_v) != len(self.U):
			raise ValueError("new_v (length " + str(len(new_v)) + ") have same length as rows of U (which is " + str(len(self.U)) + ").")
		# If haven't already done so for this new_v, do rank-one update.
		if self.new_v == None or not (self.new_v == new_v).all():
			self._rank_one_update(new_v)
		# Dot product of new_v's projection into latent V space (which is last row of V_prime)
		# with latent U space.
		predictions = self.U_prime * self.V_prime[-1, :]
		return predictions

	def _rank_one_update(self, c):
		"""Using method from Brand 'Fast online SVD revisions for lightweight recommender systems',
		Section B.1.

			Args:
				a: Column matrix to be appended. Can have missing values.
		"""
		mask_nas = np.isnan(c)
		missing_indices = np.where(mask_nas)
		not_missing_indices = np.where(~mask_nas)
		# Following notation in paper:
		# c_closed (filled circle): vector of known values in c.
		# c_open (open circle): vector of unknown values in c.
		# U_closed, U_open: corresponding rows of U.
		c_closed = c[not_missing_indices]
		c_open = c[missing_indices]
		U_closed = self.U[not_missing_indices, :]
		U_open = self.U[missing_indices, :]
		# Impute missing values.
		#c_hat = (U_open * self.diag_S) * np.linalg.pinv(U_closed * self.diag_S) * c_closed
		# Compute broken-arrow matrix (called K in Brand 2006, although not in this paper).
		upper_right_block = self.diag_S * np.linalg.pinv(U_closed * self.diag_S) * c_closed
		K = [[self.diag_S, upper_right_block],
			 [0, np.linalg.norm(c_closed - U_closed * upper_right_block)]]
		# Rediagonalise K to get U'^T K V' = S'.
		self.U_prime, _, self.V_prime  = np.linalg.svd(K)
		# Now [X c] = U' S' V'^T = ([U P] U') S' ([V Q] V')^T
