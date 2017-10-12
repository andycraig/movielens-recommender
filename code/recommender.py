import numpy as np

class Recommender:

    def __init__(self, U, V, S):
        """
            Args:
                U: Matrix.
                V: Matrix. Updates are made to this.
                S: Vector.
        """
        self.U = np.matrix(U)
        self.V = np.matrix(V)
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
        print("In _rank_one_update")
        mask_nas = np.isnan(c)
        # Following notation in paper:
        # c_closed (filled circle): vector of known values in c.
        # c_open (open circle): vector of unknown values in c.
        # U_closed, U_open: corresponding rows of U.
        c_closed = np.transpose(np.matrix(c[~mask_nas])) # Shape: n_non_missing x 1.
        c_open = c[mask_nas] # Shape: n_missing x 1.
        U_closed = self.U[~mask_nas, :] # Shape: n_non_missing x r.
        U_open = self.U[mask_nas, :] # Shape: n_missing x r.
        # Impute missing values.
        # Compute broken-arrow matrix (called K in Brand 2006, although not in this paper).
        upper_left_block = self.diag_S # Shape: r x r
        upper_right_block = self.diag_S * np.linalg.pinv(U_closed * self.diag_S) * c_closed
        # upper_right_block shape: r x 1.
        lower_right_block = np.matrix(np.linalg.norm(c_closed - U_closed * upper_right_block))
        lower_left_block = np.matrix(np.zeros([1, self.diag_S.shape[1]]))
        upper_block = np.hstack([upper_left_block, upper_right_block])
        lower_block = np.hstack([lower_left_block, lower_right_block])
        K = np.vstack([upper_block, lower_block])
        # Rediagonalise K to get U'^T K V' = S'.
        U_prime, diag_S_prime, V_prime  = np.linalg.svd(K)
        # Note that diag(S'') and diag(s') are the same.
        # Now [X c] = U' S' V'^T = ([U P] U') S' ([V Q] V')^T
        # Need to compute P, Q.
		# TODO Simplify this, since one of a and b has only one non-zero element.
        m = np.transpose(self.U) * a
        p = a - self.U * m 
        P = p / np.norm(p)
        n = np.transpose(self.V) * b
        q = b - self.V * n
        Q = q / np.norm(q)
        U_double_prime = np.hstack(self.U, P) * U_prime
        V_double_prime = np.hstack(self.V, Q) * V_prime
		# TODO Do something with this.