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
        # Precompute transposes.
        self.U_transpose = self.U.transpose()
        self.V_transpose = self.V.transpose()

    def predict_for_new_v(self, new_v):
        """Using method from Brand 'Fast online SVD revisions for lightweight recommender systems',
        Section B.1.

            Args:
                new_v: Vector representing ratings for new column of ratings matrix. Can have missing values.
        """
        if len(new_v) != len(self.U):
            raise ValueError("new_v (length " + str(len(new_v)) + ") have same length as rows of U (which is " + str(len(self.U)) + ").")
        c = np.matrix(new_v).transpose() # Column matrix.
        mask_nas = np.isnan(new_v)
        # Following notation in paper:
        # c_closed (filled circle): vector of known values in c.
        # c_open (open circle): vector of unknown values in c.
        # U_closed, U_open: corresponding rows of U.
        c_closed = c[~mask_nas, :] # Shape: n_non_missing x 1.
        c_open = c[mask_nas, :] # Shape: n_missing x 1.
        U_closed = self.U[~mask_nas, :] # Shape: n_non_missing x r.
        U_open = self.U[mask_nas, :] # Shape: n_missing x r.
        # Compute broken-arrow matrix (called K in Brand 2006, although not in this paper).
        upper_right_block = self.diag_S * np.linalg.pinv(U_closed * self.diag_S) * c_closed
        # upper_right_block shape: r x 1.
        upper_left_block = self.diag_S # Shape: r x r
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
        # To compute P, we need to impute missing values of c.
        c_hat_open = U_open * upper_right_block
        c_imputed = c.copy()
        c_imputed[mask_nas, :] = c_hat_open
        m = self.U_transpose * c_imputed # Using c_imputed as a in Appendix A.
        p = c_imputed - self.U * m # c is a in Appendix A.
        P = p / np.linalg.norm(p)
		# TODO Simplify this, since b has only one non-zero element.
        b = np.vstack([np.matrix(np.zeros([self.V.shape[0] - 1, 1])), np.ones([1, 1])])
        # q and Q are both equal to b.
        U_double_prime = np.hstack([self.U, P]) * U_prime
        V_double_prime = np.hstack([self.V, b]) * V_prime # b is equal to Q.
        # Dot product of new_v's projection into latent V space (which is last row of V_double_prime)
        # with latent U space.
        predictions = U_double_prime * np.transpose(V_double_prime[-1, :])
        return predictions
