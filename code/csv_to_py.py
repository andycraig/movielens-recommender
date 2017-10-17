import numpy as np
import pandas as pd

#TODO Save the contents of these files into .py files.
np.loadtxt(config['factorised_users_file'])
np.loadtxt(config['factorised_movies_file'])
np.loadtxt(config['factorised_diag_file'])
pd.read_csv(config['MovieID_and_row_file'])