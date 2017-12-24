import sys
import pandas as pd

try:
    URI = sys.argv[1]
except IndexError as e:
    sys.exit("First argument must be database URI.")

movies_df = pd.read_csv('../data/ml-1m/movies_for_database.csv')
movies_df.columns = [c.lower() for c in movies_df.columns] #postgres doesn't like capitals or spaces
diag_df = pd.read_csv('../data/ml-1m/diag_for_database.csv')
diag_df.columns = [c.lower() for c in diag_df.columns] #postgres doesn't like capitals or spaces

from sqlalchemy import create_engine
engine = create_engine(URI)

movies_df.to_sql("movies", engine)
diag_df.to_sql("diag", engine)
