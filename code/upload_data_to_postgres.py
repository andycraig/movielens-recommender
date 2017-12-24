import sys, yaml
import pandas as pd
from sqlalchemy import create_engine

try:
    with open(sys.argv[1], 'r') as f:
        config = yaml.load(f)
except IndexError as e:
    sys.exit("First argument must be config file.")
try:
    URI = sys.argv[2]
except IndexError as e:
    sys.exit("Second argument must be database URI.")

movies_df = pd.read_csv(config['movies_for_database'])
movies_df.columns = [c.lower() for c in movies_df.columns] #postgres doesn't like capitals or spaces
#diag_df = pd.read_csv(config['diag_for_database'])
#diag_df.columns = [c.lower() for c in diag_df.columns] #postgres doesn't like capitals or spaces

print("Creating connection to database...")
engine = create_engine(URI)

print("Upload files to database...")
movies_df.to_sql("movies", engine)
#diag_df.to_sql("diag", engine)

print("Done.")
