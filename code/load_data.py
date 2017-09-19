import pandas as pd
from scipy import sparse

def load_movies_data(f):
	# File contents:
	# 1::Toy Story (1995)::Animation|Children's|Comedy
	# 2::Jumanji (1995)::Adventure|Children's|Fantasy
	df = pd.read_csv(f,
					sep="::",
					names=["MovieID", "TitleYear", "Genre"])
	# Split TitleYear into separate columns.
	df['Year'] = df['TitleYear'].str[-5:-1].astype('int')
	df['Title'] = df['TitleYear'].str[0:-7] # Drop last characters ' (1234)'.
	# Drop TitleYear and Genre columns.
	#TODO
	# Clean based on titles.
	# Bear in mind that MovieID in ratings file uses the non-cleaned IDs.
	#TODO
	return df

def load_ratings_data(f):
	# File contents:
	# UserID::MovieID::Rating::Timestamp
	# Load file.
	df = pd.read_csv(f,
					sep="::",
					names=["UserID", "MovieID", "Rating", "Timestamp"])
	# Convert to matrix with axes movies, users.
	m = sparse #TODO Finish - probably coordinate sparse matrix.
	return m
