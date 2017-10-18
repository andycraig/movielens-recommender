from recommender_m1 import MovieLensRecommender
import pandas as pd
import numpy as np
import yaml
from data import factorised_movies, factorised_users, factorised_diag, MovieID_and_row

with open("config-1m.yaml") as f:
    config = yaml.load(f)

#TODO Preprocess factorised_users.
m = MovieLensRecommender(factorised_users=factorised_users),
                         factorised_movies=np.loadtxt(config['factorised_movies_file']),
                         factorised_diag=np.loadtxt(config['factorised_diag_file']),
                         movie_df=pd.read_csv(config['MovieID_and_row_file']))
movies_df = pd.read_csv(config['movies_csv'])
n_movies = len(movies_df)

# Some ratings for an action movie fan.
new_user_ratings = {'Die Hard':5, 'Terminator':5, 'Speed':5}
print("Ratings for new user:")
print(new_user_ratings)
predictions_df = m.get_predictions(new_user_ratings)
print("Top predictions for this user:")
print(predictions_df.sort_values('prediction', ascending=False).head())
print("Worst movies for this user:")
print(predictions_df.sort_values('prediction', ascending=True).head())
print("Top predictions at or before 1960 for this user:")
print(predictions_df.loc[predictions_df['Year'] <= 1960, :].sort_values('prediction', ascending=False).head())
print("Predictions for movies that were actually rated for this user:")
print(predictions_df.loc[predictions_df['Title'].isin(new_user_ratings), :])
