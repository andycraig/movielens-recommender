from recommender_m1 import MovieLensRecommender
import pandas as pd
import yaml

with open("config-1m.yaml") as f:
    config = yaml.load(f)

m = MovieLensRecommender(config)
movies_df = pd.read_csv(config['movies_csv'])
n_movies = len(movies_df)

# Some random ratings. See if we can get it to recommend action movies.
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
