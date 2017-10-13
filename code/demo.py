from recommender_m1 import MovieLensRecommender
import pandas as pd
import numpy as np
import yaml

with open("config-1m.yaml") as f:
	config = yaml.load(f)

m = MovieLensRecommender(config)
movies_df = pd.read_csv(config['movies_csv'])
n_movies = len(movies_df)

# Some random ratings.
new_user_ratings = {'Jumanji':5, 'Waiting to Exhale':1, 'Sudden Death':4}
print(new_user_ratings)
predictions = m.get_predictions(new_user_ratings)
print(predictions)
print(sum(~np.isnan(predictions)))
