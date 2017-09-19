import sys, yaml
import pandas as pd
from sklearn.decomposition import NMF
from load_data import load_movies_data, load_ratings_data

def main(config):
	# Load data.
	print("Loading data...")
	movies_data = load_movies_data(config['movies_file'])
	ratings_data = load_ratings_data(config['ratings_file'])
	print("Done.")
	# Perform non-negative matrix factorisation using alternating least squares fitting.
	print("Fitting...")
	model = NMF(n_components=config['NMF']['n_components'])
	W = model.fit_transform(ratings_data)
	H = model.components_
	print("Done.")
	# Save results.
	print("Saving results...")
	print(W.shape)
	print(H.shape)
	#TODO
	#TODO
	print("Done.")

if __name__ == "__main__":
	# Load config file.
	with open(sys.argv[1]) as f:
		config = yaml.load(f)
	# Process.
	main(config)
