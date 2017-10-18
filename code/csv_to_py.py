# Copy the contents of text and CSV files into .py files.

import numpy as np
import pandas as pd
import yaml, sys

def txt_to_py(source, dest):
	print("Loading from " + source)
	with open(source, 'r') as f:
		lines = f.readlines()
	with open(dest, 'w') as f:
		f.write('x = "')
		for line in lines:
			f.write(line)
		f.write('"')
	print("Wrote to " + dest)

with open(sys.argv[1]) as f:
	config = yaml.load(f)
txt_to_py(config['factorised_users_file'], config['factorised_users_py'])
txt_to_py(config['factorised_movies_file'], config['factorised_movies_py'])
txt_to_py(config['factorised_diag_file'], config['factorised_diag_py'])
txt_to_py(config['MovieID_and_row_file'], config['MovieID_and_row_py'])
