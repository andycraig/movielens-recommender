import sys, yaml
import pandas as pd

def main(config):
	pass

if __name__ == "__main__":
	try:
		with open(sys.argv[1]) as f:
			config = yaml.load(f)
	#TODO Error handling.
