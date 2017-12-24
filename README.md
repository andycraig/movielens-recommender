# movielens-recommender

Processes the MovieLens dataset for use in a recommender system, and provides functionality to make recommendations.

## Setup

Download ml-1m.zip from [https://grouplens.org/datasets/movielens/1m/](https://grouplens.org/datasets/movielens/1m/) and unpack into ~data/ml-1m/~.

Install Python requirements with ~pip install -r requirements.txt~.

R requirements can be installed using ~packrat~.

## Preprocessing

Prepare files:

```
sed 's/\:\:/,/g' data/ml-1m/ratings.dat > data/ml-1m/ratings.csv
echo "MovieID,Title,Year,Genre" > data/ml-1m/movies.csv
sed -n 's;\([0-9]*\)::\(.*\) (\([0-9]*\))::\(.*\);\1,"\2",\3,\4;p' data/ml-1m/movies.dat >> data/ml-1m/movies.csv
```

Perform ratings matrix factorisation with:

```
Rscript code/prepare.R code/config-1m.yaml
```

## Predicting

Short example:

```
python code/demo.py
```
