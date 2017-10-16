# movielens-recommender

Processes the MovieLens dataset for use in a recommender system, and provides functionality to make recommendations.

### Preprocessing

Prepare files and perform ratings matrix factorisation with:

```
sed 's/\:\:/,/g' data/ml-1m/ratings.dat > data/ml-1m/ratings.csv
echo "MovieID,Title,Year,Genre" > data/ml-1m/movies.csv
sed -n 's;\([0-9]*\)::\(.*\) (\([0-9]*\))::\(.*\);\1,"\2",\3,\4;p' data/ml-1m/movies.dat >> data/ml-1m/movies.csv
Rscript code/prepare.R code/config-1m.yaml
```

### Predicting

Short example is in demo.py. Run with:

```
python code/demo.py
```
