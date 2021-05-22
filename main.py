import pandas as pd
from sklearn.model_selection import train_test_split
from forest import RandomForest

tennis = pd.read_csv("tennis_data.csv")
train, test = train_test_split(tennis, test_size=0.90)


print("----> 5-10-RANDOM FOREST \n")
tennis_forest = RandomForest()
tennis_forest.grow(train, min_leaves=5, depth=10, post_pruning="rep")
print("Full test acc: {0:.0%}".format(tennis_forest.accuracy(test)) + "\n")


