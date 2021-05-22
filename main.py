import pandas as pd
from sklearn.model_selection import train_test_split
from tree import DecisionTree
from forest import RandomForest

tennis = pd.read_csv("data/tennis_data.csv")
train, test = train_test_split(tennis, shuffle=False, test_size=0.33)

print("----> 21-10-DECISION TREE \n")
tennis_tree = DecisionTree()
tennis_tree.grow(train, min_leaves=21, depth=10)
print("Full test acc: {0:.0%}".format(tennis_tree.accuracy(test)) + "\n")
print(tennis_tree)

