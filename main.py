import pandas as pd
from sklearn.model_selection import train_test_split
from forest import RandomForest

import pickle

tennis = pd.read_csv("data/tennis_data.csv", header=0)
train, third_label = train_test_split(tennis, test_size=0.99)
new_train, test = train_test_split(third_label, test_size=0.99)

n_tree = int(input('Choose the number of trees in the forest (default is 30): '))
subsample = int(input('Choose m - number of variables considered in every split (default is all): '))
leaves = int(input('Choose the minimum number of element in each leaf (default is 1): '))
dept = int(input('Choose max depth of the tree (default is Inf): '))
prune = input('Choose how to prune the tree: mep or rep (default is None): ')

if n_tree or subsample or leaves or dept or prune:
    print("----> M" + str(subsample) + "L" + str(leaves) + "D" + str(dept) + "-" + str(n_tree) + "-RANDOM FOREST \n")
    tennis_forest = RandomForest(b=n_tree)
    tennis_forest.grow(train, m=subsample, min_leaves=leaves, depth=dept, post_pruning=prune)
    print("Full test acc: {0:.0%}".format(tennis_forest.accuracy(test)) + "\n")

    with open("models/M" + str(subsample) + "L" + str(leaves) + "D" + str(dept) + ".obj", "wb") as file_handler:
        pickle.dump(tennis_forest, file_handler)
