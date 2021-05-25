import pandas as pd
from sklearn.model_selection import train_test_split
from forest.forest import RandomForest
import pickle

tennis = pd.read_csv("./data/tennis_data.csv")
train, third_label = train_test_split(tennis, test_size=0.9)
new_train, test = train_test_split(third_label, test_size=0.9)

n_tree = float(input('Choose the number of trees in the forest (default is 30): '))
subsample = float(input('Choose m - number of variables considered in every split (default is all): '))
leaves = float(input('Choose the minimum number of element in each leaf (default is 1): '))
dept = float(input('Choose max depth of the tree (default is Inf): '))
prune = input('Choose how to prune the tree: mep or rep (default is None): ')

if n_tree or subsample or leaves or dept or prune:
    print("----> M" + str(subsample) + "L" + str(leaves) + "D" + str(dept) + "RANDOM FOREST \n")
    tennis_forest = RandomForest(b=20)
    tennis_forest.grow(train, m=subsample, min_leaves=leaves, depth=dept, post_pruning=prune)
    print("Full test acc: {0:.0%}".format(tennis_forest.accuracy(test)) + "\n")

    file_handler = open("M" + str(subsample) + "L" + str(leaves) + "D" + str(dept) + ".obj", "wb")
    pickle.dump(tennis_forest, file_handler)
    file_handler.close()