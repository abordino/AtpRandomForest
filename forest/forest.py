from node import DecisionNode
from tree import DecisionTree
from random import choices


class RandomForest:

    def __init__(self, b=30):
        self._garden = {}
        self._how_many = b

    def get_garden(self):
        return self._garden

    def get_number(self):
        return self._how_many

    def set_garden(self, garden):
        self._garden = garden

    def set_number(self, b):
        self._how_many = b

    def grow(self, dataset, m=None, depth=float('inf'), min_leaves=1, post_pruning=None):
        tmp_tree = []
        for i in range(self.get_number()):
            N = dataset.shape[0]
            index = choices(range(N), k=N)
            bagged_data = dataset.iloc[index, :]
            bagged_data = bagged_data.drop_duplicates()

            tmp = DecisionTree(DecisionNode())
            tmp.grow(bagged_data, m, depth, min_leaves, post_pruning)

            tmp_tree.append(tmp)
            print("Tree " + str(i) + " is been added")

        self.set_garden(tmp_tree)

    def predict(self, observation):
        votes = {}
        for i in range(len(self.get_garden())):
            tree_i = self.get_garden()[i]
            class_i = tree_i.predict(observation)
            if class_i in votes:
                votes[class_i] += 1
            else:
                votes[class_i] = 1
        return max(votes, key=votes.get)

    def accuracy(self, dataset):
        N = dataset.shape[0]
        predictions = dataset.apply(lambda row: self.predict(row), axis=1)
        matches = sum(x == y for x, y in zip(predictions, dataset.iloc[:, -1]))
        percentage = matches / N
        return percentage


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split

    iris = pd.read_csv("./data/Iris.csv", index_col=0)
    train, test = train_test_split(iris, test_size=0.2)

    print("----> FOREST ON WHOLE IRIS \n")
    iris_forest = RandomForest()
    iris_forest.grow(iris)
    print("Full train accuracy: {0:.0%}".format(iris_forest.accuracy(iris)))

    print("----> SOME OF ITS TREES \n")
    print(iris_forest.get_garden()[1])
    print(iris_forest.get_garden()[11])
    print(iris_forest.get_garden()[21])

    print("----> FOREST ON TRAIN + ACCURACY \n")
    iris_forest = RandomForest()
    iris_forest.grow(train)
    print("Full test acc: {0:.0%}".format(iris_forest.accuracy(test)))

    print("----> FOREST ON TRAIN m=2 + ACCURACY \n")
    iris_forest = RandomForest()
    iris_forest.grow(train, m=2)
    print("Reduced test acc: {0:.0%}".format(iris_forest.accuracy(test)))

    print("----> FOREST WITH 5-LEAVES + ACCURACY \n")
    iris_forest = RandomForest()
    iris_forest.grow(train, min_leaves=5)
    print("Reduced test acc: {0:.0%}".format(iris_forest.accuracy(test)))

    print("----> MEP FOREST + ACCURACY \n")
    iris_forest = RandomForest()
    iris_forest.grow(train, post_pruning="mep")
    print("Reduced test acc: {0:.0%}".format(iris_forest.accuracy(test)))

    print("----> REP FOREST + ACCURACY \n")
    iris_forest = RandomForest()
    iris_forest.grow(train, post_pruning="rep")
    print("Reduced test acc: {0:.0%}".format(iris_forest.accuracy(test)))
