from prunable import Prunable
from node import DecisionNode
from queue import Queue
from sklearn.model_selection import train_test_split


class DecisionTree(Prunable):

    def __init__(self, root=DecisionNode()):
        self.root = root

    def get_root(self):
        return self.root
    
    def set_root(self, root):
        self.root = root
    
    def to_string(self, node, depth=0):
        ret = ""
        ret += (("|" + "   ")*depth) + "|---" + str(node) + "\n"
        if node.get_true() is not None:
            ret += (("|" + "   ")*depth) + self.to_string(node.get_true(), depth + 1)
        if node.get_color() is None:
            ret += (("|" + "   ")*depth) + "|---" + node.print_false() + "\n"
        if node.get_false() is not None:
            ret += (("|" + "   ")*depth) + self.to_string(node.get_false(), depth + 1)

        return ret

    def __str__(self):
        if self.get_root():
            return self.to_string(self.get_root())
        return "The tree is empty"

    def __eq__(self, tree2):
        if not tree2:
            return False

        current_node = self.get_root()
        other_node = tree2.get_root()

        _true = False
        if current_node.get_true():
            if other_node.get_true():
                true_tree = DecisionTree(current_node.get_true())
                other_true = DecisionTree(other_node.get_true())
                _true = (true_tree == other_true)
        else:
            _true = not other_node.get_true()

        _false = False
        if current_node.get_false():
            if other_node.get_false():
                false_tree = DecisionTree(current_node.get_false())
                other_false = DecisionTree(other_node.get_false())
                _false = (false_tree == other_false)
        else:
            _false = not other_node.get_false()

        return current_node == other_node and _true and _false

    def predict(self, observation):
        visiting_node = self.get_root()
        while visiting_node.get_color() is None:
            if visiting_node.ask(observation) is True:
                visiting_node = visiting_node.get_true()
            else: 
                visiting_node = visiting_node.get_false()

        return visiting_node.get_color()

    def accuracy(self, dataset):
        N = dataset.shape[0]
        predictions = dataset.apply(lambda row: self.predict(row), axis=1)
        matches = sum(x == y for x, y in zip(predictions, dataset.iloc[:, -1]))
        percentage = matches/N
        return percentage

    def breadth(self):
        nodes = []
        if self.root:
            queue = Queue()
            queue.put(self.root)

            while not queue.empty():
                current = queue.get()
                nodes.append(current)
                if current.get_true():
                    queue.put(current.get_true())
                if current.get_false():
                    queue.put(current.get_false())
        return nodes

    def deepest_split(self):
        traversal = self.breadth()
        frontier = []
        for node in traversal:
            if node.get_true() is not None and node.get_false() is not None:
                if (node.get_true().get_color() is not None) and (node.get_false().get_color() is not None):
                    frontier.append(node)

        return frontier

    def minimum_pruning(self, train_dataset, prune_dataset):
        for node in self.deepest_split():
            current_acc = self.accuracy(prune_dataset)
            node.average_color(train_dataset.loc[node.get_index(), :])
            new_acc = self.accuracy(prune_dataset)
            if new_acc >= current_acc:
                node.set_true(None)
                node.set_false(None)
            else:
                node.set_color(None)

    def reduced_pruning(self, train_dataset, prune_dataset):
        improvement = 0
        while improvement >= 0:
            base_acc = self.accuracy(prune_dataset)
            best_acc, best_node = 0, None
            for node in self.breadth():
                if (node.get_color() is None) and (node is not self.get_root()):
                    node.average_color(train_dataset.loc[node.get_index(), :])
                    new_acc = self.accuracy(prune_dataset)
                    if new_acc >= best_acc:
                        best_acc = new_acc
                        best_node = node
                    node.set_color(None)
            improvement = best_acc - base_acc
            if improvement >= 0:
                best_node.average_color(train_dataset.loc[best_node.get_index(), :])
                best_node.set_true(None)
                best_node.set_false(None)

    def basic_grow(self, dataset, m=None, depth=float('inf'), min_leaves=1):
        current_node = self.get_root()
        N = dataset.shape[0]

        if (current_node.gini(dataset) == 0) or (depth == 0) or (N <= min_leaves):
            current_node.average_color(dataset)
            return current_node.get_color()

        else:
            current_node.best_split(dataset, subsample=m)
            true_tree, false_tree = DecisionTree(current_node.get_true()), DecisionTree(current_node.get_false())
            true_data = dataset.loc[current_node.get_true().get_index(), :]
            false_data = dataset.loc[current_node.get_false().get_index(), :]

            return true_tree.grow(true_data, m, depth - 1), false_tree.grow(false_data, m, depth - 1)

    def grow(self, dataset, m=None, depth=float('inf'), min_leaves=1, post_pruning=None):
        if post_pruning is not None:
            train_data, prune_data = train_test_split(dataset, test_size=0.3)
        else:
            train_data = dataset
        self.basic_grow(train_data, m, depth, min_leaves)
        if post_pruning == "mep":
            self.minimum_pruning(train_data, prune_data)
        elif post_pruning == "rep":
            self.reduced_pruning(train_data, prune_data)


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import export_text

    iris = pd.read_csv("data/Iris.csv", index_col=0)
    train, test = train_test_split(iris, test_size=0.2)

    print("----> FULL GROWTH ON WHOLE IRIS \n")
    iris_tree = DecisionTree()
    iris_tree.grow(iris)
    iris_tree.accuracy(iris)
    print(iris_tree)

    print("----> REDUCED GROWTH ON TRAIN depth = 2 \n")
    iris_tree = DecisionTree()
    iris_tree.grow(train, depth=2)
    print("Accuracy: {0:.0%}".format(iris_tree.accuracy(test)) + "\n")
    print(iris_tree)

    print("----> REDUCED GROWTH ON TRAIN depth = 3, m = 2 \n")
    iris_tree = DecisionTree()
    iris_tree.grow(train, depth=3, m=2)
    print("Reduced growth acc: {0:.0%}".format(iris_tree.accuracy(test)) + "\n")
    print(iris_tree)

    print("----> REDUCED SPLIT ON TRAIN m = 2 \n")
    iris_tree = DecisionTree()
    iris_tree.grow(train, m=2)
    print("Reduced split acc: {0:.0%}".format(iris_tree.accuracy(test)) + "\n")
    print(iris_tree)

    print("----> MINIMUM 5 LEAVES \n")
    iris_tree = DecisionTree()
    iris_tree.grow(train, min_leaves=5)
    print("5-leaves acc: {0:.0%}".format(iris_tree.accuracy(test)) + "\n")
    print(iris_tree)

    print("----> TEST ACCURACY + TRAIN STRUCTURE \n")
    iris_tree = DecisionTree()
    iris_tree.grow(train)
    print("Full test acc: {0:.0%}".format(iris_tree.accuracy(test)) + "\n")
    print(iris_tree)

    print("----> SCIKIT ON TRAIN \n")
    X = train.iloc[:, 0:4]
    y = train.iloc[:, -1]
    decision_tree = DecisionTreeClassifier(splitter="best")
    decision_tree = decision_tree.fit(X, y)
    r = export_text(decision_tree)
    print(r)

    print("----> BREADTH REPRESENTATION \n")
    traversal = iris_tree.breadth()
    for node in traversal:
        print(node)

    print("\n----> DEEPEST SPLIT \n")
    traversal = iris_tree.deepest_split()
    for node in traversal:
        print(node)

    print("----> NO POST-PRUNING \n")
    iris_tree = DecisionTree()
    iris_tree.grow(train)
    print("No pruning acc: {0:.0%}".format(iris_tree.accuracy(test)) + "\n")
    print(iris_tree)

    print("----> MEP PRUNING \n")
    iris_tree = DecisionTree()
    iris_tree.grow(train, post_pruning="mep")
    print("Mep acc: {0:.0%}".format(iris_tree.accuracy(test)) + "\n")
    print(iris_tree)

    print("----> REP PRUNING \n")
    iris_tree = DecisionTree()
    iris_tree.grow(train, post_pruning="rep")
    print("Rep acc: {0:.0%}".format(iris_tree.accuracy(test)) + "\n")
    print(iris_tree)
