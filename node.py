from random import sample


class DecisionNode:
    """
        A Decision Node asks a question based on a column and a threshold. Color attribute is used to classify whereas
        the index attribute contains the index of the observation of the training dataset which are able to reach the node.
        It has a true and a left child.
    """

    def __init__(self, threshold=None, column=None, true_branch=None, false_branch=None, color=None, index=None):
        self._threshold = threshold
        self._column = column
        self._true_branch = true_branch
        self._false_branch = false_branch
        self._color = color
        self._index = index

    def get_threshold(self):
        return self._threshold

    def get_column(self):
        return self._column

    def get_true(self):
        return self._true_branch

    def get_false(self):
        return self._false_branch

    def get_color(self):
        return self._color

    def get_index(self):
        return self._index

    def set_threshold(self, threshold):
        self._threshold = threshold

    def set_column(self, column):
        self._column = column

    def set_true(self, true_branch):
        self._true_branch = true_branch

    def set_false(self, false_branch):
        self._false_branch = false_branch

    def set_color(self, color):
        self._color = color

    def set_index(self, index):
        self._index = index

    def is_leaf(self):
        if self.get_color() is not None:
            return True
        else:
            return False

    def __eq__(self, node2):
        if not node2:
            return False
        else:
            return (self.get_color() == node2.get_color()) and (self.get_threshold() == node2.get_threshold()) and\
                   (self.get_column() == node2.get_column())

    def print_generic(self, condition_equal, condition_numeric):
        if self.get_color() is not None:
            return str(self.get_color())
        elif self.get_column() is not None and self.get_threshold() is not None:
            condition = condition_equal
            if type(self.get_threshold()) == int or type(self.get_threshold()) == float:
                condition = condition_numeric
            return "column {} {} {}?".format(self.get_column(), condition, str(self.get_threshold()))
        else:
            return "The node is empty"

    def __str__(self):
        return self.print_generic("==", ">=")

    def print_false(self):
        return self.print_generic("!=", "<")

    def ask(self, observation):
        if (self.get_column() is not None) and (self.get_threshold() is not None):
            if type(self.get_threshold()) == int or type(self.get_threshold()) == float:
                return observation[self.get_column()] >= self.get_threshold()
            else:
                return observation[self.get_column()] == self.get_threshold()

    def average_color(self, dataset):
        self.set_color(dataset.iloc[:, -1].mode()[0])

    @staticmethod
    def gini(dataset):
        """
            Calculate the Gini Impurity for a list of rows.
            https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
            The column of the label must be the last one.
        """

        impurity = 1
        prob = {}
        N = dataset.shape[0]
        for i in range(dataset.shape[0]):
            if dataset.iloc[i, -1] in prob:
                prob[dataset.iloc[i, -1]] += 1
            else:
                prob[dataset.iloc[i, -1]] = 1

        for lab in prob.keys():
            p_lab = prob[lab] / N
            impurity -= p_lab ** 2

        return impurity

    def info_gain(self, dataset, true_data, false_data):
        """
            Calculate the gini gain in order to find the best split
            in terms of information gain.
        """

        if self.get_true() and self.get_false():
            N = dataset.shape[0]
            N_T, N_F = true_data.shape[0], false_data.shape[0]
            p_true = N_T / N
            p_false = N_F / N
            gain = self.gini(dataset) - p_true * self.get_true().gini(true_data) - p_false * self.get_false().gini(false_data)

            return gain

    def best_split(self, dataset, subsample=None):
        """
            Find the best question to ask by iterating over every feature / value
            and calculating the information gain.
            The type columns must be the last one.
        """

        info_gain = 0
        best_column, best_threshold, best_index = None, None, None
        COL_NUMBER = dataset.shape[1] - 1
        possible_column = range(COL_NUMBER)

        if (subsample is not None) and (subsample <= COL_NUMBER):
            possible_column = sample(possible_column, subsample)

        for p in possible_column:
            for value in set(dataset.iloc[:, p]):
                self.set_column(p)
                self.set_threshold(value)
                true_index = dataset.apply(lambda row: self.ask(row), axis=1)
                true_data = dataset[true_index]
                false_data = dataset[[not el for el in true_index]]

                self.set_true(DecisionNode())
                self.set_false(DecisionNode())
                candidate_info = self.info_gain(dataset, true_data, false_data)
                if candidate_info > info_gain:
                    info_gain = candidate_info
                    (best_column, best_threshold) = (p, value)
                    best_index = true_index
        if info_gain:
            self.set_column(best_column), self.set_threshold(best_threshold)
            self.get_true().set_index(best_index[best_index].index)
            self.get_false().set_index(best_index[[not el for el in best_index]].index)

        else:
            return self.best_split(dataset, subsample)