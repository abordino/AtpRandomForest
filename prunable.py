from abc import ABCMeta, abstractmethod


class Prunable(metaclass=ABCMeta):
    @abstractmethod
    def reduced_pruning(self, train_dataset, prune_dataset):
        pass

    @abstractmethod
    def minimum_pruning(self,train_dataset, prune_dataset):
        pass
