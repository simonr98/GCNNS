import torch
from abc import ABC, abstractmethod


class Algorithm(ABC):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def train(self, train_loader):
        """
            ...
        """
        raise NotImplementedError

    @abstractmethod
    def test(self, test_loader):
        """
                    ...
        """
        raise NotImplementedError
