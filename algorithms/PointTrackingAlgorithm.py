from dataclasses import dataclass
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from algorithms.Algorithm import Algorithm
from models.TrackingNet import TrackingNet


@dataclass
class Data:
    data: torch.Tensor
    batch: torch.LongTensor


class PointTrackingAlgorithm(Algorithm):
    def __init__(self, point_dim=3, out_channels=2, lr=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TrackingNet(out_channels=out_channels, k=20, point_dim=point_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        super().__init__()

    def train(self, train_loader):
        self.model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(data)

            # use Regression loss
            mse_loss = nn.MSELoss()
            loss = mse_loss(out, data.y)

            loss.backward()

            total_loss += loss.item()
            self.optimizer.step()

        return total_loss / len(train_loader.dataset)

    def test(self, test_loader):
        self.model.eval()

        error = 0
        for data in test_loader:
            data = data.to(self.device)

            with torch.no_grad():
                pred = self.model(data)

            error += torch.sum(torch.linalg.norm(pred - data.y, dim=1)).detach() / len(test_loader.dataset)

        return error
