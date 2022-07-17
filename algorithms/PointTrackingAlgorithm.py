from dataclasses import dataclass

import torch
import torch.nn.functional as F

from algorithms.Algorithm import Algorithm
from models.TrackingNet import TrackingNet


@dataclass
class Data:
    data: torch.Tensor
    batch: torch.LongTensor


class PointTrackingAlgorithm(Algorithm):
    def __init__(self, point_dim=2, out_channels=2, lr=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TrackingNet(out_channels=out_channels, k=20, point_dim=point_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        super().__init__()

    def prepare_data_for_model(self, data):
        f_size = data.shape[0] * data.shape[1]

        batch = [[i] * data.shape[1] for i in range(len(data))]
        batch = [e for sub_list in batch for e in sub_list]
        batch = torch.LongTensor(batch).to(self.device)

        data = torch.reshape(data, (f_size, data.shape[2]))

        return Data(data=data, batch=batch)

    def train(self, train_loader):
        self.model.train()

        total_loss = 0
        for pcd, y in train_loader:
            pcd = pcd.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            data = self.prepare_data_for_model(pcd)
            out = self.model(data)

            # use Regression loss
            loss = F.mse_loss(out.to(torch.float32), y.to(torch.float32))

            loss.backward()

            total_loss += loss.item() * train_loader.batch_size
            self.optimizer.step()

        return total_loss / len(train_loader.dataset)

    def test(self, test_loader):
        self.model.eval()

        error = 0
        for pcd, y in test_loader:
            pcd = pcd.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                data = self.prepare_data_for_model(pcd)
                pred = self.model(data)

            error += abs(pred - y).sum().item() / len(test_loader.dataset)
        return error
