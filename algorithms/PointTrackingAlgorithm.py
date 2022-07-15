import torch
import torch.nn.functional as F

from algorithms.Algorithm import Algorithm
from models.TrackingNet import TrackingNet


class PointTrackingAlgorithm(Algorithm):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TrackingNet(out_channels=2, k=20).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        super().__init__()

    def train(self, train_loader):
        self.model.train()

        total_loss = 0
        for pcd, y in train_loader:
            pcd = pcd.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(pcd)

            # use Regression loss
            loss = F.mse_loss(out.to(torch.float32), y.to(torch.float32))

            loss.backward()

            total_loss += loss.item() #* data.num_graphs
            self.optimizer.step()
        return total_loss / len(train_loader.dataset)

    def test(self, test_loader):
        self.model.eval()

        error = 0
        for pcd, y in test_loader:
            pcd = pcd.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                pred = self.model(pcd)

            error += abs(pred - y).sum().item() / len(test_loader.dataset)
        return error