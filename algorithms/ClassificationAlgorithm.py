import torch
import torch.nn.functional as F

from algorithms.Algorithm import Algorithm
from models.ClassificationNet import ClassificationNet


class ClassificationAlgorithm(Algorithm):
    def __init__(self, num_classes: int):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ClassificationNet(num_classes, k=20).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        super().__init__()

    def train(self, train_loader):
        self.model.train()

        total_loss = 0
        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            self.optimizer.step()
        return total_loss / len(train_loader.dataset)

    def test(self, test_loader):
        self.model.eval()

        correct = 0
        for data in test_loader:
            data = data.to(self.device)
            with torch.no_grad():
                pred = self.model(data).max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        return correct / len(test_loader.dataset)