import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader

from algorithms.ClassificationAlgorithm import ClassificationAlgorithm

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')

number_of_points = 50

# Transform Mesh to Point Cloud
pre_transform, transform = T.NormalizeScale(), T.SamplePoints(50)

train_dataset = ModelNet(path, '10', True, transform, pre_transform)
test_dataset = ModelNet(path, '10', False, transform, pre_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6)

algorithm = ClassificationAlgorithm(num_classes=train_dataset.num_classes)

for epoch in range(1, 201):
    loss = algorithm.train(train_loader)
    test_acc = algorithm.test(test_loader)
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
    algorithm.scheduler.step()
