from algorithms.ClassificationAlgorithm import ClassificationAlgorithm
from definitions import *
from util.get_data import get_model_net_data, get_voxel_data_loaders, get_torus__data_loaders
from algorithms.PointTrackingAlgorithm import PointTrackingAlgorithm
from util.preprocessing import get_torus_data


class Experiment:
    def __init__(self, config):
        self.track_point_index = config.get('track_point_index', 67)
        self.num_epochs = config.get('num_epochs', 200)
        self.algorithm = config.get('algorithm', 'point_net_classification')

    def run_experiment(self):
        if self.algorithm == 'point_net_classification':
            train_loader, test_loader = get_model_net_data(num_points_pc=NUM_POINTS_POINT_CLOUD,
                                                           batch_size=BATCH_SIZE_VOXEL)
            algorithm = ClassificationAlgorithm(num_classes=train_loader.dataset.num_classes)

            for epoch in range(1, self.num_epochs):
                loss = algorithm.train(train_loader)
                test_acc = algorithm.test(test_loader)
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
                algorithm.scheduler.step()

        if self.algorithm == 'voxel_point_prediction':
            train_loader, test_loader = get_voxel_data_loaders(num_points_pc=NUM_POINTS_POINT_CLOUD,
                                                               track_point_index=INDEX_TRACK_POINT_VOXEL)
            algorithm = PointTrackingAlgorithm(point_dim=2, out_channels=2)

            for epoch in range(1, self.num_epochs):
                loss = algorithm.train(train_loader)
                error = algorithm.test(test_loader)
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Error per sample: {error:.4f}')
                algorithm.scheduler.step()

        if self.algorithm == 'torus_com_prediction':
            train_loader, test_loader = get_torus__data_loaders()
            algorithm = PointTrackingAlgorithm(point_dim=3, out_channels=3)

            for epoch in range(1, self.num_epochs):
                loss = algorithm.train(train_loader)
                error = algorithm.test(test_loader)
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Error per sample: {error:.4f}')
                algorithm.scheduler.step()
