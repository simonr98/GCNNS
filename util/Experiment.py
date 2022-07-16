from algorithms.ClassificationAlgorithm import ClassificationAlgorithm
from util.get_data import get_model_net_data
from algorithms.PointTrackingAlgorithm import PointTrackingAlgorithm
from util.get_data import get_voxel_data


class Experiment:
    def __init__(self, config):
        self.track_point_index = config.get('track_point_index', 67)
        self.num_epochs = config.get('num_epochs', 200)
        self.algorithm = config.get('algorithm', 'point_net_classification')

    def run_experiment(self):
        if self.algorithm == 'point_net_classification':
            train_loader, test_loader = get_model_net_data(num_points_pc=60, batch_size=32)

            algorithm = ClassificationAlgorithm(num_classes=train_loader.dataset.num_classes)

            for epoch in range(1, self.num_epochs):
                loss = algorithm.train(train_loader)
                test_acc = algorithm.test(test_loader)
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
                algorithm.scheduler.step()

        else:
            train_loader, test_loader = get_voxel_data(num_points_pc=60, track_point_index=67)

            algorithm = PointTrackingAlgorithm()

            for epoch in range(1, self.num_epochs):
                loss = algorithm.train(train_loader)
                error = algorithm.test(test_loader)
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Error per sample: {error:.4f}')
                algorithm.scheduler.step()
