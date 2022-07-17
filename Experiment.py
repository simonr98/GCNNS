import wandb
import torch as T
from algorithms.ClassificationAlgorithm import ClassificationAlgorithm
from definitions import *
from util.get_data import get_model_net_data, get_voxel_data_loaders, get_torus_data_loaders
from algorithms.PointTrackingAlgorithm import PointTrackingAlgorithm


class Experiment:
    def __init__(self, config):
        self.track_point_index = config.get('track_point_index', 67)
        self.num_epochs = config.get('num_epochs', 200)
        self.algorithm = config.get('algorithm', 'point_net_classification')
        self.save = config.get('save', False)
        self.lr = config.get('lr', 0.001)
        self.project_name = 'GCNNS'
        self.model_save_path = config.get('model_save_path', f'models/{self.algorithm}')
        self.run_name = config.get('run_name', 'test')
        self.wandb = config.get('wandb', False)
        self.wandb_config = {}
        self.wandb_logging_parameters = {}

    def setup_wandb(self):
        self.wandb_config.update({
            'project_name': self.project_name,
            'run_name': self.run_name,
        })

    def run_experiment(self):
        if self.wandb:
            self.setup_wandb()
            wandb.init(project=self.wandb_config['project_name'],
                       config=self.wandb_config,
                       name=self.wandb_config['run_name'],
                       entity="simonr98",
                       reinit=True,
                       settings=wandb.Settings(start_method="thread"))

        if self.algorithm == 'point_net_classification':
            train_loader, test_loader = get_model_net_data(num_points_pc=NUM_POINTS_POINT_CLOUD,
                                                           batch_size=BATCH_SIZE_VOXEL)

            algorithm = ClassificationAlgorithm(num_classes=train_loader.dataset.num_classes, lr=self.lr)

            for epoch in range(1, self.num_epochs + 1):
                loss = algorithm.train(train_loader)
                test_acc = algorithm.test(test_loader)
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
                if self.save:
                    T.save(algorithm.model.state_dict(), f'{self.model_save_path}_{epoch}_epochs')
                algorithm.scheduler.step()

                if self.wandb:
                    self.wandb_logging_parameters.update({'loss': loss, 'epochs': epoch, 'Test': test_acc})
                    wandb.log(self.wandb_logging_parameters)

        if self.algorithm == 'voxel_point_prediction':
            train_loader, test_loader = get_voxel_data_loaders(num_points_pc=NUM_POINTS_POINT_CLOUD,
                                                               track_point_index=INDEX_TRACK_POINT_VOXEL)
            algorithm = PointTrackingAlgorithm(point_dim=2, out_channels=2)

            for epoch in range(1, self.num_epochs + 1):
                loss = algorithm.train(train_loader)
                error = algorithm.test(test_loader)
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Error per sample: {error:.4f}')
                if self.save:
                    T.save(algorithm.model.state_dict(), f'{self.model_save_path}_{epoch}_epochs')
                algorithm.scheduler.step()

                if self.wandb:
                    self.wandb_logging_parameters.update({'loss': loss, 'epochs': epoch, 'error': error})
                    wandb.log(self.wandb_logging_parameters)

        def run_torus_algorithm(com: bool):
            train_loader, test_loader = get_torus_data_loaders(com=com)
            algorithm = PointTrackingAlgorithm(point_dim=3, out_channels=3)

            for epoch in range(1, self.num_epochs + 1):
                loss = algorithm.train(train_loader)
                error = algorithm.test(test_loader)
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Error per sample: {error:.4f}')
                if self.save:
                    T.save(algorithm.model.state_dict(), f'{self.model_save_path}_{epoch}_epochs')
                algorithm.scheduler.step()

                if self.wandb:
                    self.wandb_logging_parameters.update({'loss': loss, 'epochs': epoch, 'error': error})
                    wandb.log(self.wandb_logging_parameters)

        if self.algorithm == 'torus_com_prediction':
            run_torus_algorithm(com=True)

        if self.algorithm == 'torus_pos_prediction':
            run_torus_algorithm(com=False)

        else:
            print('Algorithm not implemented')
