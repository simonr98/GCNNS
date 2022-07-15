from torch_geometric.loader import DataLoader
from VoxelDataset import VoxelDataset
from algorithms.PointTrackingAlgorithm import PointTrackingAlgorithm
from preprocessing import get_all_trajectories, get_train_data, get_test_data, join_trajectories

# Track Graph point with index 1!
_, pcd_train, y_train = get_all_trajectories(get_train_data(), 1)
pcd_train, y_train = join_trajectories(pcd_train), join_trajectories(y_train)

train_dataset = VoxelDataset(pcd_train, y_train)


_, pcd_test, y_test = get_all_trajectories(get_test_data(), 40)
pcd_test, y_test = join_trajectories(pcd_test), join_trajectories(y_test)

test_dataset = VoxelDataset(pcd_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=6)

algorithm = PointTrackingAlgorithm()

for epoch in range(1, 201):
    loss = algorithm.train(train_loader)
    error = algorithm.test(test_loader)
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Error per sample: {error:.4f}')
    algorithm.scheduler.step()

