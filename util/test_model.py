import torch as T

from algorithms.PointTrackingAlgorithm import Data
from models.TrackingNet import TrackingNet
from util.get_data import get_torus_data_loaders

############################################################
# TESTS
############################################################


def prepare_data_for_model(data):
    f_size = data.shape[0] * data.shape[1]
    batch = [[i] * data.shape[1] for i in range(len(data))]
    batch = [e for sub_list in batch for e in sub_list]
    batch = T.LongTensor(batch)
    data = T.reshape(data, (f_size, data.shape[2]))

    return Data(data=data, batch=batch)


train_loader, test_loader = get_torus_data_loaders(com=True, key='mesh')

MODEL_PATH = 'models_8_epochs'

for mesh, com in test_loader:
    mesh, com = mesh[:1, :, :], com[:1, :]

    print(mesh.shape)
    print(com.shape)

    input = prepare_data_for_model(mesh)
    model = TrackingNet(out_channels=3, point_dim=3)
    model.load_state_dict(T.load('MODEL_PATH'), strict=False)

    model.eval()

    predicted_com = model(input)

    print('real_com', com, 'predicted_com', predicted_com)

############################################################