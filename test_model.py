import torch as T
from matplotlib import pyplot as plt
from util.get_data import get_torus_data_loaders
from dataclasses import dataclass


@dataclass
class BatchData:
    data: T.Tensor
    batch: T.LongTensor


############################################################
# TESTS
############################################################

train_loader, test_loader = get_torus_data_loaders(com=True, key='pcd')

MODEL_PATH = 'models/torus_com_prediction_pcd_4_epochs'

for data in train_loader:
    model = T.load(MODEL_PATH).cpu()
    model.eval()

    predicted_com = model(data)
    predicted_com = predicted_com.detach().numpy()

    num_inputs = len(data.pos) // data.batch[-1]

    # --------------------------------------------------------------------------------------------------------

    #####################
    # plot point of interest over time
    #####################

    fig = plt.figure(1, figsize=(100, 100))

    for i in range(1, 20):
        fig.add_subplot(5, 4, i)

        ax = fig.add_subplot(5, 4, i, projection='3d')
        ax.set_title('Point cloud with noise', {'fontsize': 10})
        ax.scatter(data.pos[:num_inputs * i, 0], data.pos[:num_inputs * i, 1], data.pos[:num_inputs * i, 2], c='red')

        # x, y, z = CAMERA_COORDINATES_TORUS[0], CAMERA_COORDINATES_TORUS[1], CAMERA_COORDINATES_TORUS[2]

        plt.plot([predicted_com[i, 0]], [predicted_com[i, 1]], [predicted_com[i, 2]], marker='o', markersize=10,
                 color="blue")

        plt.plot([data.y[i, 0]], [data.y[i, 1]], [data.y[i, 2]], marker='o', markersize=10, color="green")

        i += 1

    plt.show()

############################################################
