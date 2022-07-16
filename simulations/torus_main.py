import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import pickle
from numpy.random import default_rng
from typing import Optional
from definitions import INDEX_TRACK_POINT_TORUS, TRAJECTORY_LENGTH_TORUS


def distance(a, b):
    return np.linalg.norm(a - b)


def get_environment_plot(camera_coordinates: Optional[list], mesh: np.ndarray,
                         pcd_without_noise: np.ndarray, pcd: np.ndarray):
    # Plot Mesh
    fig = plt.figure(1, figsize=(100, 100))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.title.set_text('Point cloud without noise')
    ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], c='red')

    # --------------------------------------------------------------------------------------------------------
    # Plot Point Cloud without noise
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.title.set_text('Point cloud without noise')
    ax.scatter(pcd_without_noise[:, 0], pcd_without_noise[:, 1], pcd_without_noise[:, 2], zdir='z', c='red')

    x, y, z = camera_coordinates[0], camera_coordinates[1], camera_coordinates[2]
    plt.plot([x], [y], [z], marker='o', markersize=10, color="black")

    # --------------------------------------------------------------------------------------------------------
    # Plot Point Cloud with noise
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.title.set_text('Point cloud with noise')
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='red')

    x, y, z = camera_coordinates[0], camera_coordinates[1], camera_coordinates[2]
    plt.plot([x], [y], [z], marker='o', markersize=10, color="black")
    # ----------------------------------------------------------------------------------------------------------

    plt.subplots_adjust()
    plt.show()


def main(test_mode: bool = False, save: bool = False, file_name: str = 't1.pkl'):
    p.connect(p.GUI)

    camera_coordinates = [0.3, 0.9, -1]

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    p.resetDebugVisualizerCamera(3, -420, -30, camera_coordinates)
    p.setGravity(0, 0, -10)

    tex = p.loadTexture("uvmap.png")
    planeId = p.loadURDF("plane.urdf", [0, 0, -2])
    boxId = p.loadURDF("cube.urdf", [0, 3, 2], useMaximalCoordinates=True)

    bunnyId = p.loadSoftBody("torus/torus_textured.obj", simFileName="torus.vtk", mass=4, useNeoHookean=1,
                             NeoHookeanMu=180, NeoHookeanLambda=600, NeoHookeanDamping=0.01, collisionMargin=0.006,
                             useSelfCollision=1, frictionCoeff=0.5, repulsionStiffness=800)

    p.changeVisualShape(bunnyId, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=tex, flags=0)
    p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
    p.setRealTimeSimulation(0)

    meshes, center_of_mass, pcds, pcds_without_noise, point_on_surface = [], [], [], [], []
    counter = 0

    while p.isConnected() and counter < TRAJECTORY_LENGTH_TORUS:
        p.stepSimulation()
        mesh = np.array(p.getMeshData(bunnyId)[1])
        average_of_mesh = np.sum(mesh, axis=0) / len(mesh)

        # store meshes, average of mesh, point_clouds,
        meshes.append(mesh)
        center_of_mass.append(average_of_mesh)
        p.getCameraImage(320, 200)

        # -------------------------------------------------------------------------
        # Sample a approximation of a point cloud from a set of mesh points

        # Compute the distance of all points to the Camera and take only the lower 30 percent quantile
        distance_to_camera = np.apply_along_axis(distance, 1, mesh, np.array(camera_coordinates))
        quantile = np.quantile(distance_to_camera, [0.3], axis=0)[0]
        pcd_without_noise = np.delete(mesh, np.where(distance_to_camera > quantile)[0], 0)

        # Randomize the order of the points
        np.random.shuffle(pcd_without_noise)

        # Add gaussian noise to the points (5% of the minimum of the mean of x,y and z axis)
        x_mean, y_mean, z_mean = np.mean(pcd_without_noise[:, 0]), np.mean(pcd_without_noise[:, 1]), np.mean(
            pcd_without_noise[:, 2])
        min_mean = min(abs(x_mean), abs(y_mean), abs(z_mean))
        noise = np.random.normal(0, 0.1 * min_mean, pcd_without_noise.shape)
        pcd = pcd_without_noise + noise
        # ---------------------------------------------------------------------------
        pcds.append(pcd)
        pcds_without_noise.append(pcd_without_noise)
        point_on_surface.append(mesh[INDEX_TRACK_POINT_TORUS])

        # ###############################################################################
        # TEST
        ###############################################################################
        if test_mode:
            get_environment_plot(camera_coordinates, mesh, pcd_without_noise, pcd)
            exit()
        # ###############################################################################

        # show average point in simulation
        p.addUserDebugPoints(np.reshape(average_of_mesh, (1, 3)), [[255, 0, 0]], 5)
        p.setGravity(0, 0, -10)
        counter += 1

    # pcds, pcds without noise have different lengths - randomly select n points from them
    min_num_points_pcd = min([len(x) for x in pcds] + [len(x) for x in pcds_without_noise])
    rng = default_rng()
    random_index = rng.choice(min_num_points_pcd, size=min_num_points_pcd, replace=False)

    def reduce(p):
        return [p[i] for i in random_index]

    pcds, pcds_without_noise = [reduce(pcds[i]) for i in range(len(pcds))], \
                               [reduce(pcds_without_noise[i]) for i in range(len(pcds))]

    if save:
        data = {
            'mesh': meshes,
            'com': center_of_mass,
            'pcd_without_noise': pcds_without_noise,
            'pcd': pcds,
            'pos': point_on_surface
        }

        with open(f'../data/torus/train/{file_name}', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        p.disconnect()


if __name__ == '__main__':
    test_mode = False
    save, file_name = True, 't2.pkl'

    main(test_mode=test_mode, save=save, file_name=file_name)
