import pybullet as p
import pybullet_data
import numpy as np


physicsClient = p.connect(p.GUI)

camera_coordinates = [0.3, 0.9, -1]

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
p.resetDebugVisualizerCamera(3, -420, -30, camera_coordinates)
p.setGravity(0, 0, -10)

tex = p.loadTexture("uvmap.png")
planeId = p.loadURDF("plane.urdf", [0, 0, -2])
boxId = p.loadURDF("cube.urdf", [0, 3, 2], useMaximalCoordinates=True)

bunnyId = p.loadSoftBody("torus/torus_textured.obj", simFileName="torus.vtk", mass=3, useNeoHookean=1,
                         NeoHookeanMu=180, NeoHookeanLambda=600, NeoHookeanDamping=0.01, collisionMargin=0.006,
                         useSelfCollision=1, frictionCoeff=0.5, repulsionStiffness=800)

p.changeVisualShape(bunnyId, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=tex, flags=0)
p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
p.setRealTimeSimulation(0)

meshes, averages, pcds = [], [], []

counter = 0

while p.isConnected():
    p.stepSimulation()
    mesh = np.array(p.getMeshData(bunnyId)[1])
    average_of_mesh = np.sum(mesh, axis=0) / len(mesh)

    # store meshes
    meshes.append(mesh)
    averages.append(average_of_mesh)
    p.getCameraImage(320, 200)

    def distance(a, b):
        return np.linalg.norm(a-b)

    distance_to_camera = np.apply_along_axis(distance, 1, mesh, np.array(camera_coordinates))

    quantile = np.quantile(distance_to_camera, [0.1], axis=0)[0]

    #  Remove the rows whose first item is between 20 and 25
    pcd = np.delete(mesh, np.where(distance_to_camera > quantile)[0], 0)

    # Random order for points to get Point Cloud data
    np.random.shuffle(pcd)

    pcds.append(pcd)

    ###############################################################################
    # TEST
    ###############################################################################

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(mesh[:,0], mesh[:,1], mesh[:,2], zdir='z', c='red')
    #
    # plt.show()
    #
    # exit()

    # show average point in simulation
    p.addUserDebugPoints(np.reshape(average_of_mesh, (1,3)), [[255,0,0]], 5)
    p.setGravity(0, 0, -10)

    if counter == 1000:
        p.disconnect()

    counter += 1
