import pybullet as p
import pybullet_data

physicsClient = p.connect(p.GUI)
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

counter = 0

while p.isConnected():
    p.stepSimulation()
    p.setGravity(0, 0, -10)

    if counter == 300:
        p.disconnect()

    counter += 1