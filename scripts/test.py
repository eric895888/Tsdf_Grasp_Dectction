import pybullet as p
p.connect(p.GUI)
p.resetSimulation()
p.setGravity(0,0,-9.8)
p.setTimeStep(1./60)
p.setRealTimeSimulation(0)


#有時間的話玩玩測試用的
id = p.loadURDF("/home/eric/catkin_ws/src/vgn/scripts/data/urdfs/panda/robotiq_85.urdf")
numJoints = p.getNumJoints(id)
joints_indexes = [i for i in range(numJoints) if p.getJointInfo(id, i)[2] != p.JOINT_FIXED]  # 可以使用的关节索引
print("_____")
for index in joints_indexes:
    print(p.getJointInfo(id, index) )
    break
print("!!!!!!!!!_____")
print(p.getJointInfo(id,0))
#024579