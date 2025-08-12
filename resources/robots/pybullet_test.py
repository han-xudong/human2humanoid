import pathlib
import pybullet as p

robot = "ballbot"

p.connect(p.GUI)
robot_id = p.loadURDF(str(pathlib.Path(__file__).parent / f"{robot}/urdf/{robot}.urdf"))

# 查看base link信息
base_info = p.getBodyInfo(robot_id)
print("Base link name:", base_info[0].decode("utf-8"))

# 查看所有joint和link信息
num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    print(f"Joint {i}: name={joint_info[1].decode('utf-8')}, link={joint_info[12].decode('utf-8')}")
input("Press Enter to exit...")
p.disconnect()