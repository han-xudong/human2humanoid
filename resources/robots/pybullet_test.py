import pathlib
import pybullet as p

robot = "ballbot"

p.connect(p.GUI)
p.loadURDF(str(pathlib.Path(__file__).parent / f"{robot}/urdf/{robot}.urdf"))
input("Press Enter to exit...")
p.disconnect()