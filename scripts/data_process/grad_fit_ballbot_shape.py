import glob
import os
import sys
import pdb
import os.path as osp
import copy

sys.path.append(os.getcwd())

from phc.utils import torch_utils
from smpl_sim.poselib.skeleton.skeleton3d import (
    SkeletonTree,
    SkeletonMotion,
    SkeletonState,
)
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)

import joblib
import torch
import torch.nn.functional as F
import math
from phc.utils.pytorch3d_transforms import axis_angle_to_matrix
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter1d
from tqdm.notebook import tqdm
from smpl_sim.smpllib.smpl_joint_names import (
    SMPL_MUJOCO_NAMES,
    SMPL_BONE_ORDER_NAMES,
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)
from phc.utils.torch_ballbot_batch import Ballbot_Batch, BALLBOT_ROTATION_AXIS

ballbot_joint_names = [
    "base_link",
    "torso_link",
    "neck_link",
    "head_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_hand_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_hand_link",
    "axle_1_link",
    "fat_roller_1_1_link",
    "fat_roller_1_2_link",
    "fat_roller_1_3_link",
    "fat_roller_1_4_link",
    "fat_roller_1_5_link",
    "fat_roller_1_6_link",
    "thin_roller_1_1_link",
    "thin_roller_1_2_link",
    "thin_roller_1_3_link",
    "thin_roller_1_4_link",
    "thin_roller_1_5_link",
    "thin_roller_1_6_link",
    "axle_2_link",
    "fat_roller_2_1_link",
    "fat_roller_2_2_link",
    "fat_roller_2_3_link",
    "fat_roller_2_4_link",
    "fat_roller_2_5_link",
    "fat_roller_2_6_link",
    "thin_roller_2_1_link",
    "thin_roller_2_2_link",
    "thin_roller_2_3_link",
    "thin_roller_2_4_link",
    "thin_roller_2_5_link",
    "thin_roller_2_6_link",
    "axle_3_link",
    "fat_roller_3_1_link",
    "fat_roller_3_2_link",
    "fat_roller_3_3_link",
    "fat_roller_3_4_link",
    "fat_roller_3_5_link",
    "fat_roller_3_6_link",
    "thin_roller_3_1_link",
    "thin_roller_3_2_link",
    "thin_roller_3_3_link",
    "thin_roller_3_4_link",
    "thin_roller_3_5_link",
    "thin_roller_3_6_link",
    "ball_link",
]
ballbot_fk = Ballbot_Batch(extend_hand=False)  # load forward kinematics model

#### Define correspondences between ballbot and smpl joints
ballbot_joint_names_augment = copy.deepcopy(ballbot_joint_names)
ballbot_joint_pick = [
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_hand_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_hand_link",
    "head_link",
]
smpl_joint_pick = [
    "Torso",
    "L_Shoulder",
    "L_Elbow",
    "L_Hand",
    "R_Shoulder",
    "R_Elbow",
    "R_Hand",
    "Head",
]
ballbot_joint_pick_idx = [
    ballbot_joint_names_augment.index(name) for name in ballbot_joint_pick
]
smpl_joint_pick_idx = [
    SMPL_BONE_ORDER_NAMES.index(name) for name in smpl_joint_pick
]

#### Preparing fitting variables
device = torch.device("cpu")
dof_pos = torch.zeros((1, len(BALLBOT_ROTATION_AXIS)))
pose_aa_ballbot = torch.cat(
    [
        torch.zeros((1, 2, 3)),
        BALLBOT_ROTATION_AXIS * dof_pos[..., None],
        torch.zeros((1, 3, 3)),
    ],
    axis=1,
)
print(f"Shape of pose_aa_ballbot: {pose_aa_ballbot.shape}")

root_trans = torch.zeros((1, 1, 3))

###### prepare SMPL default pose for Ballbot
pose_aa_stand = np.zeros((1, 72))
rotvec = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec()
pose_aa_stand[:, :3] = rotvec
pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index("L_Shoulder")] = sRot.from_euler(
    "xyz", [0, 0, -np.pi / 2], degrees=False
).as_rotvec()
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index("R_Shoulder")] = sRot.from_euler(
    "xyz", [0, 0, np.pi / 2], degrees=False
).as_rotvec()
pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))

smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")

###### Shape fitting
trans = torch.zeros([1, 3])
beta = torch.zeros([1, 10])
verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta, trans)
offset = joints[:, 0] - trans
root_trans_offset = trans + offset

print(f"SMPL root trans offset: {root_trans_offset}")

fk_return = ballbot_fk.fk_batch(pose_aa_ballbot[None,], root_trans_offset[None, 0:1])

shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
scale = Variable(torch.ones([1]).to(device), requires_grad=True)
optimizer_shape = torch.optim.Adam([shape_new, scale], lr=0.1)

for iteration in range(2000):
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])
    root_pos = joints[:, 0]
    joints = (joints - joints[:, 0]) * scale + root_pos
    diff = (
        fk_return.global_translation[:, :, ballbot_joint_pick_idx]
        - joints[:, smpl_joint_pick_idx]
    )
    loss_g = diff.norm(dim=-1).mean()
    loss = loss_g
    if iteration % 100 == 0:
        print(iteration, loss.item() * 1000)

    optimizer_shape.zero_grad()
    loss.backward()
    optimizer_shape.step()
    
# Save the fitted shape parameters
os.makedirs("data/ballbot", exist_ok=True)
joblib.dump(
    (shape_new.detach(), scale), "data/ballbot/shape_optimized_v1.pkl"
)  # V2 has hip jointsrea
print(f"shape fitted and saved to data/ballbot/shape_optimized_v1.pkl")
print(f"shape: {shape_new.detach().cpu().numpy()}, scale: {scale.detach().cpu().numpy()}")
