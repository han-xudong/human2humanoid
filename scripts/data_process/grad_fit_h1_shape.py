import glob
import os
import sys
import pdb
import os.path as osp
import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch, H1_ROTATION_AXIS


h1_fk = Humanoid_Batch(extend_head=True)  # load forward kinematics model
#### Define corresonpdances between h1 and smpl joints
h1_joint_names_augment = copy.deepcopy(h1_fk.model_names)
h1_joint_pick = [
    "pelvis",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_hand_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_hand_link",
    "head_link",
]
smpl_joint_pick = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "L_Shoulder",
    "L_Elbow",
    "L_Hand",
    "R_Shoulder",
    "R_Elbow",
    "R_Hand",
    "Head",
]
h1_joint_pick_idx = [h1_joint_names_augment.index(j) for j in h1_joint_pick]
smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]


#### Preparing fitting varialbes
device = torch.device("cpu")
dof_pos = torch.zeros((1, 19))
pose_aa_h1 = torch.cat(
    [
        torch.zeros((1, 1, 3)),
        H1_ROTATION_AXIS * dof_pos[..., None],
        torch.zeros((1, 2, 3)),
    ],
    axis=1,
)
print(f"Shape of pose_aa_h1: {pose_aa_h1.shape}")

root_trans = torch.zeros((1, 1, 3))

###### prepare SMPL default pause for H1
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
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index("L_Elbow")] = sRot.from_euler(
    "xyz", [0, -np.pi / 2, 0], degrees=False
).as_rotvec()
pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index("R_Elbow")] = sRot.from_euler(
    "xyz", [0, np.pi / 2, 0], degrees=False
).as_rotvec()
pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))

smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")

###### Shape fitting
trans = torch.zeros([1, 3])
beta = torch.zeros([1, 10])
verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta, trans)
offset = joints[:, 0] - trans
root_trans_offset = trans + offset
print("root_trans_offset:", root_trans_offset)

fk_return = h1_fk.fk_batch(pose_aa_h1[None,], root_trans_offset[None, 0:1])

shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
scale = Variable(torch.ones([1]).to(device), requires_grad=True)
optimizer_shape = torch.optim.Adam([shape_new, scale], lr=0.1)

h1_xyz = fk_return.global_translation_extend[:, :, h1_joint_pick_idx].detach().cpu().numpy()
smpl_xyz = joints[:, smpl_joint_pick_idx].detach().cpu().numpy()

n_h1 = min(len(h1_joint_pick), h1_xyz.shape[2])
n_smpl = min(len(smpl_joint_pick), smpl_xyz.shape[1])
smpl_skeleton_edges = [
    (0, 1),   # Pelvis -> L_Hip
    (1, 2),   # L_Hip -> L_Knee
    (2, 3),   # L_Knee -> L_Ankle
    (0, 4),   # Pelvis -> R_Hip
    (4, 5),   # R_Hip -> R_Knee
    (5, 6),   # R_Knee -> R_Ankle
    (0, 7),   # Pelvis -> L_Shoulder
    (7, 8),   # L_Shoulder -> L_Elbow
    (8, 9),   # L_Elbow -> L_Hand
    (0, 10),  # Pelvis -> R_Shoulder
    (10, 11), # R_Shoulder -> R_Elbow
    (11, 12), # R_Elbow -> R_Hand
    (0, 13),  # Pelvis -> Head
]
h1_skeleton_edges = smpl_skeleton_edges

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

plt.ion()
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
h1_xyz = np.asarray(h1_xyz).reshape(-1, 3)
smpl_xyz = np.asarray(smpl_xyz).reshape(-1, 3)
ax.scatter(h1_xyz[:,0], h1_xyz[:,1], h1_xyz[:,2], c='r', label='h1')
ax.scatter(smpl_xyz[:,0], smpl_xyz[:,1], smpl_xyz[:,2], c='b', label='SMPL')
for (i, j) in h1_skeleton_edges:
    ax.plot([h1_xyz[i,0], h1_xyz[j,0]],
            [h1_xyz[i,1], h1_xyz[j,1]],
            [h1_xyz[i,2], h1_xyz[j,2]], c='r')
for (i, j) in smpl_skeleton_edges:
    ax.plot([smpl_xyz[i,0], smpl_xyz[j,0]],
            [smpl_xyz[i,1], smpl_xyz[j,1]],
            [smpl_xyz[i,2], smpl_xyz[j,2]], c='b')
for i in range(n_h1):
    ax.text(h1_xyz[i,0], h1_xyz[i,1], h1_xyz[i,2], h1_joint_pick[i], color='r')
for i in range(n_smpl):
    ax.text(smpl_xyz[i,0], smpl_xyz[i,1], smpl_xyz[i,2], smpl_joint_pick[i], color='b')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('H1 and SMPL Joint Positions')
ax.legend()
set_axes_equal(ax)
plt.draw()
plt.pause(0.1)

for iteration in range(1000):
    verts, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])
    root_pos = joints[:, 0]
    joints = (joints - joints[:, 0]) * scale + root_pos
    diff = (
        fk_return.global_translation_extend[:, :, h1_joint_pick_idx]
        - joints[:, smpl_joint_pick_idx]
    )
    loss_g = diff.norm(dim=-1).mean()
    loss = loss_g
    if iteration % 100 == 0:
        print(iteration, loss.item() * 1000)
        
        ax.cla()
        h1_xyz = fk_return.global_translation_extend[:, :, h1_joint_pick_idx].detach().cpu().numpy()
        smpl_xyz = joints[:, smpl_joint_pick_idx].detach().cpu().numpy()
        h1_xyz = np.asarray(h1_xyz).reshape(-1, 3)
        smpl_xyz = np.asarray(smpl_xyz).reshape(-1, 3)
        ax.scatter(h1_xyz[:,0], h1_xyz[:,1], h1_xyz[:,2], c='r', label='h1')
        ax.scatter(smpl_xyz[:,0], smpl_xyz[:,1], smpl_xyz[:,2], c='b', label='SMPL')
        for (i, j) in h1_skeleton_edges:
            ax.plot([h1_xyz[i,0], h1_xyz[j,0]],
                    [h1_xyz[i,1], h1_xyz[j,1]],
                    [h1_xyz[i,2], h1_xyz[j,2]], c='r')
        for (i, j) in smpl_skeleton_edges:
            ax.plot([smpl_xyz[i,0], smpl_xyz[j,0]],
                    [smpl_xyz[i,1], smpl_xyz[j,1]],
                    [smpl_xyz[i,2], smpl_xyz[j,2]], c='b')
        for i in range(n_h1):
            ax.text(h1_xyz[i,0], h1_xyz[i,1], h1_xyz[i,2], h1_joint_pick[i], color='r')
        for i in range(n_smpl):
            ax.text(smpl_xyz[i,0], smpl_xyz[i,1], smpl_xyz[i,2], smpl_joint_pick[i], color='b')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('H1 and SMPL Joint Positions')
        set_axes_equal(ax)
        plt.draw()
        plt.pause(0.01)

    optimizer_shape.zero_grad()
    loss.backward()
    optimizer_shape.step()

print(f"shape_new = {shape_new.detach().cpu().numpy()}")
print(f"scale = {scale.detach().cpu().numpy()}")

os.makedirs("data/h1", exist_ok=True)
joblib.dump(
    (shape_new.detach(), scale), "data/h1/shape_optimized_v1.pkl"
)  # V2 has hip jointsrea
print(f"shape fitted and saved to data/h1/shape_optimized_v1.pkl")
print(f"shape: {shape_new.detach().cpu().numpy()}, scale: {scale.detach().cpu().numpy()}")

plt.ioff()
ax.cla()
h1_xyz = fk_return.global_translation_extend[:, :, h1_joint_pick_idx].detach().cpu().numpy()
smpl_xyz = joints[:, smpl_joint_pick_idx].detach().cpu().numpy()
h1_xyz = np.asarray(h1_xyz).reshape(-1, 3)
smpl_xyz = np.asarray(smpl_xyz).reshape(-1, 3)
ax.scatter(h1_xyz[:,0], h1_xyz[:,1], h1_xyz[:,2], c='r', label='h1')
ax.scatter(smpl_xyz[:,0], smpl_xyz[:,1], smpl_xyz[:,2], c='b', label='SMPL')
for (i, j) in h1_skeleton_edges:
    ax.plot([h1_xyz[i,0], h1_xyz[j,0]],
            [h1_xyz[i,1], h1_xyz[j,1]],
            [h1_xyz[i,2], h1_xyz[j,2]], c='r')
for (i, j) in smpl_skeleton_edges:
    ax.plot([smpl_xyz[i,0], smpl_xyz[j,0]],
            [smpl_xyz[i,1], smpl_xyz[j,1]],
            [smpl_xyz[i,2], smpl_xyz[j,2]], c='b')
for i in range(n_h1):
    ax.text(h1_xyz[i,0], h1_xyz[i,1], h1_xyz[i,2], h1_joint_pick[i], color='r')
for i in range(n_smpl):
    ax.text(smpl_xyz[i,0], smpl_xyz[i,1], smpl_xyz[i,2], smpl_joint_pick[i], color='b')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('H1 and SMPL Joint Positions')
set_axes_equal(ax)
plt.draw()
plt.show()