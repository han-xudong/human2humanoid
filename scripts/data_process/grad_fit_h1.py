import glob
import os
import sys
import pdb
import os.path as osp
import copy
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from smpl_sim.poselib.skeleton.skeleton3d import (
    SkeletonTree,
    SkeletonMotion,
    SkeletonState,
)
from smpl_sim.utils.smoothing_utils import gaussian_filter_1d_batch
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from phc.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
    SMPL_BONE_ORDER_NAMES,
)
import joblib
from phc.utils.rotation_conversions import axis_angle_to_matrix
from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch
from torch.autograd import Variable
from tqdm import tqdm
import argparse


def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    if not "mocap_framerate" in entry_data:
        return
    framerate = entry_data["mocap_framerate"]

    root_trans = entry_data["trans"]
    pose_aa = np.concatenate(
        [entry_data["poses"][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1
    )
    betas = entry_data["betas"]
    gender = entry_data["gender"]
    N = pose_aa.shape[0]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amass_root", type=str, default="data/AMASS/stable_punch"
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    h1_rotation_axis = torch.tensor(
        [
            [
                [0, 0, 1],  # l_hip_yaw
                [1, 0, 0],  # l_hip_roll
                [0, 1, 0],  # l_hip_pitch
                [0, 1, 0],  # kneel
                [0, 1, 0],  # ankle
                [0, 0, 1],  # r_hip_yaw
                [1, 0, 0],  # r_hip_roll
                [0, 1, 0],  # r_hip_pitch
                [0, 1, 0],  # kneel
                [0, 1, 0],  # ankle
                [0, 0, 1],  # torso
                [0, 1, 0],  # l_shoulder_pitch
                [1, 0, 0],  # l_roll_pitch
                [0, 0, 1],  # l_yaw_pitch
                [0, 1, 0],  # l_elbow
                [0, 1, 0],  # r_shoulder_pitch
                [1, 0, 0],  # r_roll_pitch
                [0, 0, 1],  # r_yaw_pitch
                [0, 1, 0],  # r_elbow
            ]
        ]
    ).to(device)

    h1_joint_names = [
        "pelvis",
        "left_hip_yaw_link",
        "left_hip_roll_link",
        "left_hip_pitch_link",
        "left_knee_link",
        "left_ankle_link",
        "right_hip_yaw_link",
        "right_hip_roll_link",
        "right_hip_pitch_link",
        "right_knee_link",
        "right_ankle_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
    ]

    h1_joint_names_augment = h1_joint_names + ["left_hand_link", "right_hand_link"]
    h1_joint_pick = [
        "pelvis",
        "left_knee_link",
        "left_ankle_link",
        "right_knee_link",
        "right_ankle_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_hand_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_hand_link",
    ]
    smpl_joint_pick = [
        "Pelvis",
        "L_Knee",
        "L_Ankle",
        "R_Knee",
        "R_Ankle",
        "L_Shoulder",
        "L_Elbow",
        "L_Hand",
        "R_Shoulder",
        "R_Elbow",
        "R_Hand",
    ]
    h1_joint_pick_idx = [h1_joint_names_augment.index(j) for j in h1_joint_pick]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
    smpl_parser_n.to(device)

    shape_new, scale = joblib.load("data/h1/shape_optimized_v1.pkl")
    shape_new = shape_new.to(device)

    amass_root = args.amass_root
    all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {
        "0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path
        for data_path in all_pkls
    }

    if len(key_name_to_pkls) == 0:
        raise ValueError(f"No motion files found in {amass_root}")

    h1_fk = Humanoid_Batch(device=device)
    
    plt.ion()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    fig2 = plt.figure(figsize=(10, 10))
    ax2 = fig2.add_subplot(111)

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
    
    data_dump = {}
    pbar = tqdm(key_name_to_pkls.keys())
    for data_key in pbar:
        amass_data = load_amass_data(key_name_to_pkls[data_key])
        if not amass_data:
            continue
        skip = int(amass_data["fps"] // 30)
        trans = torch.from_numpy(amass_data["trans"][::skip]).float().to(device)
        N = trans.shape[0]
        pose_aa_walk = (
            torch.from_numpy(
                np.concatenate(
                    (amass_data["pose_aa"][::skip, :66], np.zeros((N, 6))), axis=-1
                )
            )
            .float()
            .to(device)
        )

        verts, joints = smpl_parser_n.get_joints_verts(
            pose_aa_walk, torch.zeros((1, 10)).to(device), trans
        )
        offset = joints[:, 0] - trans
        root_trans_offset = trans + offset

        pose_aa_h1 = np.repeat(
            np.repeat(
                sRot.identity().as_rotvec()[
                    None,
                    None,
                    None,
                ],
                len(h1_joint_names_augment),
                axis=2,
            ),
            N,
            axis=1,
        )
        pose_aa_h1[..., 0, :] = (
            sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3])
            * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
        ).as_rotvec()
        pose_aa_h1 = torch.from_numpy(pose_aa_h1).float().to(device)
        # print(f"Shape of pose_aa_h1: {pose_aa_h1.shape}")
        gt_root_rot = (
            torch.from_numpy(
                (
                    sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3])
                    * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
                ).as_rotvec()
            )
            .float()
            .to(device)
        )

        dof_pos = torch.zeros((1, N, h1_rotation_axis.shape[1], 1)).to(device)

        dof_pos_new = Variable(dof_pos, requires_grad=True)
        optimizer_pose = torch.optim.Adadelta([dof_pos_new], lr=100)
        
        fk_return = h1_fk.fk_batch(pose_aa_h1, root_trans_offset[None,])
        
        h1_xyz = fk_return["global_translation_extend"][:, :, h1_joint_pick_idx].detach().cpu().numpy()
        smpl_xyz = joints[:, smpl_joint_pick_idx].detach().cpu().numpy()

        n_h1 = min(len(h1_joint_pick), h1_xyz.shape[2])
        n_smpl = min(len(smpl_joint_pick), smpl_xyz.shape[1])
        smpl_skeleton_edges = [
            (0, 1),   # Pelvis -> L_Knee
            (1, 2),   # L_Knee -> L_Ankle
            (0, 3),   # Pelvis -> R_Knee
            (3, 4),   # R_Knee -> R_Ankle
            (0, 5),   # Pelvis -> L_Shoulder
            (5, 6),   # L_Shoulder -> L_Elbow
            (6, 7),   # L_Elbow -> L_Hand
            (0, 8),   # Pelvis -> R_Shoulder
            (8, 9),   # R_Shoulder -> R_Elbow
            (9, 10),  # R_Elbow -> R_Hand
        ]
        h1_skeleton_edges = smpl_skeleton_edges
        
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

        for iteration in range(500):
            verts, joints = smpl_parser_n.get_joints_verts(
                pose_aa_walk, shape_new, trans
            )
            pose_aa_h1_new = torch.cat(
                [
                    gt_root_rot[None, :, None],
                    h1_rotation_axis * dof_pos_new,
                    torch.zeros((1, N, 2, 3)).to(device),
                ],
                axis=2,
            ).to(device)
            # print(f"Shape of pose_aa_h1_new: {pose_aa_h1_new.shape}")
            fk_return = h1_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None,])

            diff = (
                fk_return["global_translation_extend"][:, :, h1_joint_pick_idx]
                - joints[:, smpl_joint_pick_idx]
            )
            loss_g = diff.norm(dim=-1).mean()
            loss = loss_g

            pbar.set_description_str(f"{iteration} {loss.item() * 1000}")

            optimizer_pose.zero_grad()
            loss.backward()
            optimizer_pose.step()

            dof_pos_new.data.clamp_(
                h1_fk.joints_range[:, 0, None], h1_fk.joints_range[:, 1, None]
            )
            
            dof_pos_new.data = gaussian_filter_1d_batch(
                dof_pos_new.squeeze().transpose(1, 0)[None, ],
                kernel_size=5,
                sigma=0.75
            ).transpose(2, 1)[..., None]
            
            if iteration % 50 == 0:
                ax.cla()
                h1_xyz = fk_return["global_translation_extend"][:, :, h1_joint_pick_idx].detach().cpu().numpy()
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
                ax.legend()
                set_axes_equal(ax)
                plt.draw()
                plt.pause(0.01)
                
                ax2.cla()
                ax2.plot(dof_pos_new.data.squeeze().detach().cpu().numpy())
                ax2.set_xlabel('Joint Index')
                ax2.set_ylabel('Joint Angle (rad)')
                ax2.set_title("Ballbot Joint Angles")


        dof_pos_new.data.clamp_(
            h1_fk.joints_range[:, 0, None], h1_fk.joints_range[:, 1, None]
        )
        pose_aa_h1_new = torch.cat(
            [
                gt_root_rot[None, :, None],
                h1_rotation_axis * dof_pos_new,
                torch.zeros((1, N, 2, 3)).to(device),
            ],
            axis=2,
        )
        fk_return = h1_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None,])

        root_trans_offset_dump = root_trans_offset.clone()

        root_trans_offset_dump[..., 2] -= (
            fk_return.global_translation[..., 2].min().item() - 0.08
        )

        data_dump[data_key] = {
            "root_trans_offset": root_trans_offset_dump.squeeze()
            .cpu()
            .detach()
            .numpy(),
            "pose_aa": pose_aa_h1_new.squeeze().cpu().detach().numpy(),
            "dof": dof_pos_new.squeeze().detach().cpu().numpy(),
            "root_rot": sRot.from_rotvec(gt_root_rot.cpu().numpy()).as_quat(),
            "fps": 30,
        }

    #     print(f"dumping {data_key} for testing, remove the line if you want to process all data")
    #     import ipdb; ipdb.set_trace()
    #     joblib.dump(data_dump, "data/h1/test.pkl")

    # import ipdb; ipdb.set_trace()
    joblib.dump(data_dump, f"data/h1/{args.amass_root.split('/')[-1]}.pkl")
