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
from phc.utils.torch_utils import calc_heading_quat
from phc.utils.torch_ballbot_batch import Ballbot_Batch
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
    parser.add_argument("--amass_root", type=str, default="data/AMASS/test")
    args = parser.parse_args()

    device = torch.device("cpu")

    ballbot_rotation_axis = torch.tensor(
        [
            [
                [0, 0, 1],  # r_pitch
                [0, 0, 1],  # r_rolls
                [0, 0, 1],  # r_yaw
                [0, 0, 1],  # r_elbow
                [0, 0, 1],  # l_pitch
                [0, 0, 1],  # l_roll
                [0, 0, 1],  # l_yaw
                [0, 0, 1],  # l_elbow
                [0, 0, 1],  # neck
                [0, 0, 1],  # head
                # [0, 0, 1],  # axle_1
                # [1, 0, 0],  # fat_roller_1_1
                # [1, 0, 0],  # fat_roller_1_2
                # [1, 0, 0],  # fat_roller_1_3
                # [1, 0, 0],  # fat_roller_1_4
                # [1, 0, 0],  # fat_roller_1_5
                # [1, 0, 0],  # fat_roller_1_6
                # [1, 0, 0],  # thin_roller_1_1
                # [1, 0, 0],  # thin_roller_1_2
                # [1, 0, 0],  # thin_roller_1_3
                # [1, 0, 0],  # thin_roller_1_4
                # [1, 0, 0],  # thin_roller_1_5
                # [1, 0, 0],  # thin_roller_1_6
                # [0, 0, 1],  # axle_2
                # [1, 0, 0],  # fat_roller_2_1
                # [1, 0, 0],  # fat_roller_2_2
                # [1, 0, 0],  # fat_roller_2_3
                # [1, 0, 0],  # fat_roller_2_4
                # [1, 0, 0],  # fat_roller_2_5
                # [1, 0, 0],  # fat_roller_2_6
                # [1, 0, 0],  # thin_roller_2_1
                # [1, 0, 0],  # thin_roller_2_2
                # [1, 0, 0],  # thin_roller_2_3
                # [1, 0, 0],  # thin_roller_2_4
                # [1, 0, 0],  # thin_roller_2_5
                # [1, 0, 0],  # thin_roller_2_6
                # [0, 0, 1],  # axle_3
                # [1, 0, 0],  # fat_roller_3_1
                # [1, 0, 0],  # fat_roller_3_2
                # [1, 0, 0],  # fat_roller_3_3
                # [1, 0, 0],  # fat_roller_3_4
                # [1, 0, 0],  # fat_roller_3_5
                # [1, 0, 0],  # fat_roller_3_6
                # [1, 0, 0],  # thin_roller_3_1
                # [1, 0, 0],  # thin_roller_3_2
                # [1, 0, 0],  # thin_roller_3_3
                # [1, 0, 0],  # thin_roller_3_4
                # [1, 0, 0],  # thin_roller_3_5
                # [1, 0, 0],  # thin_roller_3_6
            ]
        ]
    ).to(device)

    ballbot_fk = Ballbot_Batch(device=device)
    
    ballbot_joint_names_augment = copy.deepcopy(ballbot_fk.model_names)
    ballbot_joint_pick = [
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_hand_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_hand_link",
        # "head_link",
    ]
    smpl_joint_pick = [
        "Torso",
        "L_Shoulder",
        "L_Elbow",
        "L_Hand",
        "R_Shoulder",
        "R_Elbow",
        "R_Hand",
        # "Head",
    ]
    ballbot_joint_pick_idx = [
        ballbot_joint_names_augment.index(j) for j in ballbot_joint_pick
    ]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
    smpl_parser_n.to(device)

    shape_new, scale = joblib.load("data/ballbot/shape_optimized_v1.pkl")
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

        with torch.no_grad():
            verts, joints = smpl_parser_n.get_joints_verts(
                pose_aa_walk, shape_new, trans
            )
            root_pos = joints[:, 0:1]
            joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos
        
        joints[..., 2] -= verts[0, :, 2].min().item()
        offset = joints[:, 0] - trans
        
        root_trans_offset = (trans + offset).clone()
        root_pos_offset = torch.zeros(1, 3)
        root_pos_offset[:, 2] = -0.3  # Set a fixed z-offset for the root position
        root_trans_offset += root_pos_offset
        # root_trans_offset[:, 2] = -0.001  # Set a fixed z-offset for the root translation
        print(f"root_trans_offset:", root_trans_offset[None, ])
        pose_aa_ballbot = np.repeat(
            np.repeat(
                sRot.identity().as_rotvec()[
                    None,
                    None,
                    None,
                ],
                len(ballbot_joint_names_augment),
                axis=2,
            ),
            N,
            axis=1,
        )
        pose_aa_ballbot[..., 0, :] = (
            sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3])
            * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
        ).as_rotvec()
        pose_aa_ballbot = torch.from_numpy(pose_aa_ballbot).float().to(device)
        # print(f"Shape of pose_aa_ballbot: {pose_aa_ballbot.shape}")
        gt_root_rot_quat = torch.from_numpy((sRot.from_rotvec(pose_aa_walk[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat()).float() # can't directly use this 
        gt_root_rot = torch.from_numpy(sRot.from_quat(calc_heading_quat(gt_root_rot_quat)).as_rotvec()).float() # so only use the heading. 

        dof_pos = torch.zeros((1, N, ballbot_rotation_axis.shape[1], 1)).to(device)
        root_rot = Variable(gt_root_rot.clone(), requires_grad=True)
        dof_pos_new = Variable(dof_pos, requires_grad=True)
        optimizer_pose = torch.optim.Adadelta([dof_pos_new], lr=100)
        
        fk_return = ballbot_fk.fk_batch(pose_aa_ballbot, root_trans_offset[None,])
        

        ballbot_xyz = fk_return["global_translation_extend"][:, :, ballbot_joint_pick_idx].detach().cpu().numpy()
        smpl_xyz = joints[:, smpl_joint_pick_idx].detach().cpu().numpy()
        
        n_ballbot = min(len(ballbot_joint_pick), ballbot_xyz.shape[2])
        n_smpl = min(len(smpl_joint_pick), smpl_xyz.shape[1])
        smpl_skeleton_edges = [
            (0, 1),  # Pelvis -> L_Shoulder
            (1, 2),  # L_Shoulder -> L_Elbow
            (2, 3),  # L_Elbow -> L_Hand
            (0, 4),  # Pelvis -> R_Shoulder
            (4, 5),  # R_Shoulder -> R_Elbow
            (5, 6),  # R_Elbow -> R_Hand
        ]
        ballbot_skeleton_edges = smpl_skeleton_edges
        
        plt.ion()
        ax = fig.add_subplot(111, projection='3d')
        ballbot_xyz = np.asarray(ballbot_xyz).reshape(-1, 3)
        smpl_xyz = np.asarray(smpl_xyz).reshape(-1, 3)
        ax.scatter(ballbot_xyz[:, 0], ballbot_xyz[:, 1], ballbot_xyz[:, 2], c='r', label='Ballbot')
        ax.scatter(smpl_xyz[:, 0], smpl_xyz[:, 1], smpl_xyz[:, 2], c='b', label='SMPL')
        for (i, j) in ballbot_skeleton_edges:
            ax.plot([ballbot_xyz[i, 0], ballbot_xyz[j, 0]],
                    [ballbot_xyz[i, 1], ballbot_xyz[j, 1]],
                    [ballbot_xyz[i, 2], ballbot_xyz[j, 2]], c='r')
        for (i, j) in smpl_skeleton_edges:
            ax.plot([smpl_xyz[i, 0], smpl_xyz[j, 0]],
                    [smpl_xyz[i, 1], smpl_xyz[j, 1]],
                    [smpl_xyz[i, 2], smpl_xyz[j, 2]], c='b')
        for i in range(n_ballbot):
            ax.text(ballbot_xyz[i, 0], ballbot_xyz[i, 1], ballbot_xyz[i, 2], ballbot_joint_pick[i], color='r')
        for i in range(n_smpl):
            ax.text(smpl_xyz[i, 0], smpl_xyz[i, 1], smpl_xyz[i, 2], smpl_joint_pick[i], color='b')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Ballbot and SMPL Joint Positions")
        ax.legend()
        set_axes_equal(ax)
        plt.draw()
        plt.pause(0.1)
        
        ax2.plot(dof_pos_new.data.squeeze().detach().cpu().numpy())
        ax2.set_xlabel('Frames')
        ax2.set_ylabel('Joint Angle (rad)')
        ax2.set_title("Ballbot Joint Angles")
        ax2.legend(ballbot_joint_names_augment[2:2 + dof_pos.shape[2]], loc='upper right')

        for iteration in range(500):
            verts, joints = smpl_parser_n.get_joints_verts(
                pose_aa_walk, shape_new, trans
            )
            pose_aa_ballbot_new = torch.cat(
                [
                    root_rot[None, :, None],
                    torch.zeros((1, N, 1, 3)).to(device),
                    ballbot_rotation_axis * dof_pos_new,
                    torch.zeros((1, N, 36, 3)).to(device),
                ],
                axis=2,
            ).to(device)
            # print(f"Shape of pose_aa_ballbot_new: {pose_aa_ballbot_new.shape}")
            fk_return = ballbot_fk.fk_batch(
                pose_aa_ballbot_new, root_trans_offset[None,]
            )

            diff = (
                fk_return["global_translation_extend"][:, :, ballbot_joint_pick_idx]
                - joints[:, smpl_joint_pick_idx]
            )
            loss_g = diff.norm(dim=-1).mean()
            loss = loss_g

            pbar.set_description_str(f"{iteration} {loss.item() * 1000}")

            optimizer_pose.zero_grad()
            loss.backward()
            optimizer_pose.step()
            
            dof_pos_new.data.clamp_(
                ballbot_fk.joints_range[:dof_pos_new.data.shape[2], 0, None], ballbot_fk.joints_range[:dof_pos_new.data.shape[2], 1, None]
            )
            
            dof_pos_new.data = gaussian_filter_1d_batch(
                dof_pos_new.squeeze().transpose(1, 0)[None, ],
                kernel_size=5,
                sigma=0.75
            ).transpose(2, 1)[..., None]
            
            if iteration % 50 == 0:
                ax.cla()
                ballbot_xyz = fk_return["global_translation_extend"][:, :, ballbot_joint_pick_idx].detach().cpu().numpy()
                smpl_xyz = joints[:, smpl_joint_pick_idx].detach().cpu().numpy()
                ballbot_xyz = np.asarray(ballbot_xyz).reshape(-1, 3)
                smpl_xyz = np.asarray(smpl_xyz).reshape(-1, 3)
                ax.scatter(ballbot_xyz[:, 0], ballbot_xyz[:, 1], ballbot_xyz[:, 2], c='r', label='Ballbot')
                ax.scatter(smpl_xyz[:, 0], smpl_xyz[:, 1], smpl_xyz[:, 2], c='b', label='SMPL')
                for (i, j) in ballbot_skeleton_edges:
                    ax.plot([ballbot_xyz[i, 0], ballbot_xyz[j, 0]],
                            [ballbot_xyz[i, 1], ballbot_xyz[j, 1]],
                            [ballbot_xyz[i, 2], ballbot_xyz[j, 2]], c='r')
                for (i, j) in smpl_skeleton_edges:
                    ax.plot([smpl_xyz[i, 0], smpl_xyz[j, 0]],
                            [smpl_xyz[i, 1], smpl_xyz[j, 1]],
                            [smpl_xyz[i, 2], smpl_xyz[j, 2]], c='b')
                for i in range(n_ballbot):
                    ax.text(ballbot_xyz[i, 0], ballbot_xyz[i, 1], ballbot_xyz[i, 2], ballbot_joint_pick[i], color='r')
                for i in range(n_smpl):
                    ax.text(smpl_xyz[i, 0], smpl_xyz[i, 1], smpl_xyz[i, 2], smpl_joint_pick[i], color='b')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title("Ballbot and SMPL Joint Positions")
                ax.legend()
                set_axes_equal(ax)
                plt.draw()
                plt.pause(0.01)
                
                ax2.cla()
                ax2.plot(dof_pos_new.data.squeeze().detach().cpu().numpy())
                ax2.set_xlabel('Frames')
                ax2.set_ylabel('Joint Angle (rad)')
                ax2.set_title("Ballbot Joint Angles")
                ax2.legend(ballbot_joint_names_augment[2:2 + dof_pos.shape[2]], loc='upper right')

        dof_pos_new.data.clamp_(
            ballbot_fk.joints_range[:dof_pos_new.data.shape[2], 0, None], ballbot_fk.joints_range[:dof_pos_new.data.shape[2], 1, None]
        )
        pose_aa_ballbot_new = torch.cat(
            [
                root_rot[None, :, None],
                torch.zeros((1, N, 1, 3)).to(device),
                ballbot_rotation_axis * dof_pos_new,
                torch.zeros((1, N, 36, 3)).to(device),
            ],
            axis=2,
        )
        fk_return = ballbot_fk.fk_batch(pose_aa_ballbot_new, root_trans_offset[None,])

        height_diff = fk_return.global_translation[..., 2].min().item() 
        
        root_trans_offset_dump = (root_trans_offset + root_pos_offset).clone()

        root_trans_offset_dump[..., 2] -= height_diff
        joints_dump = joints.numpy().copy()
        joints_dump[..., 2] -= height_diff
        # root_trans_offset_dump[..., 2] -= (
        #     fk_return.global_translation_extend[..., 2].min().item() - 0.012
        # )

        data_dump[data_key] = {
            "root_trans_offset": root_trans_offset_dump.squeeze()
            .cpu()
            .detach()
            .numpy(),
            "pose_aa": pose_aa_ballbot_new.squeeze().cpu().detach().numpy(),
            "dof": dof_pos_new.squeeze().detach().cpu().numpy(),
            "root_rot": sRot.from_rotvec(root_rot.detach().cpu().numpy()).as_quat(),
            "fps": 30,
        }
        
        plt.ioff()
        ax.cla()
        ballbot_xyz = np.asarray(ballbot_xyz).reshape(-1, 3)
        smpl_xyz = np.asarray(smpl_xyz).reshape(-1, 3)
        ax.scatter(ballbot_xyz[:, 0], ballbot_xyz[:, 1], ballbot_xyz[:, 2], c='r', label='Ballbot')
        ax.scatter(smpl_xyz[:, 0], smpl_xyz[:, 1], smpl_xyz[:, 2], c='b', label='SMPL')
        for (i, j) in ballbot_skeleton_edges:
            ax.plot([ballbot_xyz[i, 0], ballbot_xyz[j, 0]],
                    [ballbot_xyz[i, 1], ballbot_xyz[j, 1]],
                    [ballbot_xyz[i, 2], ballbot_xyz[j, 2]], c='r')
        for (i, j) in smpl_skeleton_edges:
            ax.plot([smpl_xyz[i, 0], smpl_xyz[j, 0]],
                    [smpl_xyz[i, 1], smpl_xyz[j, 1]],
                    [smpl_xyz[i, 2], smpl_xyz[j, 2]], c='b')
        for i in range(n_ballbot):
            ax.text(ballbot_xyz[i, 0], ballbot_xyz[i, 1], ballbot_xyz[i, 2], ballbot_joint_pick[i], color='r')
        for i in range(n_smpl):
            ax.text(smpl_xyz[i, 0], smpl_xyz[i, 1], smpl_xyz[i, 2], smpl_joint_pick[i], color='b')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Ballbot and SMPL Joint Positions")
        ax.legend()
        set_axes_equal(ax)
        plt.draw()
        plt.show()

    #     print(f"dumping {data_key} for testing, remove the line if you want to process all data")
    #     import ipdb; ipdb.set_trace()
    #     joblib.dump(data_dump, "data/ballbot/test.pkl")

    # import ipdb; ipdb.set_trace()
    joblib.dump(data_dump, f"data/ballbot/{args.amass_root.split('/')[-1]}.pkl")
