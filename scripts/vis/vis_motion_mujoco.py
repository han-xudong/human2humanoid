import os
import sys
import copy
import joblib
import numpy as np
import torch
from phc.utils.motion_lib_h1 import MotionLibH1
from phc.utils.motion_lib_ballbot import MotionLibBallbot
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch
from phc.utils.torch_ballbot_batch import Ballbot_Batch
import mujoco
import mujoco.viewer
import glfw
import matplotlib.pyplot as plt
import time
import threading

# Use same robot and motion_lib logic as vis_motion.py
robot = "ballbot"  # "h1" or "ballbot"
robot_xml = f"resources/robots/{robot}/{robot}.xml"
sk_tree = SkeletonTree.from_mjcf(robot_xml)
motion_file = f"data/{robot}/test.pkl"
if robot == "ballbot":
    robot_batch = Ballbot_Batch()
elif robot == "h1":
    robot_batch = Humanoid_Batch(extend_head=True)

# Load motion_lib
motion_lib = MotionLibBallbot(motion_file=motion_file, device=torch.device("cpu"), masterfoot_conifg=None, fix_height=False, multi_thread=False, mjcf_file=robot_xml) if robot == "ballbot" else MotionLibH1(motion_file=motion_file, device=torch.device("cpu"), masterfoot_conifg=None, fix_height=False, multi_thread=False, mjcf_file=robot_xml)
with open(motion_file, "rb") as f:
    motion_data = joblib.load(f)
num_motions = len(motion_data)
motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False)
motion_keys = motion_lib.curr_motion_keys

# Load Mujoco model
mujoco_model = mujoco.MjModel.from_xml_path(robot_xml)
data = mujoco.MjData(mujoco_model)

# Visualization loop with glfw
motion_id = 0
time_step = 0.0
dt = mujoco_model.opt.timestep

start_event = threading.Event()
pause_event = threading.Event()
pause_event.set()  # 初始为运行状态

# GLFW key callback

def key_callback(window, key, scancode, action, mods):
    global motion_id, time_step, dof_history, step_count, lines
    if action == glfw.PRESS:
        if key == glfw.KEY_LEFT:
            motion_id = (motion_id - 1) % num_motions
            time_step = 0.0
        elif key == glfw.KEY_RIGHT:
            motion_id = (motion_id + 1) % num_motions
            time_step = 0.0
        elif key == glfw.KEY_SPACE:
            if pause_event.is_set():
                pause_event.clear()  # 暂停
            else:
                pause_event.set()    # 继续
        elif key == glfw.KEY_R:
            time_step = 0.0  # 重新开始
            dof_history.clear()
            step_count = 0
            # 清理 matplotlib axes 上的所有 Line2D 对象
            for line in ax.lines:
                line.remove()
            ax.lines.clear()
            lines.clear()
            joint_names = [mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mujoco_model.njnt)]
            joint_names = joint_names[1:num_dof_plot + 1]
            lines = setup_plot(ax, num_dof_plot, joint_names)
            # 立即补一帧数据，保证有折线显示
            if last_dof_pos is not None:
                dof_pos_np = last_dof_pos[0, 1:].cpu().numpy() if hasattr(last_dof_pos, 'cpu') else np.array(last_dof_pos[0, 1:])
                dof_history.append(dof_pos_np.copy())
                hist_arr = np.array(dof_history)
                x_vals = np.arange(len(dof_history))
                for i in range(num_dof_plot):
                    lines[i].set_data(x_vals, hist_arr[:, i])
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.001)

# Initialize glfw window
if not glfw.init():
    raise Exception("Could not initialize glfw")
window = glfw.create_window(1280, 720, "Mujoco Motion Viewer", None, None)
glfw.make_context_current(window)
glfw.set_key_callback(window, key_callback)

# Initialize mujoco OpenGL rendering context
scene = mujoco.MjvScene(mujoco_model, maxgeom=10000)
cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
con = mujoco.MjrContext(mujoco_model, mujoco.mjtFont.mjFONT_NORMAL)
viewport = mujoco.MjrRect(0, 0, 1280, 720)

# Camera initial parameters
cam.azimuth = 90.0
cam.elevation = -20.0
cam.distance = 3.0
cam.lookat[:] = np.array([0.0, 0.0, 1.0])

mouse_left_pressed = False
mouse_right_pressed = False
last_x, last_y = 0, 0

def mouse_button_callback(window, button, action, mods):
    global mouse_left_pressed, mouse_right_pressed, last_x, last_y
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            mouse_left_pressed = True
            last_x, last_y = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            mouse_left_pressed = False
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        if action == glfw.PRESS:
            mouse_right_pressed = True
            last_x, last_y = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            mouse_right_pressed = False

def cursor_pos_callback(window, xpos, ypos):
    global last_x, last_y
    if mouse_left_pressed:
        dx = xpos - last_x
        dy = ypos - last_y
        cam.azimuth -= dx * 0.5
        cam.elevation -= dy * 0.5
        last_x, last_y = xpos, ypos
    elif mouse_right_pressed:
        dx = xpos - last_x
        dy = ypos - last_y
        azimuth_rad = np.deg2rad(cam.azimuth)
        right = np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0])
        forward = np.array([-np.sin(azimuth_rad), np.cos(azimuth_rad), 0])
        speed = 0.001 * cam.distance
        cam.lookat[:3] += right * (dy * speed) + forward * (dx * speed)
        last_x, last_y = xpos, ypos

def scroll_callback(window, xoffset, yoffset):
    cam.distance *= (1.0 - yoffset * 0.05)
    cam.distance = max(0.1, cam.distance)

glfw.set_mouse_button_callback(window, mouse_button_callback)
glfw.set_cursor_pos_callback(window, cursor_pos_callback)
glfw.set_scroll_callback(window, scroll_callback)

def window_size_callback(window, width, height):
    viewport.width = width
    viewport.height = height

glfw.set_window_size_callback(window, window_size_callback)

plt.ion()
fig = plt.figure("DOF POS", figsize=(10, 8))
ax = fig.add_subplot(111)
dof_history = []
lines = []
num_dof_plot = 10
step_count = 0
last_fps_update = time.time()
fps = 0.0

last_dof_pos = None

def setup_plot(ax, num_dof_plot, joint_names):
    """初始化matplotlib曲线、标签、legend等设置，并返回新的 lines 列表"""
    ax.cla()  # 清空axes，防止legend重复
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('dof_pos', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    lines = []
    for i in range(num_dof_plot):
        line, = ax.plot([], [], label=joint_names[i] if i < len(joint_names) else f'dof {i}')
        lines.append(line)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=3, frameon=False, fontsize=12)
    fig.subplots_adjust(bottom=0.3)
    plt.pause(0.001)
    return lines

# 初始化matplotlib曲线
joint_names = [mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mujoco_model.njnt)]
joint_names = joint_names[1:num_dof_plot + 1]
lines = setup_plot(ax, num_dof_plot, joint_names)

while not glfw.window_should_close(window):
    if pause_event.is_set():
        # 正常获取新dof_pos
        motion_len = motion_lib.get_motion_length(motion_id).item()
        motion_time = time_step % motion_len
        motion_res = motion_lib.get_motion_state(torch.tensor([motion_id]), torch.tensor([motion_time]))
        root_pos, root_rot, dof_pos = motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"]
        last_dof_pos = dof_pos
    else:
        # 暂停时，保持上一个dof_pos
        dof_pos = last_dof_pos

    dof_pos = dof_pos[:, 1:]
    # Print dof_pos and mjdata for debugging
    # print("dof_pos:", dof_pos)
    # print("dof_pos shape:", dof_pos.shape)
    # print("mjdata.qpos shape:", data.qpos.shape)
    # print("mj_joint_names:", [mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mujoco_model.njnt)])
    # Set root position and orientation (protect z and normalize quaternion)
    root_pos_np = root_pos[0].cpu().numpy()
    root_pos_np[2] = max(root_pos_np[2], 0.0)
    # print("root_pos_np:", root_pos_np)
    data.qpos[0:3] = root_pos_np
    # Convert root_rot from xyzw (motion_lib) to wxyz (Mujoco)
    root_rot_np = root_rot[0].cpu().numpy()
    root_rot_np /= np.linalg.norm(root_rot_np) + 1e-8
    root_rot_mj = np.array([root_rot_np[3], root_rot_np[0], root_rot_np[1], root_rot_np[2]])
    # print("root_rot_np (xyzw):", root_rot_np)
    # print("root_rot_mj (wxyz):", root_rot_mj)
    data.qpos[3:7] = root_rot_mj
    # 不做映射，直接赋值到 qpos[7:]
    num_dof = min(dof_pos.shape[1], data.qpos.shape[0] - 7)
    for i in range(num_dof):
        val = dof_pos[0, i].item()
        val = np.clip(val, -np.pi, np.pi)
        data.qpos[i + 7] = val
    # data.qpos[15:17] = 0.0
    # print("mjdata.qpos:", data.qpos)
    mujoco.mj_step(mujoco_model, data)
    if step_count % 5 == 0:
        now = time.time()
        elapsed = now - last_fps_update
        if elapsed > 0:
            fps = 5 / elapsed
        last_fps_update = now
        mujoco.mjv_updateScene(mujoco_model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, con)
        fps_str = f'FPS: {fps:.1f}'
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            fps_str,
            "",
            con
        )
        glfw.swap_buffers(window)
    if pause_event.is_set():
        time_step += dt
    glfw.poll_events()
    
    
    # matplotlib数据更新逻辑
    if num_dof_plot is None or step_count == 0:
        dof_pos_np = dof_pos[0].cpu().numpy()
        num_dof_plot = len(dof_pos_np) if num_dof_plot is None else num_dof_plot
        joint_names = [mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mujoco_model.njnt)]
        joint_names = joint_names[1:num_dof_plot + 1]
        lines = setup_plot(ax, num_dof_plot, joint_names)
        dof_history.append(dof_pos_np.copy())
        hist_arr = np.array(dof_history)  # shape: [num_frames, dof]
        x_vals = np.arange(len(dof_history))
        for i in range(num_dof_plot):
            lines[i].set_data(x_vals, hist_arr[:, i])
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)
        pause_event.clear()
    elif pause_event.is_set():
        # 仅在运行状态下更新matplotlib数据
        if step_count % 20 == 0:
            dof_pos_np = dof_pos[0].cpu().numpy()
            dof_history.append(dof_pos_np.copy())
            hist_arr = np.array(dof_history)  # shape: [num_frames, dof]
            x_vals = np.arange(len(dof_history))
            for i in range(num_dof_plot):
                lines[i].set_data(x_vals, hist_arr[:, i])
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.001)
    step_count += 1

# Clean up
print("Done.")
glfw.terminate()
