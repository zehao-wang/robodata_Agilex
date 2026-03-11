"""3D robot arm visualizer using matplotlib offscreen rendering.

Computes forward kinematics from PIPER DH parameters and renders
joint positions as a 3D skeleton into a BGR numpy image suitable
for embedding in an OpenCV window.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


# PIPER DH parameters (offset mode 0x01, from piper_sdk piper_fk.py)
_A = [0.0, 0.0, 285.03, -21.98, 0.0, 0.0]           # link length (mm)
_ALPHA = [0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2]  # twist
_THETA_OFFSET = [0.0, -172.22 * np.pi / 180, -102.78 * np.pi / 180,
                 0.0, 0.0, 0.0]                        # joint angle offset
_D = [123.0, 0.0, 0.0, 250.75, 0.0, 91.0]             # link offset (mm)


def _dh_matrix(alpha, a, theta, d):
    """Compute 4x4 DH transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,       -st,       0,      a],
        [st * ca,   ct * ca,  -sa,   -sa * d],
        [st * sa,   ct * sa,   ca,    ca * d],
        [0,         0,         0,     1],
    ])


# Distance from link6 (gripper_base) origin to fingertip center, along link6 z-axis.
# Derived from URDF: finger joint roots at z=135.8mm, finger CoM at z=86.6mm,
# fingertip (CoM + 49.2mm extension) at z≈37mm from link6.
PIPER_FINGERTIP_OFFSET_M = 0.037


def fingertip_center_from_T_ee(T_ee: np.ndarray) -> np.ndarray:
    """Return fingertip center position in meters from DH frame-6 transform (in mm).

    The fingertip center is PIPER_FINGERTIP_OFFSET_M ahead of the gripper_base
    (link6) origin along the link6 z-axis (approach direction).
    """
    pos_m = T_ee[:3, 3] / 1000.0
    z_axis = T_ee[:3, 2]
    return pos_m + z_axis * PIPER_FINGERTIP_OFFSET_M


def forward_kinematics(joint_angles):
    """Compute 3D positions of each joint given 6 joint angles (radians).

    Returns:
        positions: (7, 3) array — base + 6 joint positions, in mm.
        T_ee: (4, 4) array — end-effector transformation matrix.
    """
    positions = [np.array([0.0, 0.0, 0.0])]
    T = np.eye(4)
    for i in range(6):
        theta = joint_angles[i] + _THETA_OFFSET[i]
        Ti = _dh_matrix(_ALPHA[i], _A[i], theta, _D[i])
        T = T @ Ti
        positions.append(T[:3, 3].copy())
    return np.array(positions), T.copy()


# Link colors: base->j1 blue, j1->j2 blue, j2->j3 green, j3->j4 green,
#              j4->j5 red, j5->j6 red
_LINK_COLORS = ["#2196F3", "#2196F3", "#4CAF50", "#4CAF50", "#F44336", "#F44336"]
_JOINT_COLORS = ["#333333"] + _LINK_COLORS  # base is dark gray


class ArmVisualizer:
    """Renders a 3D arm skeleton to a BGR numpy image."""

    def __init__(self, size=480, render_scale=2):
        self._size = size
        self._render_size = size * render_scale
        dpi = 150
        figsize = self._render_size / dpi
        self._fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
        self._canvas = FigureCanvasAgg(self._fig)
        self._ax = self._fig.add_subplot(111, projection="3d")
        # Push the 3D axes to fill nearly the entire figure
        self._ax.set_position([-.15, -.05, 1.3, 1.15])

    def render(self, joint_angles, gripper_width=0.0) -> np.ndarray:
        """Render the arm and return a (size, size, 3) BGR uint8 image.

        Args:
            joint_angles: 6 joint angles in radians.
            gripper_width: Gripper opening width in meters.
        """
        positions, T_ee = forward_kinematics(joint_angles)

        ax = self._ax
        ax.cla()

        # Draw links
        for i in range(6):
            p0, p1 = positions[i], positions[i + 1]
            ax.plot(
                [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color=_LINK_COLORS[i], linewidth=3, solid_capstyle="round",
            )

        # Draw joints
        for i, pos in enumerate(positions):
            ax.scatter(
                pos[0], pos[1], pos[2],
                color=_JOINT_COLORS[i], s=40, zorder=5, depthshade=False,
            )

        # --- End-effector center ---
        ee = positions[-1]
        # EE approach direction (z-axis of last frame) — extend a short line
        ee_z = T_ee[:3, 2]  # approach direction
        ee_tip = ee + ee_z * 30  # 30mm ahead
        ax.plot([ee[0], ee_tip[0]], [ee[1], ee_tip[1]], [ee[2], ee_tip[2]],
                color="#FF9800", linewidth=2, linestyle="--")
        ax.scatter(ee[0], ee[1], ee[2], color="#FF9800", s=70,
                   marker="o", zorder=6, depthshade=False, label="EE center")

        # --- Gripper fingers ---
        # Gripper opens along the EE y-axis; each finger is half the width offset
        ee_y = T_ee[:3, 1]  # lateral direction
        half_w = gripper_width * 1000.0 / 2.0  # meters -> mm, half per side
        half_w = max(half_w, 2.0)  # minimum visible gap
        finger_len = 40.0  # mm, visual finger length along approach axis

        for sign in (+1, -1):
            base_pt = ee + ee_y * sign * half_w
            tip_pt = base_pt + ee_z * finger_len
            ax.plot([base_pt[0], tip_pt[0]], [base_pt[1], tip_pt[1]],
                    [base_pt[2], tip_pt[2]],
                    color="#9C27B0", linewidth=3, solid_capstyle="round")
            ax.scatter(tip_pt[0], tip_pt[1], tip_pt[2],
                       color="#9C27B0", s=25, zorder=6, depthshade=False)

        # Fixed view
        ax.view_init(elev=25, azim=-60)
        ax.set_xlabel("X", fontsize=9, labelpad=2)
        ax.set_ylabel("Y", fontsize=9, labelpad=2)
        ax.set_zlabel("Z", fontsize=9, labelpad=2)
        ax.tick_params(labelsize=7, pad=0)

        # Tighter axis limits so the arm fills the viewport
        ax.set_xlim(-350, 350)
        ax.set_ylim(-350, 350)
        ax.set_zlim(-50, 550)

        ax.set_title("Arm 3D", fontsize=11, pad=0)

        # Render to numpy array at high resolution, then downscale
        self._canvas.draw()
        buf = np.frombuffer(self._canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(self._render_size, self._render_size, 4)
        # RGBA -> BGR
        bgr = buf[:, :, 2::-1].copy()
        if self._render_size != self._size:
            import cv2
            bgr = cv2.resize(bgr, (self._size, self._size),
                              interpolation=cv2.INTER_AREA)
        return bgr
