"""Viser-based interactive arm control app.

Provides:
- 3D arm visualization with URDF
- Manual EEF pose input
- Interactive drag control via transform handles
- pyroki IK solving and motion execution
"""

import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
import viser
from viser.extras import ViserUrdf

from robot.arm_controller import (
    create_piper,
    enable_arm,
    read_joints,
    move_joints_path,
    move_to_joint_waypoint,
    wait_motion_done,
    RAW_TO_RAD,
    RAD_TO_RAW,
    DEG_TO_RAW,
    RAW_TO_DEG,
)
from solver.pyroki_ik import PiperIKSolver
from utils.arm_visualizer import PIPER_FINGERTIP_OFFSET_M
from utils.urdf_loader import (
    load_piper_urdf,
    can_qpos_to_urdf_cfg_with_gripper,
    euler_deg_to_wxyz,
    wxyz_to_euler_deg,
)
from utils.world_frame import (
    pose_base_to_world,
    pose_world_to_base,
    add_world_frame_visual,
)


class ArmControlApp:
    """Viser web app for interactive PIPER arm control."""

    def __init__(
        self,
        piper=None,
        port: int = 8080,
        speed: int = 50,
        demo_mode: bool = False,
        world_config: dict | None = None,
    ):
        self._piper = piper
        self._port = port
        self._speed = speed
        self._ik_solver = PiperIKSolver()
        self._executing = False
        self._last_ik_cfg = None
        self._demo_mode = demo_mode
        self._world_config = world_config
        if world_config is not None:
            self._T_base_from_world = np.asarray(
                world_config["T_base_from_world"], dtype=np.float64
            )
            self._T_world_from_base = np.asarray(
                world_config["T_world_from_base"], dtype=np.float64
            )
        else:
            self._T_base_from_world = None
            self._T_world_from_base = None

    def run(self):
        """Start the viser server and run the main loop."""
        server = viser.ViserServer(port=self._port)
        self._server = server

        # --- 3D Scene ---
        server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        urdf = load_piper_urdf()
        self._urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

        # World frame calibration visualization
        if self._world_config is not None:
            add_world_frame_visual(server, self._world_config)

        # Initial IK target handle position = fingertip center at home.
        init_wxyz = euler_deg_to_wxyz(0, 85, 0)
        _R_home = Rotation.from_quat([init_wxyz[1], init_wxyz[2], init_wxyz[3], init_wxyz[0]]).as_matrix()
        _gb_home = np.array([0.057, 0.0, 0.215])
        _tip_home = _gb_home + _R_home[:, 2] * PIPER_FINGERTIP_OFFSET_M
        init_pos = tuple(_tip_home)

        self._ik_target = server.scene.add_transform_controls(
            "/ik_target",
            scale=0.15,
            position=init_pos,
            wxyz=tuple(init_wxyz),
        )

        # --- Sidebar GUI ---
        pose_label = (
            "EEF Pose - World Frame" if self._world_config is not None
            else "EEF Pose (meters/degrees)"
        )
        if self._T_world_from_base is not None:
            init_pos_disp, init_wxyz_disp = pose_base_to_world(
                _tip_home, init_wxyz, self._T_world_from_base
            )
            init_rx, init_ry, init_rz = wxyz_to_euler_deg(init_wxyz_disp)
        else:
            init_pos_disp, init_wxyz_disp = _tip_home, init_wxyz
            init_rx, init_ry, init_rz = 0.0, 85.0, 0.0
        with server.gui.add_folder(pose_label):
            self._x_input = server.gui.add_number("X (m)", initial_value=float(init_pos_disp[0]), step=0.01)
            self._y_input = server.gui.add_number("Y (m)", initial_value=float(init_pos_disp[1]), step=0.01)
            self._z_input = server.gui.add_number("Z (m)", initial_value=float(init_pos_disp[2]), step=0.01)
            self._rx_input = server.gui.add_number("RX (deg)", initial_value=float(init_rx), step=1.0)
            self._ry_input = server.gui.add_number("RY (deg)", initial_value=float(init_ry), step=1.0)
            self._rz_input = server.gui.add_number("RZ (deg)", initial_value=float(init_rz), step=1.0)

        # Track last handle position to detect drag changes
        self._last_handle_pos = np.array(init_pos)
        self._last_handle_wxyz = np.array(init_wxyz)
        self._updating_from_input = False  # Prevent feedback loops

        with server.gui.add_folder("Motion"):
            self._speed_slider = server.gui.add_slider(
                "Speed %", min=1, max=100, step=1, initial_value=self._speed
            )
            self._execute_btn = server.gui.add_button("Execute Motion")
            self._home_btn = server.gui.add_button("Go Home")

        with server.gui.add_folder("Obstacles"):
            self._add_box_btn = server.gui.add_button("Add Test Box")
            self._clear_boxes_btn = server.gui.add_button("Clear Boxes")
            self._obs_status = server.gui.add_markdown("**Obstacles:** Ground plane")

        self._obstacle_meshes = []  # Track viser mesh handles for boxes
        self._add_box_btn.on_click(self._on_add_box)
        self._clear_boxes_btn.on_click(self._on_clear_boxes)

        # Show default ground plane
        self._init_ground_visual()

        with server.gui.add_folder("Status"):
            self._ik_status = server.gui.add_markdown("**IK:** Ready")
            if self._demo_mode:
                arm_status_text = "**Arm:** Demo Mode (preview only)"
            elif self._piper:
                arm_status_text = "**Arm:** Connected"
            else:
                arm_status_text = "**Arm:** Not connected"
            self._arm_status = server.gui.add_markdown(arm_status_text)

        # Event handlers
        self._execute_btn.on_click(self._on_execute_click)
        self._home_btn.on_click(self._on_home_click)

        print(f"[ArmControl] Server started at http://localhost:{self._port}")
        print("Press Ctrl+C to stop.\n")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[ArmControl] Shutting down...")

    def _tip_to_gripper_base(self, tip_pos: np.ndarray, tip_wxyz: np.ndarray) -> np.ndarray:
        """Convert fingertip target position → gripper_base target position.

        The IK handle represents the fingertip center. IK solver targets gripper_base
        (link6), so we subtract the fixed fingertip offset along the approach z-axis.
        """
        xyzw = np.array([tip_wxyz[1], tip_wxyz[2], tip_wxyz[3], tip_wxyz[0]])
        R = Rotation.from_quat(xyzw).as_matrix()
        return tip_pos - R[:, 2] * PIPER_FINGERTIP_OFFSET_M

    def _main_loop(self):
        """Main loop: synchronize 3D handle and input fields bidirectionally.

        When world frame is active, the input fields show world-frame coordinates.
        The 3D handle and IK solver always work in base frame.
        """
        while True:
            # Get current handle position (always in base frame)
            handle_pos_base = np.array(self._ik_target.position)
            handle_wxyz_base = np.array(self._ik_target.wxyz)

            # Get current input values (in world frame if active, else base)
            input_pos_display = np.array([
                self._x_input.value,
                self._y_input.value,
                self._z_input.value,
            ])
            input_wxyz_display = euler_deg_to_wxyz(
                self._rx_input.value,
                self._ry_input.value,
                self._rz_input.value,
            )

            # Convert display input → base frame for comparison
            if self._T_base_from_world is not None:
                input_pos_base, input_wxyz_base = pose_world_to_base(
                    input_pos_display, input_wxyz_display, self._T_base_from_world
                )
            else:
                input_pos_base = input_pos_display
                input_wxyz_base = input_wxyz_display

            # Check if handle was dragged (position changed from last frame)
            handle_moved = (
                np.linalg.norm(handle_pos_base - self._last_handle_pos) > 1e-6 or
                np.linalg.norm(handle_wxyz_base - self._last_handle_wxyz) > 1e-6
            )

            # Check if input fields changed (compare base-frame versions)
            input_changed = (
                np.linalg.norm(input_pos_base - self._last_handle_pos) > 1e-6 or
                np.linalg.norm(input_wxyz_base - self._last_handle_wxyz) > 1e-6
            )

            if handle_moved:
                # Handle was dragged (base frame) → update input fields
                if self._T_world_from_base is not None:
                    # Convert base → world for display
                    disp_pos, disp_wxyz = pose_base_to_world(
                        handle_pos_base, handle_wxyz_base, self._T_world_from_base
                    )
                else:
                    disp_pos, disp_wxyz = handle_pos_base, handle_wxyz_base

                self._x_input.value = float(disp_pos[0])
                self._y_input.value = float(disp_pos[1])
                self._z_input.value = float(disp_pos[2])
                rx, ry, rz = wxyz_to_euler_deg(disp_wxyz)
                self._rx_input.value = rx
                self._ry_input.value = ry
                self._rz_input.value = rz
                # IK uses base frame
                pos, wxyz = handle_pos_base, handle_wxyz_base
                self._last_handle_pos = handle_pos_base.copy()
                self._last_handle_wxyz = handle_wxyz_base.copy()

            elif input_changed:
                # Input fields changed → update handle (base frame)
                self._ik_target.position = tuple(input_pos_base)
                self._ik_target.wxyz = tuple(input_wxyz_base)
                # IK uses base frame
                pos, wxyz = input_pos_base, input_wxyz_base
                self._last_handle_pos = input_pos_base.copy()
                self._last_handle_wxyz = input_wxyz_base.copy()

            else:
                # No change, use current base-frame position
                pos, wxyz = handle_pos_base, handle_wxyz_base

            # Solve IK and update arm visualization (always base frame).
            # Handle position = fingertip target; convert to gripper_base for IK.
            gripper_base_pos = self._tip_to_gripper_base(pos, wxyz)
            cfg = self._ik_solver.solve(gripper_base_pos, wxyz)
            if cfg is not None:
                self._last_ik_cfg = cfg
                self._urdf_vis.update_cfg(cfg)
                self._ik_status.content = "**IK:** Solution found"
            else:
                self._ik_status.content = "**IK:** No solution"

            time.sleep(0.03)  # ~30 FPS

    def _on_execute_click(self, _event):
        """Execute motion to the IK solution."""
        if self._executing:
            print("[ArmControl] Already executing motion.")
            return
        if self._last_ik_cfg is None:
            print("[ArmControl] No valid IK solution to execute.")
            self._arm_status.content = "**Arm:** No IK solution"
            return
        if self._piper is None:
            if self._demo_mode:
                # Demo mode: just show the IK solution info
                joints_deg = np.rad2deg(self._last_ik_cfg[:6])
                print(f"[ArmControl] Demo mode - IK solution (deg): {joints_deg}")
                self._arm_status.content = "**Arm:** Demo - IK shown above"
            else:
                print("[ArmControl] No arm connected.")
                self._arm_status.content = "**Arm:** Not connected"
            return

        # Run motion in background thread
        thread = threading.Thread(target=self._execute_motion, daemon=True)
        thread.start()

    def _execute_motion(self):
        """Background: plan and execute collision-free trajectory."""
        self._executing = True
        self._arm_status.content = "**Arm:** Planning trajectory..."

        try:
            # Get current joint configuration
            cur_j_raw = read_joints(self._piper)
            cur_cfg = np.array([v * RAW_TO_RAD for v in cur_j_raw])
            # Pad to 8 joints (add gripper joints as zeros)
            cur_cfg_full = np.zeros(8)
            cur_cfg_full[:6] = cur_cfg

            # Get target EEF pose (handle = fingertip; convert to gripper_base for IK)
            tip_pos = np.array(self._ik_target.position)
            target_wxyz = np.array(self._ik_target.wxyz)
            target_pos = self._tip_to_gripper_base(tip_pos, target_wxyz)

            speed = self._speed_slider.value

            # Plan trajectory using pyroki trajopt
            print(f"[ArmControl] Planning collision-free trajectory...")
            self._arm_status.content = "**Arm:** Optimizing trajectory..."
            trajectory = self._ik_solver.plan_trajectory(
                start_cfg=cur_cfg_full,
                target_position=target_pos,
                target_wxyz=target_wxyz,
                timesteps=30,
                dt=0.05,
            )

            if trajectory is None:
                print("[ArmControl] Trajectory planning failed!")
                self._arm_status.content = "**Arm:** Planning failed"
                self._executing = False
                return

            print(f"[ArmControl] Trajectory planned: {trajectory.shape[0]} waypoints")
            self._arm_status.content = "**Arm:** Executing..."

            # Execute forward trajectory
            waypoints_raw = []
            for i, cfg in enumerate(trajectory):
                # Convert to raw units (only 6 arm joints)
                wp_raw = [int(round(cfg[j] * RAD_TO_RAW)) for j in range(6)]
                waypoints_raw.append(wp_raw)

                print(f"\r[ArmControl] Forward: waypoint {i+1}/{len(trajectory)}", end="", flush=True)
                ok = move_to_joint_waypoint(self._piper, wp_raw, speed, timeout=5.0, tol=5000)
                if not ok:
                    print(f"\n[ArmControl] Waypoint {i+1} failed, continuing...")

            print("\n[ArmControl] Target reached.")
            wait_motion_done(self._piper, timeout=3.0)

            # Return via reverse trajectory
            print("[ArmControl] Returning via reverse path...")
            self._arm_status.content = "**Arm:** Returning..."
            self._piper.MotionCtrl_2(0x00, 0x00, 0, 0x00)
            time.sleep(0.3)

            reversed_wps = list(reversed(waypoints_raw))
            for i, wp in enumerate(reversed_wps):
                print(f"\r[ArmControl] Return: waypoint {i+1}/{len(reversed_wps)}", end="", flush=True)
                ok = move_to_joint_waypoint(self._piper, wp, speed, timeout=5.0, tol=5000)
                if not ok:
                    print(f"\n[ArmControl] Return waypoint {i+1} failed, continuing...")

            print("\n[ArmControl] Motion complete.")
            self._arm_status.content = "**Arm:** Ready"

        except Exception as e:
            print(f"[ArmControl] Motion error: {e}")
            self._arm_status.content = f"**Arm:** Error: {e}"

        self._executing = False

    def _init_ground_visual(self):
        """Show ground plane visual (already added to collision by default)."""
        import trimesh
        ground_mesh = trimesh.creation.box(extents=[2.0, 2.0, 0.01])
        ground_mesh.apply_translation([0, 0, -0.005])
        ground_mesh.visual.face_colors = [100, 100, 100, 80]
        self._server.scene.add_mesh_trimesh("/ground_plane", ground_mesh)

    def _on_add_box(self, _event):
        """Add a test box obstacle in front of the robot."""
        import trimesh
        extent = np.array([0.1, 0.1, 0.2])
        position = np.array([0.15, 0.0, 0.1])
        self._ik_solver.add_box_obstacle(extent, position, name="test_box")
        # Visualize
        box_mesh = trimesh.creation.box(extents=extent)
        box_mesh.apply_translation(position)
        box_mesh.visual.face_colors = [255, 100, 100, 150]
        idx = len(self._obstacle_meshes)
        handle = self._server.scene.add_mesh_trimesh(f"/obstacles/box_{idx}", box_mesh)
        self._obstacle_meshes.append(handle)
        self._update_obstacle_status()

    def _on_clear_boxes(self, _event):
        """Remove box obstacles (keep ground plane)."""
        # Clear all world collisions and re-add ground
        self._ik_solver.clear_obstacles()
        self._ik_solver.add_ground_plane(height=0.0)
        # Remove box visuals
        for handle in self._obstacle_meshes:
            handle.remove()
        self._obstacle_meshes.clear()
        self._update_obstacle_status()

    def _update_obstacle_status(self):
        """Update obstacle count in GUI."""
        box_count = len(self._obstacle_meshes)
        if box_count == 0:
            self._obs_status.content = "**Obstacles:** Ground plane"
        else:
            self._obs_status.content = f"**Obstacles:** Ground + {box_count} box(es)"

    def _on_home_click(self, _event):
        """Go to home position."""
        home_wxyz = euler_deg_to_wxyz(0, 85, 0)
        # Fingertip home = gripper_base home + offset along approach z-axis
        _R = Rotation.from_quat([home_wxyz[1], home_wxyz[2], home_wxyz[3], home_wxyz[0]]).as_matrix()
        _gb_home = np.array([0.057, 0.0, 0.215])
        home_pos = _gb_home + _R[:, 2] * PIPER_FINGERTIP_OFFSET_M

        # Update input fields and handle. Inputs follow world frame if loaded.
        if self._T_world_from_base is not None:
            home_pos_disp, home_wxyz_disp = pose_base_to_world(
                home_pos, home_wxyz, self._T_world_from_base
            )
            home_rx, home_ry, home_rz = wxyz_to_euler_deg(home_wxyz_disp)
        else:
            home_pos_disp = home_pos
            home_rx, home_ry, home_rz = 0.0, 85.0, 0.0

        self._x_input.value = float(home_pos_disp[0])
        self._y_input.value = float(home_pos_disp[1])
        self._z_input.value = float(home_pos_disp[2])
        self._rx_input.value = float(home_rx)
        self._ry_input.value = float(home_ry)
        self._rz_input.value = float(home_rz)
        self._ik_target.position = tuple(home_pos)
        self._ik_target.wxyz = tuple(home_wxyz)
        self._last_handle_pos = home_pos.copy()
        self._last_handle_wxyz = home_wxyz.copy()

        # If arm connected, also execute motion
        if self._executing:
            print("[ArmControl] Already executing motion.")
            return

        if self._piper is None:
            if self._demo_mode:
                print("[ArmControl] Demo mode - moved to home position (preview)")
                self._arm_status.content = "**Arm:** Demo - at home"
            return

        home_gripper_base_pos = self._tip_to_gripper_base(home_pos, home_wxyz)
        cfg = self._ik_solver.solve(home_gripper_base_pos, home_wxyz)
        if cfg is not None:
            self._last_ik_cfg = cfg
            thread = threading.Thread(target=self._execute_motion, daemon=True)
            thread.start()
        else:
            print("[ArmControl] Could not solve IK for home position.")
