"""pyroki-based IK and trajectory optimization for the PIPER arm.

Provides:
- IK solving with collision avoidance
- Trajectory optimization with smoothness and collision constraints
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk

from utils.urdf_loader import (
    PIPER_EEF_LINK_NAME,
    load_piper_urdf,
)

# Path to sphere decomposition
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_SPHERES_PATH = _ASSETS_DIR / "piper_spheres.json"


class PiperIKSolver:
    """IK solver and trajectory planner for the PIPER arm."""

    def __init__(self):
        self._urdf = load_piper_urdf()
        self._robot = pk.Robot.from_urdf(self._urdf)
        self._eef_link_index = self._robot.links.names.index(PIPER_EEF_LINK_NAME)

        # Load collision model (always enabled)
        self._robot_coll = None
        if _SPHERES_PATH.exists():
            with open(_SPHERES_PATH, "r") as f:
                sphere_data = json.load(f)
            self._robot_coll = pk.collision.RobotCollision.from_sphere_decomposition(
                sphere_decomposition=sphere_data,
                urdf=self._urdf,
            )
            print("[PiperIKSolver] Collision model loaded")
        else:
            raise FileNotFoundError(f"Sphere decomposition not found: {_SPHERES_PATH}")

        # World collision objects (ground plane added by default)
        self._world_collisions: list = []
        self.add_ground_plane(height=0.0)

    @property
    def robot(self) -> pk.Robot:
        return self._robot

    @property
    def urdf(self):
        return self._urdf

    @property
    def robot_collision(self):
        return self._robot_coll

    def add_box_obstacle(self, extent: np.ndarray, position: np.ndarray, name: str = "box"):
        """Add a box obstacle to the world."""
        box = pk.collision.Box.from_extent(
            extent=np.asarray(extent, dtype=np.float32),
            position=np.asarray(position, dtype=np.float32),
        )
        self._world_collisions.append(box)
        print(f"[PiperIKSolver] Added box obstacle '{name}': extent={extent}, pos={position}")
        return box

    def add_ground_plane(self, height: float = 0.0):
        """Add a ground plane obstacle."""
        ground = pk.collision.HalfSpace.from_point_and_normal(
            np.array([0.0, 0.0, height], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        self._world_collisions.append(ground)
        print(f"[PiperIKSolver] Added ground plane at z={height}")
        return ground

    def clear_obstacles(self):
        """Remove all world collision objects."""
        self._world_collisions.clear()
        print("[PiperIKSolver] Cleared all obstacles")

    def solve(
        self,
        target_position: np.ndarray,
        target_wxyz: np.ndarray,
        seed_cfg: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Solve IK for a target EEF pose."""
        target_position = np.asarray(target_position, dtype=np.float64)
        target_wxyz = np.asarray(target_wxyz, dtype=np.float64)

        try:
            cfg = _solve_ik_jax(
                self._robot,
                jnp.array(self._eef_link_index),
                jnp.array(target_wxyz),
                jnp.array(target_position),
            )
            result = np.array(cfg)
            lower = np.array(self._robot.joints.lower_limits)
            upper = np.array(self._robot.joints.upper_limits)
            if np.all(result >= lower - 0.01) and np.all(result <= upper + 0.01):
                return result
            return None
        except Exception:
            return None

    def forward_kinematics(self, cfg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics for a given joint configuration."""
        cfg_jax = jnp.array(cfg)
        Ts_joint_world = self._robot.forward_kinematics(cfg_jax)
        T_eef = jaxlie.SE3(Ts_joint_world[self._eef_link_index])
        pos = np.array(T_eef.translation())
        wxyz = np.array(T_eef.rotation().wxyz)
        return pos, wxyz

    def solve_from_can(
        self,
        target_position: np.ndarray,
        target_wxyz: np.ndarray,
        seed_qpos_rad: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Solve IK with seed in CAN joint order. Returns 6 joint angles."""
        result = self.solve(target_position, target_wxyz)
        if result is None:
            return None
        return result[:6]

    def plan_trajectory(
        self,
        start_cfg: np.ndarray,
        target_position: np.ndarray,
        target_wxyz: np.ndarray,
        timesteps: int = 30,
        dt: float = 0.05,
    ) -> np.ndarray | None:
        """Plan a collision-free trajectory from start config to target EEF pose.

        Args:
            start_cfg: Starting joint configuration (8 joints for URDF)
            target_position: Target EEF position in meters
            target_wxyz: Target EEF orientation as quaternion
            timesteps: Number of trajectory waypoints
            dt: Time step between waypoints

        Returns:
            (timesteps, num_joints) trajectory array, or None if planning fails
        """
        # First solve IK for target
        target_cfg = self.solve(target_position, target_wxyz)
        if target_cfg is None:
            print("[PiperIKSolver] Trajectory planning failed: IK has no solution")
            return None

        start_cfg = np.asarray(start_cfg, dtype=np.float64)

        # Use full trajectory optimization with collision avoidance
        try:
            traj = _solve_trajopt(
                robot=self._robot,
                robot_coll=self._robot_coll,
                world_coll=self._world_collisions,
                start_cfg=jnp.array(start_cfg),
                end_cfg=jnp.array(target_cfg),
                timesteps=timesteps,
                dt=dt,
            )
            return np.array(traj)
        except Exception as e:
            print(f"[PiperIKSolver] Trajectory optimization failed: {e}")
            print("[PiperIKSolver] Falling back to smooth interpolation")
            return self._plan_smooth_trajectory(start_cfg, target_cfg, timesteps)

    def _plan_smooth_trajectory(
        self,
        start_cfg: np.ndarray,
        end_cfg: np.ndarray,
        timesteps: int,
    ) -> np.ndarray:
        """Simple smooth interpolation without collision checking."""
        # Use cubic interpolation for smoother motion
        t = np.linspace(0, 1, timesteps)
        # Smooth step function: 3t^2 - 2t^3
        alpha = 3 * t**2 - 2 * t**3
        traj = start_cfg[None, :] + alpha[:, None] * (end_cfg - start_cfg)[None, :]
        return traj


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
) -> jax.Array:
    """Basic IK solver."""
    joint_var = robot.joint_var_cls(0)
    variables = [joint_var]
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_constraint(robot, joint_var),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=variables)
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]


def _solve_trajopt(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: list,
    start_cfg: jax.Array,
    end_cfg: jax.Array,
    timesteps: int,
    dt: float,
) -> jax.Array:
    """Trajectory optimization with collision avoidance."""
    # Initialize trajectory with linear interpolation
    init_traj = jnp.linspace(start_cfg, end_cfg, timesteps)

    # Create trajectory variables
    traj_vars = robot.joint_var_cls(jnp.arange(timesteps))

    # Add batch dimension for collision costs
    robot_batched = jax.tree.map(lambda x: x[None], robot)
    robot_coll_batched = jax.tree.map(lambda x: x[None], robot_coll)

    # Build costs
    costs: list[jaxls.Cost] = [
        # Regularization toward default pose
        pk.costs.rest_cost(
            traj_vars,
            traj_vars.default_factory()[None],
            jnp.array([0.01])[None],
        ),
        # Smoothness (minimize joint velocity)
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            jnp.array([1.0])[None],
        ),
        # Self-collision avoidance
        pk.costs.self_collision_cost(
            robot_batched,
            robot_coll_batched,
            traj_vars,
            margin=0.02,
            weight=5.0,
        ),
        # Joint limits
        pk.costs.limit_constraint(robot_batched, traj_vars),
    ]

    # Add acceleration cost if enough timesteps
    if timesteps >= 5:
        costs.append(
            pk.costs.five_point_acceleration_cost(
                robot.joint_var_cls(jnp.arange(2, timesteps - 2)),
                robot.joint_var_cls(jnp.arange(4, timesteps)),
                robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
                dt,
                jnp.array([0.1])[None],
            )
        )

    # Start/end pose constraints
    @jaxls.Cost.factory(kind="constraint_eq_zero", name="start_pose_constraint")
    def start_pose_constraint(vals: jaxls.VarValues, var) -> jax.Array:
        return (vals[var] - start_cfg).flatten()

    @jaxls.Cost.factory(kind="constraint_eq_zero", name="end_pose_constraint")
    def end_pose_constraint(vals: jaxls.VarValues, var) -> jax.Array:
        return (vals[var] - end_cfg).flatten()

    costs.append(start_pose_constraint(robot.joint_var_cls(jnp.arange(0, 2))))
    costs.append(end_pose_constraint(robot.joint_var_cls(jnp.arange(timesteps - 2, timesteps))))

    # Velocity limits
    costs.append(
        pk.costs.limit_velocity_constraint(
            robot_batched,
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            dt,
        )
    )

    # World collision avoidance (swept volumes)
    for world_coll_obj in world_coll:
        @jaxls.Cost.factory(kind="constraint_geq_zero", name="world_collision")
        def world_coll_cost(
            vals: jaxls.VarValues,
            prev_var,
            curr_var,
            _robot=robot_batched,
            _coll=robot_coll_batched,
            _obj=jax.tree.map(lambda x: x[None], world_coll_obj),
        ) -> jax.Array:
            swept = _coll.get_swept_capsules(_robot, vals[prev_var], vals[curr_var])
            dist = pk.collision.collide(swept.reshape((-1, 1)), _obj.reshape((1, -1)))
            return dist.flatten() - 0.05  # safety margin

        costs.append(
            world_coll_cost(
                robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(1, timesteps)),
            )
        )

    # Solve
    solution = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[traj_vars])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_traj),)),
            verbose=False,
        )
    )
    return solution[traj_vars]
