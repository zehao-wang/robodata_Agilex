"""PIPER URDF loading utilities.

Loads the PIPER arm URDF with correct mesh path resolution,
and provides joint-order mapping between CAN protocol and URDF.
"""

from pathlib import Path

import numpy as np

# Path to assets relative to this file
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "piper_description"
_URDF_PATH = _ASSETS_DIR / "urdf" / "piper_description.urdf"

# URDF link name for the end-effector (fixed to link6)
PIPER_EEF_LINK_NAME = "gripper_base"

# The 6 revolute actuated joint names in URDF order
PIPER_ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def _resolve_mesh_path(fname: str, _dir: Path = _ASSETS_DIR) -> str:
    """Resolve ``package://piper_description/meshes/...`` to a local path."""
    prefix = "package://piper_description/"
    if fname.startswith(prefix):
        return str(_dir / fname[len(prefix):])
    return fname


def load_piper_urdf():
    """Load the PIPER URDF with mesh paths resolved to the local assets dir.

    Returns:
        yourdfpy.URDF instance.
    """
    import yourdfpy

    if not _URDF_PATH.exists():
        raise FileNotFoundError(
            f"PIPER URDF not found at {_URDF_PATH}. "
            "Please download from https://github.com/agilexrobotics/piper_ros"
        )
    urdf = yourdfpy.URDF.load(
        str(_URDF_PATH),
        filename_handler=_resolve_mesh_path,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    return urdf


def can_qpos_to_urdf_cfg(qpos_rad: np.ndarray) -> np.ndarray:
    """Map 6-element CAN joint-order qpos (radians) to URDF actuated-joint order.

    The PIPER CAN protocol and URDF both use joint1..joint6 in the same order,
    so this is currently an identity mapping.  The function exists so that any
    future re-ordering is handled in one place.

    Args:
        qpos_rad: (6,) joint positions in radians from CAN bus.

    Returns:
        (N,) array sized to the number of actuated joints in the URDF.
        For the standard PIPER URDF with gripper, N=8 (6 revolute + 2 prismatic).
        The gripper prismatic joints (joint7, joint8) are set to 0.
    """
    qpos_rad = np.asarray(qpos_rad, dtype=np.float64)
    assert qpos_rad.shape == (6,), f"Expected (6,), got {qpos_rad.shape}"
    # URDF actuated joints: joint1..6 (revolute) + joint7, joint8 (prismatic gripper)
    cfg = np.zeros(8, dtype=np.float64)
    cfg[:6] = qpos_rad
    return cfg


def can_qpos_to_urdf_cfg_with_gripper(qpos_rad: np.ndarray, gripper_m: float) -> np.ndarray:
    """Like ``can_qpos_to_urdf_cfg`` but also sets gripper finger positions.

    Args:
        qpos_rad: (6,) joint positions in radians.
        gripper_m: gripper opening width in meters (0 = closed, ~0.07 = full open).
            Each finger moves half the opening.

    Returns:
        (8,) array for URDF actuated joints.
    """
    cfg = can_qpos_to_urdf_cfg(qpos_rad)
    half = gripper_m / 2.0
    cfg[6] = np.clip(half, 0.0, 0.035)      # joint7 (0..0.035)
    cfg[7] = np.clip(-half, -0.035, 0.0)     # joint8 (-0.035..0)
    return cfg


def euler_deg_to_wxyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert Euler angles (degrees, XYZ extrinsic) to quaternion [w, x, y, z].

    This matches the PIPER SDK convention where rx/ry/rz are extrinsic XYZ Euler
    angles in degrees.
    """
    from scipy.spatial.transform import Rotation
    r = Rotation.from_euler("XYZ", [rx, ry, rz], degrees=True)
    # scipy returns [x,y,z,w]; we need [w,x,y,z]
    xyzw = r.as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def wxyz_to_euler_deg(wxyz: np.ndarray) -> tuple[float, float, float]:
    """Convert quaternion [w, x, y, z] to Euler angles (degrees, XYZ extrinsic).

    Returns:
        (rx, ry, rz) in degrees.
    """
    from scipy.spatial.transform import Rotation
    wxyz = np.asarray(wxyz, dtype=np.float64)
    xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    r = Rotation.from_quat(xyzw)
    angles = r.as_euler("XYZ", degrees=True)
    return float(angles[0]), float(angles[1]), float(angles[2])
