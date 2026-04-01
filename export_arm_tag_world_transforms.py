#!/usr/bin/env python3
"""Export calibrated arm/tag transforms in the collect_viser world frame."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from utils.apriltag_reconstruction import load_static_reconstruction_model
from utils.arm_world_calibration import load_arm_world_calibration


def _as_list(array: np.ndarray) -> list:
    return np.asarray(array, dtype=np.float64).tolist()


def _matrix_to_wxyz(rotation: np.ndarray) -> list[float]:
    xyzw = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_quat()
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = np.asarray(transform[:3, :3], dtype=np.float64)
    translation = np.asarray(transform[:3, 3], dtype=np.float64)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Encountered a degenerate tag axis while building a transform.")
    return vector / norm


def _build_transform(rotation_columns: tuple[np.ndarray, np.ndarray, np.ndarray], origin: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.column_stack(rotation_columns)
    transform[:3, 3] = np.asarray(origin, dtype=np.float64).reshape(3)
    return transform


def _tag_axes_from_corners(corners_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corners_world = np.asarray(corners_world, dtype=np.float64).reshape(4, 3)
    corner_2 = corners_world[2]
    x_axis = _normalize(corner_2 - corners_world[3])
    y_seed = corner_2 - corners_world[1]
    z_axis = _normalize(np.cross(x_axis, y_seed))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return x_axis, y_axis, z_axis


def _tag_payload(corners_world: np.ndarray, sample_counts: dict[str, int]) -> dict:
    corners_world = np.asarray(corners_world, dtype=np.float64).reshape(4, 3)
    x_axis, y_axis, z_axis = _tag_axes_from_corners(corners_world)
    corner_2_origin = corners_world[2]
    center_origin = corners_world.mean(axis=0)

    T_world_from_tag_corner2 = _build_transform((x_axis, y_axis, z_axis), corner_2_origin)
    T_world_from_tag_center = _build_transform((x_axis, y_axis, z_axis), center_origin)

    edge_lengths = [
        float(np.linalg.norm(corners_world[(i + 1) % 4] - corners_world[i]))
        for i in range(4)
    ]

    return {
        "frame_convention": {
            "corner2_frame_origin": "tag corner 2",
            "center_frame_origin": "mean of 4 tag corners",
            "x_axis": "corner 3 -> corner 2",
            "y_axis": "corner 1 -> corner 2 after orthogonalization",
            "z_axis": "x cross y (right-handed)",
        },
        "corners_world_m": {
            str(corner_idx): _as_list(point_world)
            for corner_idx, point_world in enumerate(corners_world)
        },
        "center_world_m": _as_list(center_origin),
        "edge_lengths_m": edge_lengths,
        "tag_size_estimate_m": float(np.mean(edge_lengths)),
        "sample_counts_by_corner": sample_counts,
        "T_world_from_tag_corner2": _as_list(T_world_from_tag_corner2),
        "T_tag_corner2_from_world": _as_list(_invert_transform(T_world_from_tag_corner2)),
        "quaternion_wxyz_world_from_tag_corner2": _matrix_to_wxyz(T_world_from_tag_corner2[:3, :3]),
        "T_world_from_tag_center": _as_list(T_world_from_tag_center),
        "T_tag_center_from_world": _as_list(_invert_transform(T_world_from_tag_center)),
        "quaternion_wxyz_world_from_tag_center": _matrix_to_wxyz(T_world_from_tag_center[:3, :3]),
    }


def export_transforms(
    arm_calibration_path: Path,
    model_dir: Path,
    output_path: Path,
) -> Path:
    arm_result = load_arm_world_calibration(arm_calibration_path)
    if arm_result is None:
        raise FileNotFoundError(f"Arm calibration file not found: {arm_calibration_path}")

    model = load_static_reconstruction_model(model_dir)
    raw_payload = json.loads(model.json_path.read_text(encoding="utf-8"))
    raw_tags_payload = raw_payload.get("tags", {})
    raw_tag_size_m = float(raw_payload.get("tag_size_m", 0.0))

    tag_payloads = {}
    for tag_id in sorted(model.corner_points_by_tag):
        corners_world = model.corner_points_by_tag[tag_id]
        sample_counts = {str(corner_idx): 0 for corner_idx in range(4)}
        raw_tag_payload = raw_tags_payload.get(str(tag_id), {})
        for corner_idx in range(4):
            sample_counts[str(corner_idx)] = int(raw_tag_payload.get(str(corner_idx), {}).get("sample_count", 0))
        tag_payload = _tag_payload(corners_world, sample_counts)
        tag_payload["tag_size_m"] = raw_tag_size_m
        tag_payloads[str(tag_id)] = tag_payload

    transforms_only = {
        "arm_base_to_world": {
            "source_frame": "arm_base",
            "target_frame": "world",
            "T_world_from_arm_base": _as_list(arm_result.T_world_from_base),
            "quaternion_wxyz_world_from_arm_base": _matrix_to_wxyz(arm_result.T_world_from_base[:3, :3]),
            "translation_world_m": _as_list(arm_result.T_world_from_base[:3, 3]),
        },
        "tags_to_world": {
            str(tag_id): {
                "source_frame": f"tag_{tag_id}",
                "target_frame": "world",
                "origin_conventions": {
                    "corner2": {
                        "T_world_from_tag": tag_payloads[str(tag_id)]["T_world_from_tag_corner2"],
                        "quaternion_wxyz_world_from_tag": tag_payloads[str(tag_id)][
                            "quaternion_wxyz_world_from_tag_corner2"
                        ],
                        "translation_world_m": tag_payloads[str(tag_id)]["corners_world_m"]["2"],
                    },
                    "center": {
                        "T_world_from_tag": tag_payloads[str(tag_id)]["T_world_from_tag_center"],
                        "quaternion_wxyz_world_from_tag": tag_payloads[str(tag_id)][
                            "quaternion_wxyz_world_from_tag_center"
                        ],
                        "translation_world_m": tag_payloads[str(tag_id)]["center_world_m"],
                    },
                },
            }
            for tag_id in sorted(model.corner_points_by_tag)
        },
    }
    payload = {
        "type": "collect_viser_world_transforms_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "arm_world_calibration_json": str(arm_calibration_path.resolve()),
            "apriltag_model_dir": str(model_dir.resolve()),
            "apriltag_model_json_raw": str(model.json_path),
            "apriltag_model_mesh": str(model.mesh_path),
        },
        "world_frame": {
            "description": (
                "collect_viser AprilTag world frame after applying the model T1 alignment; "
                "tag 100 corner 2 is the world origin."
            ),
            "T1_world_from_model_raw": _as_list(model.T1.matrix()),
            "background_tag_ids": [int(tag_id) for tag_id in model.background_tag_ids],
            "object_tag_ids": [int(tag_id) for tag_id in model.object_tag_ids],
        },
        "transforms_only": transforms_only,
        "arm_base": {
            "frame": "arm_base",
            "parent_frame": "world",
            "tip_position_in_eef_m": _as_list(arm_result.tip_position_in_eef_m),
            "T_world_from_base": _as_list(arm_result.T_world_from_base),
            "T_base_from_world": _as_list(arm_result.T_base_from_world),
            "translation_world_m": _as_list(arm_result.T_world_from_base[:3, 3]),
            "quaternion_wxyz_world_from_base": _matrix_to_wxyz(arm_result.T_world_from_base[:3, :3]),
            "rmse_m": float(arm_result.rmse_m),
            "max_error_m": float(arm_result.max_error_m),
            "sample_counts_by_target": {
                str(key): int(value) for key, value in arm_result.sample_counts_by_target.items()
            },
        },
        "tags": tag_payloads,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm-calibration",
        type=Path,
        default=Path("./data/records/arm_world_calibration.json"),
        help="Path to collect_viser arm-world calibration JSON.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./data/model"),
        help="Directory containing the AprilTag static model files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/records/arm_tags_world_transforms.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    output_path = export_transforms(
        arm_calibration_path=args.arm_calibration,
        model_dir=args.model_dir,
        output_path=args.output,
    )
    print(f"[Export] Saved world transforms -> {output_path}")


if __name__ == "__main__":
    main()
