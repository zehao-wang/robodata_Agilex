"""Arm reader that treats the PushT service as the robot backend.

The service owns CAN/piper_sdk. This client only speaks HTTP:

- PushTClient.get_robot_qpos() -> lightweight joint feedback polling
- POST /plan             -> plan to the prepared target XYZ waypoint
- POST /arm/execute      -> execute that planned trajectory
- PushTClient.step_eef() -> plan + execute relative EEF motion
- PushTClient.arm_status() -> fallback joint/status data
"""

import json
import threading
import time
from urllib import error, request

import numpy as np
from pusht_client import PushTClient

from .arm_reader import ArmState


_SETUP_CAN_HINT = (
    "If CAN is not ready, run `./setup_can.sh` first, then restart the "
    "PushT service and collect_viser."
)


class PushTHTTPArmReader:
    """Poll arm state from the PushT HTTP service."""

    def __init__(
        self,
        service_url: str = "http://127.0.0.1:8012",
        poll_hz: float = 30.0,
        timeout_s: float = 1.0,
        connect_on_start: bool = False,
        can_interface: str = "socketcan",
        can_channel: str = "can0",
    ):
        self.service_url = service_url.rstrip("/")
        self.poll_hz = float(poll_hz)
        self.timeout_s = float(timeout_s)
        self.connect_on_start = bool(connect_on_start)
        self.can_interface = can_interface
        self.can_channel = can_channel
        self._client = PushTClient(service_url=self.service_url)

        self._state = ArmState()
        self._prev_qpos = np.zeros(6, dtype=np.float64)
        self._prev_time = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._last_error = ""
        self._known_connected = False

    def start(self) -> None:
        """Start polling the service."""
        if self.connect_on_start:
            try:
                self._client.connect_arm(
                    channel=self.can_channel,
                    interface=self.can_interface,
                )
                self._known_connected = True
                self._client.update_server_state()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to connect PushT arm on {self.can_interface}:{self.can_channel}. "
                    f"{_SETUP_CAN_HINT}"
                ) from exc
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(f"[PushTHTTPArmReader] Polling {self.service_url}")

    def stop(self) -> None:
        """Stop polling. The remote arm service remains connected."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[PushTHTTPArmReader] Stopped")

    def get_state(self) -> ArmState:
        """Return the latest HTTP-backed arm state."""
        with self._lock:
            return ArmState(
                qpos=self._state.qpos.copy(),
                qvel=self._state.qvel.copy(),
                gripper=self._state.gripper,
                timestamp=self._state.timestamp,
                eef_pos_m=(
                    None if self._state.eef_pos_m is None else self._state.eef_pos_m.copy()
                ),
                eef_euler_deg=(
                    None
                    if self._state.eef_euler_deg is None
                    else self._state.eef_euler_deg.copy()
                ),
                pose_timestamp=self._state.pose_timestamp,
                pose_source=self._state.pose_source,
            )

    @property
    def last_error(self) -> str:
        return self._last_error

    def connect_arm(self) -> dict:
        """Ask the PushT service client to connect the server-owned arm."""
        try:
            result = self._client.connect_arm(
                channel=self.can_channel,
                interface=self.can_interface,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect PushT arm on {self.can_interface}:{self.can_channel}. "
                f"{_SETUP_CAN_HINT}"
            ) from exc
        self._known_connected = True
        self._read_once()
        return result

    def lock_pose(self, speed: int = 50) -> dict:
        """Ask the PushT service to enable motors and latch CAN_Ctrl hold."""
        if not self._known_connected:
            self.connect_arm()
        result = self._client.lock_pose(speed=int(speed))
        self._read_once()
        return result

    def step_eef(
        self,
        delta_base_m: np.ndarray,
        *,
        timesteps: int = 15,
        speed: int = 1,
        timeout_s: float = 8.0,
    ) -> dict:
        """Plan and execute a relative EEF step through ``PushTClient.step_eef``."""
        delta = np.asarray(delta_base_m, dtype=np.float64).reshape(3)
        result = self._client.step_eef(
            dx=float(delta[0]),
            dy=float(delta[1]),
            dz=float(delta[2]),
            timesteps=int(timesteps),
            speed=int(speed),
            step_timeout=float(timeout_s),
        )
        self._read_once()
        return {
            "status": result.get("status", "ok"),
            "delta_base_m": [float(v) for v in delta],
            "timesteps": int(timesteps),
            "speed": int(speed),
            "step": result,
        }

    def move_to_base_position(
        self,
        target_base_m: np.ndarray,
        *,
        timesteps: int = 15,
        speed: int = 1,
        timeout_s: float = 8.0,
    ) -> dict:
        """Move toward an absolute 3D pusher target in robot base coordinates."""
        target = np.asarray(target_base_m, dtype=np.float64).reshape(3)
        trajectory = self._client.plan_and_execute(
            waypoints=[(float(target[0]), float(target[1]), float(target[2]))],
            speed=int(speed),
            timesteps=int(timesteps),
        )
        self._read_once()
        return {
            "status": "ok",
            "target_base_m": [float(v) for v in target],
            "timesteps": int(timesteps),
            "speed": int(speed),
            "planned_configs": len(trajectory),
        }

    def move_by_base_delta(
        self,
        delta_base_m: np.ndarray,
        *,
        timesteps: int = 15,
        speed: int = 1,
        timeout_s: float = 8.0,
    ) -> dict:
        """Compatibility wrapper for older GUI code; prefer ``step_eef``."""
        return self.step_eef(
            delta_base_m,
            timesteps=timesteps,
            speed=speed,
            timeout_s=timeout_s,
        )

    def _read_loop(self) -> None:
        target_dt = 1.0 / max(self.poll_hz, 1.0)
        while self._running:
            t0 = time.perf_counter()
            self._read_once()
            sleep_time = target_dt - (time.perf_counter() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _read_once(self) -> bool:
        now = time.time()
        status = None
        if not self._known_connected:
            try:
                status = self._client.arm_status()
            except Exception as exc:
                self._last_error = f"arm_status failed: {exc}"
                return False

            if not status.get("connected", False):
                self._last_error = (
                    "PushT service arm is not connected. Click Lock Robot Arm, "
                    "start collect_viser with --arm-http-connect, or call "
                    "client.connect_arm() first."
                )
                return False
            self._known_connected = True

        try:
            qpos_live = self._client.get_robot_qpos()
        except Exception as exc:
            self._known_connected = False
            if status is None:
                try:
                    status = self._client.arm_status()
                except Exception as fallback_exc:
                    self._last_error = (
                        f"get_robot_qpos failed: {exc}; arm_status fallback failed: {fallback_exc}"
                    )
                    return False
            return self._read_status_payload(status, now)

        if len(qpos_live) < 1:
            self._last_error = "PushT get_robot_qpos returned no joints"
            return False

        qpos = np.zeros(6, dtype=np.float64)
        qpos[: min(len(qpos_live), 6)] = np.asarray(qpos_live[:6], dtype=np.float64)

        qvel = self._estimate_qvel(qpos, now)

        with self._lock:
            self._state = ArmState(
                qpos=qpos,
                qvel=qvel,
                gripper=0.0,
                timestamp=now,
                eef_pos_m=None,
                eef_euler_deg=None,
                pose_timestamp=0.0,
                pose_source="pusht-client-qpos",
            )
        self._last_error = ""
        return True

    def _read_status_payload(self, status: dict, now: float) -> bool:
        joints = status.get("joints_rad") or []
        if len(joints) < 1:
            self._last_error = "PushT service returned no joints_rad"
            return False

        qpos = np.zeros(6, dtype=np.float64)
        qpos[: min(len(joints), 6)] = np.asarray(joints[:6], dtype=np.float64)

        qvel = self._estimate_qvel(qpos, now)

        with self._lock:
            self._state = ArmState(
                qpos=qpos,
                qvel=qvel,
                gripper=0.0,
                timestamp=now,
                eef_pos_m=None,
                eef_euler_deg=None,
                pose_timestamp=0.0,
                pose_source="pusht-http-status",
            )
        self._last_error = ""
        return True

    def _estimate_qvel(self, qpos: np.ndarray, now: float) -> np.ndarray:
        qvel = np.zeros(6, dtype=np.float64)
        if self._prev_time > 0.0:
            dt = now - self._prev_time
            if dt > 0.0:
                qvel = (qpos - self._prev_qpos) / dt
        self._prev_qpos = qpos.copy()
        self._prev_time = now
        return qvel

    def _get_json(self, path: str) -> dict:
        req = request.Request(f"{self.service_url}{path}", method="GET")
        return self._request_json(req)

    def _post_json(self, path: str, payload: dict, timeout_s: float | None = None) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.service_url}{path}",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        return self._request_json(req, timeout_s=timeout_s)

    def _request_json(self, req: request.Request, timeout_s: float | None = None) -> dict:
        try:
            with request.urlopen(req, timeout=self.timeout_s if timeout_s is None else timeout_s) as resp:
                data = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {req.full_url}: {detail}") from exc
        return json.loads(data) if data else {}
