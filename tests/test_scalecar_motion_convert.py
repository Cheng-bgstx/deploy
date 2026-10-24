"""Scalecar 50 Hz motion NPZ -> any4hdmi dataset conversion (issue #4)."""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.scalecar.convert_motions import (
    TAR_MOTIONS_DIR,
    convert_motion_files,
    extract_takes,
)

# Scalecar NPZ / tracker contract joint order (deployment.json `joint_names`).
TRACKER_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

# any4hdmi keeps every loaded motion field in a float16 backing, and sim2real's
# MotionDataset hands those out as float16 arrays, so values that went through
# the loader are only good to about half a float16 ulp (<= 4.9e-4 for |x| < 2).
# A re-centred pelvis would be off by metres.
LOADER_ATOL = 1e-3


def _quat_wxyz_from_rpy(roll, pitch, yaw) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        axis=-1,
    )


def _yaw_of_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(np.asarray(quat, dtype=np.float64), -1, 0)
    return np.arctan2(2 * (w * z + x * y), w * w + x * x - y * y - z * z)


def make_scalecar_motion(frames: int = 60) -> dict[str, np.ndarray]:
    """A Scalecar-style 50 Hz NPZ payload whose pelvis sits away from the origin."""
    t = np.arange(frames, dtype=np.float64)
    alpha = (t / (frames - 1))[:, None]
    pelvis_pos = (1 - alpha) * [-0.216, -0.954, 0.793] + alpha * [-0.089, -0.321, 0.536]
    yaw = 1.2 + 0.8 * alpha[:, 0]
    pelvis_quat = _quat_wxyz_from_rpy(0.05 + 0 * t, -0.1 + 0 * t, yaw)
    joint_pos = 0.4 * np.sin(0.1 * t[:, None] + np.arange(29)[None, :])
    body_names = ["left_hip_pitch_link", "torso_link", "pelvis"]
    pelvis = body_names.index("pelvis")
    body_pos = np.zeros((frames, len(body_names), 3))
    body_quat = np.zeros((frames, len(body_names), 4))
    body_quat[..., 0] = 1.0
    body_pos[:, pelvis] = pelvis_pos
    body_quat[:, pelvis] = pelvis_quat
    return {
        "fps": np.array(50),
        "robot_joint_names": np.array(TRACKER_JOINT_NAMES),
        "robot_body_names": np.array(body_names),
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": np.gradient(joint_pos, 0.02, axis=0).astype(np.float32),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": np.zeros((frames, len(body_names), 3), np.float32),
        "body_ang_vel_w": np.zeros((frames, len(body_names), 3), np.float32),
        "object_pos_w": np.zeros((frames, 1, 3), np.float32),
    }


def _npz_bytes(**arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def _write_sums(path: Path, tar_path: Path, digest: str | None = None) -> None:
    if digest is None:
        digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    path.write_text(
        f"{'0' * 64}  other.tar\n{digest}  {tar_path.name}\n", encoding="utf-8"
    )


class ExtractTakesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.tar_path = self.tmp / "motions.tar"
        _write_tar(
            self.tar_path,
            {
                f"{TAR_MOTIONS_DIR}/take0001.npz": _npz_bytes(fps=np.array(50.0)),
                f"{TAR_MOTIONS_DIR}/take0002.npz": _npz_bytes(fps=np.array(51.0)),
                "data/README.md": b"not a motion",
            },
        )
        self.sums_path = self.tmp / "SHA256SUMS"
        self.out_dir = self.tmp / "out"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_extracts_only_requested_takes_after_checksum(self) -> None:
        _write_sums(self.sums_path, self.tar_path)

        paths = extract_takes(self.tar_path, self.sums_path, ["take0002"], self.out_dir)

        self.assertEqual(paths, {"take0002": self.out_dir / "take0002.npz"})
        self.assertEqual(sorted(p.name for p in self.out_dir.iterdir()), ["take0002.npz"])
        with np.load(paths["take0002"], allow_pickle=False) as data:
            self.assertEqual(float(data["fps"]), 51.0)

    def test_refuses_tar_whose_checksum_does_not_match(self) -> None:
        _write_sums(self.sums_path, self.tar_path, digest="f" * 64)

        with self.assertRaisesRegex(ValueError, "sha256"):
            extract_takes(self.tar_path, self.sums_path, ["take0001"], self.out_dir)
        self.assertFalse(self.out_dir.exists() and any(self.out_dir.iterdir()))

    def test_refuses_tar_missing_from_checksum_list(self) -> None:
        self.sums_path.write_text(f"{'0' * 64}  other.tar\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "SHA256SUMS"):
            extract_takes(self.tar_path, self.sums_path, ["take0001"], self.out_dir)

    def test_reports_takes_missing_from_tar(self) -> None:
        _write_sums(self.sums_path, self.tar_path)

        with self.assertRaisesRegex(KeyError, "take9999"):
            extract_takes(self.tar_path, self.sums_path, ["take0001", "take9999"], self.out_dir)

    def test_rejects_take_names_that_are_not_plain_identifiers(self) -> None:
        _write_sums(self.sums_path, self.tar_path)

        with self.assertRaisesRegex(ValueError, "take name"):
            extract_takes(self.tar_path, self.sums_path, ["../take0001"], self.out_dir)


class _ConvertedMotionCase(unittest.TestCase):
    """Writes one synthetic Scalecar take, converts it, and isolates the any4hdmi cache."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        env = mock.patch.dict(
            os.environ,
            {
                "ANY4HDMI_QPOS_CACHE_ROOT": str(self.tmp / "any4hdmi_cache"),
                "ANY4HDMI_CACHE_BUILD_NUM_WORKERS": "0",
            },
        )
        env.start()
        self.addCleanup(env.stop)
        self.source = make_scalecar_motion()
        self.source_path = self.tmp / "take9999.npz"
        np.savez(self.source_path, **self.source)
        self.dataset_root = self.tmp / "motions"
        self.motion_paths = convert_motion_files(
            {"take9999": self.source_path}, self.dataset_root
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()


class ConvertedMotionLoadsInWorldFrameTest(_ConvertedMotionCase):
    def test_sim2real_loader_keeps_world_pelvis_pose_and_joint_names(self) -> None:
        from sim2real.config.robots import get_robot_cfg
        from sim2real.rl_policy.utils.motion import MotionDataset

        robot_cfg = get_robot_cfg("g1")
        self.assertEqual(
            self.motion_paths, {"take9999": self.dataset_root / "motions" / "take9999.npz"}
        )
        dataset = MotionDataset.create_from_path(
            str(self.motion_paths["take9999"]), robot_cfg=robot_cfg
        )
        frames = len(self.source["joint_pos"])
        self.assertEqual(dataset.num_steps, frames)

        data = dataset.get_slice(np.array([0]), np.array([0]), np.arange(frames))
        pelvis = dataset.body_names.index("pelvis")
        pelvis_pos = data.body_pos_w[0, :, pelvis]
        pelvis_quat = data.body_quat_w[0, :, pelvis]
        source_pelvis = list(self.source["robot_body_names"]).index("pelvis")
        np.testing.assert_allclose(
            pelvis_pos, self.source["body_pos_w"][:, source_pelvis], atol=LOADER_ATOL
        )
        np.testing.assert_allclose(
            _yaw_of_wxyz(pelvis_quat),
            _yaw_of_wxyz(self.source["body_quat_w"][:, source_pelvis]),
            atol=LOADER_ATOL,
        )

        by_name = [TRACKER_JOINT_NAMES.index(name) for name in robot_cfg.joint_names]
        np.testing.assert_allclose(
            data.joint_pos[0], self.source["joint_pos"][:, by_name], atol=LOADER_ATOL
        )


SCALECAR_REFERENCE_PY = Path(
    "/home/galbot/projects/Scalecar_v2/source/scalecar_policy/scalecar_policy/reference.py"
)


@unittest.skipUnless(SCALECAR_REFERENCE_PY.is_file(), "Scalecar_v2 checkout not available")
class ReferenceTokenComparisonTest(_ConvertedMotionCase):
    def setUp(self) -> None:
        super().setUp()
        from scripts.scalecar.compare_reference_tokens import (
            compare_reference_tokens,
            load_reference_only_features,
        )

        self.result = compare_reference_tokens(
            self.motion_paths["take9999"],
            self.source_path,
            reference_only_features=load_reference_only_features(SCALECAR_REFERENCE_PY),
        )

    def test_conversion_matches_scalecar_tokens_before_float16_storage(self) -> None:
        errors = self.result["fp32_fk_cache"]
        for component in ("q", "displacement", "height", "rotation"):
            self.assertLessEqual(errors[component], 1e-5, component)
        self.assertEqual(errors["offset"], 0.0)

    def test_loader_tokens_differ_from_conversion_only_by_float16_rounding(self) -> None:
        self.assertEqual(max(self.result["loader_vs_fp16_rounded_fp32"].values()), 0.0)
        for component in ("q", "displacement", "height", "rotation"):
            self.assertLessEqual(self.result["loader"][component], 2 * LOADER_ATOL, component)

    def test_reports_end_frame_dq_of_both_sides(self) -> None:
        last = self.result["dq_last_frame"]
        q = self.source["joint_pos"].astype(np.float64)
        forward_difference = np.abs((q[-1] - q[-2]) * 50).max()
        self.assertEqual(last["scalecar_max_abs"], 0.0)
        self.assertAlmostEqual(last["sim2real_max_abs"], forward_difference, delta=1e-2)


if __name__ == "__main__":
    unittest.main()
