"""Synthetic PM01 wave reference: starts and ends at the default pose, moves only the waving arm."""
import numpy as np
import pytest

from sim2real.config.robots import PM01_CFG

pytestmark = pytest.mark.skipif(not PM01_CFG.resolve_mjcf_path().is_file(),
                                reason='prepared PM01 model missing (scripts/pm01/prepare.py)')


@pytest.mark.parametrize('arm', ['right', 'left'])
def test_wave_moves_only_one_arm_and_returns_to_default(tmp_path, arm):
    from any4hdmi.core.format import load_motion
    from scripts.pm01.make_wave_motion import ARM_JOINTS, make_wave

    make_wave(tmp_path / 'wave', arm, duration=10.0, freq=1.0, ramp=1.5, hold=1.0, min_gap=0.05)
    qpos = load_motion(tmp_path / 'wave' / 'motions' / f'wave_{arm}.npz')
    default = np.asarray(PM01_CFG.default_qpos)
    np.testing.assert_allclose(qpos[0], default, atol=1e-6)
    np.testing.assert_allclose(qpos[-1], default, atol=1e-6)
    side = 'R' if arm == 'right' else 'L'
    moving = {f'{prefix}_{key}_{side}' for key, prefix in ARM_JOINTS[arm].items()}
    for i, name in enumerate(PM01_CFG.joint_names):
        span = np.ptp(qpos[:, 7 + i])
        if name in moving:
            assert span > 0.3, name
        else:
            assert span < 1e-9, name
    assert np.abs(np.diff(qpos[:, 7:], axis=0)).max() * 50 < 3.0  # rad/s
