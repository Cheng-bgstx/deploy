from types import SimpleNamespace

import numpy as np
import pytest

from sim2real.rl_policy.utils.shaping_gravity import ShapingGravity


@pytest.fixture
def cfg(tmp_path):
    path = tmp_path / 'pendulum.xml'
    path.write_text('''<mujoco><worldbody><body name="pelvis"><freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body><joint name="arm" axis="0 1 0"/>
        <geom type="sphere" size="0.1" pos="1 0 0" mass="1"/>
      </body></body></worldbody></mujoco>''')
    return SimpleNamespace(resolve_mjcf_path=lambda: path, joint_names=('arm',), joint_effort_limit={'arm': 20.})


def test_gravity_direction_follows_measured_orientation_and_ramps_with_torque_cap(cfg):
    gravity = ShapingGravity(cfg, max_effort_fraction=.1, ramp_s=2.)
    np.testing.assert_allclose(gravity.compute([0.], [1, 0, 0, 0], 0.), [0.])
    np.testing.assert_allclose(gravity.compute([0.], [1, 0, 0, 0], 1.), [-1.])
    np.testing.assert_allclose(gravity.compute([0.], [1, 0, 0, 0], 2.), [-2.])
    # Rotating the root 180 degrees about x flips gravity in joint coordinates.
    np.testing.assert_allclose(gravity.compute([0.], [0, 1, 0, 0], 2.), [2.])
    np.testing.assert_allclose(gravity.compute([np.pi / 2], [1, 0, 0, 0], 2.), [0.], atol=1e-12)


def test_invalid_measured_orientation_is_rejected(cfg):
    gravity = ShapingGravity(cfg)
    with pytest.raises(ValueError, match='Invalid measured pose'):
        gravity.compute([0.], [0, 0, 0, 0], 2.)
    with pytest.raises(ValueError, match='Invalid measured pose'):
        gravity.compute([np.nan], [1, 0, 0, 0], 2.)
