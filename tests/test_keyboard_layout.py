from types import SimpleNamespace

from sim2real.rl_policy.base_policy import BasePolicy
from sim2real.rl_policy.control_mode import KEYBOARD_PLAY_KEYS, resolve_keyboard_control_mode


def test_default_layout_is_unchanged():
    assert [resolve_keyboard_control_mode(k) for k in ('i', ']', 'o', 'a', ' ')] == ['init', 'policy', 'zero', None, None]
    assert KEYBOARD_PLAY_KEYS['default'] == ('space', ' ')


def test_et1_layout_matches_the_et1_operator_keys():
    resolve = lambda k: resolve_keyboard_control_mode(k, 'et1')
    assert [resolve(k) for k in ('a', 'A', 'space', ' ', 'x')] == ['init', 'init', 'zero', 'zero', 'zero']
    # s plays from Hold (an extra key); the default-layout keys do nothing.
    assert [resolve(k) for k in ('s', 'i', ']', 'o')] == [None, None, None, None]
    assert KEYBOARD_PLAY_KEYS['et1'] == ('s',)


class _Policy:
    """The parts of BasePolicy that process_controllers touches."""

    def __init__(self, mode, init_count):
        self.joystick_controller = None
        self.controller = SimpleNamespace(name='keyboard', repeat_init_enters_policy=True,
                                          get_control_mode=lambda: mode)
        self.state_dict = {'control_mode': 'init'}
        self.init_count, self.rl_dt, self.calls = init_count, 0.02, []
        self.set_policy_mode = lambda source: self.calls.append('policy')
        self.set_init_mode = lambda source: self.calls.append('init')
        self.set_zero_mode = lambda source: self.calls.append('zero')


def test_second_a_enters_hold_only_after_shaping():
    early = _Policy('init', init_count=100)  # 2 s into the 10 s shaping
    BasePolicy.process_controllers(early)
    assert early.calls == []
    done = _Policy('init', init_count=600)
    BasePolicy.process_controllers(done)
    assert done.calls == ['policy']


def test_a_from_zero_shapes_and_stop_returns_to_zero():
    first = _Policy('init', init_count=0)
    first.state_dict['control_mode'] = 'zero'
    BasePolicy.process_controllers(first)
    assert first.calls == ['init']
    stop = _Policy('zero', init_count=600)
    BasePolicy.process_controllers(stop)
    assert stop.calls == ['zero']
