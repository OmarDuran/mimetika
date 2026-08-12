"""RKTimeStepping: the step sizes and the scheme are independent choices."""

import numpy as np
import pytest

from mimetika.simulation import RKTimeStepping
from mimetika.simulation.timestepping import TABLEAUS


def test_exactly_one_of_dt_and_schedule():
    with pytest.raises(ValueError):
        RKTimeStepping()
    with pytest.raises(ValueError):
        RKTimeStepping(dt=0.1, schedule=[0.1, 0.2])
    RKTimeStepping(dt=0.1)
    RKTimeStepping(schedule=[0.1, 0.2])


def test_unknown_tableau_raises_rather_than_misintegrating():
    with pytest.raises(NotImplementedError):
        RKTimeStepping(dt=0.1, tableau="dormand-prince")


def test_backward_euler_butcher_tableau():
    A, b, c = RKTimeStepping(dt=0.1).butcher
    assert np.allclose(A, [[1.0]]) and np.allclose(b, [1.0]) \
        and np.allclose(c, [1.0])
    assert "backward-euler" in TABLEAUS


def test_constant_flag_names_the_fast_path_condition():
    assert RKTimeStepping(dt=0.5).constant
    assert not RKTimeStepping(schedule=[1.0, 2.0]).constant


def test_constant_steps_by_count():
    steps = list(RKTimeStepping(dt=0.25).steps(n_steps=4))
    assert steps == [(0.25, 0.25), (0.5, 0.25), (0.75, 0.25), (1.0, 0.25)]


def test_constant_steps_by_final_time():
    steps = list(RKTimeStepping(dt=0.25).steps(t_end=1.0))
    assert len(steps) == 4
    assert steps[-1][0] == pytest.approx(1.0)


def test_constant_steps_need_a_horizon():
    with pytest.raises(ValueError):
        next(RKTimeStepping(dt=0.25).steps())


def test_schedule_yields_the_consecutive_differences():
    ts = RKTimeStepping(schedule=[0.1, 0.3, 1.0], t0=0.0)
    steps = list(ts.steps())
    assert [t for t, _ in steps] == pytest.approx([0.1, 0.3, 1.0])
    assert [dt for _, dt in steps] == pytest.approx([0.1, 0.2, 0.7])


def test_schedule_respects_t0():
    steps = list(RKTimeStepping(schedule=[2.0, 3.0], t0=1.0).steps())
    assert steps[0] == pytest.approx((2.0, 1.0))


def test_schedule_must_increase_strictly_after_t0():
    with pytest.raises(ValueError):
        RKTimeStepping(schedule=[0.2, 0.2, 0.3])
    with pytest.raises(ValueError):
        RKTimeStepping(schedule=[0.5, 1.0], t0=0.5)
    with pytest.raises(ValueError):
        RKTimeStepping(schedule=[])
