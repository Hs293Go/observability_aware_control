import enum

import casadi as cs


class ObservationKind(enum.Enum):
    POSITION = 0
    RANGE = 1
    BEARING = 2


def observation(x, _, lm, kind=ObservationKind.POSITION):
    sx, cx = cs.sin(x[2]), cs.cos(x[2])
    rot = cs.vertcat(cs.horzcat(cx, sx), cs.horzcat(-sx, cx))  # type: ignore
    relative_position = rot @ (lm - x[0:2])
    if kind == ObservationKind.POSITION:
        return relative_position
    if kind == ObservationKind.RANGE:
        return cs.norm_2(relative_position)
    if kind == ObservationKind.BEARING:
        return cs.atan2(relative_position[1], relative_position[0])
    raise ValueError(f"{kind} is not a valid kind of interrobot observation")


def dynamics(x, u):
    sx, cx = cs.sin(x[2]), cs.cos(x[2])
    return cs.vertcat(cx * u[0], sx * u[0], u[1])
