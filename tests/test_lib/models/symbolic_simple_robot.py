import casadi as cs


def observation(x, _, lm):
    sx, cx = cs.sin(x[2]), cs.cos(x[2])
    rot = cs.vertcat(cs.horzcat(cx, sx), cs.horzcat(-sx, cx))  # type: ignore
    return rot @ (lm - x[0:2])


def dynamics(x, u):
    sx, cx = cs.sin(x[2]), cs.cos(x[2])
    return cs.vertcat(cx * u[0], sx * u[0], u[1])
