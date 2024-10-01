"""
Copyright © 2024 H S Helson Go and Ching Lok Chong

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import enum

import casadi as cs


class ObservationKind(enum.Enum):
    POSITION = 0
    RANGE = 1
    BEARING = 2


def observation(x, _, lm, kind=ObservationKind.POSITION):
    heading = x[2]
    sx, cx = cs.sin(heading), cs.cos(heading)
    rot = cs.vertcat(cs.horzcat(cx, sx), cs.horzcat(-sx, cx))  # type: ignore
    relative_position = rot @ (lm - x[0:2])
    if kind == ObservationKind.POSITION:
        return cs.vertcat(relative_position, heading)
    if kind == ObservationKind.RANGE:
        return cs.vertcat(cs.norm_2(relative_position), heading)
    if kind == ObservationKind.BEARING:
        return cs.vertcat(cs.atan2(relative_position[1], relative_position[0]), heading)
    raise ValueError(f"{kind} is not a valid kind of interrobot observation")


def dynamics(x, u):
    sx, cx = cs.sin(x[2]), cs.cos(x[2])
    return cs.vertcat(cx * u[0], sx * u[0], u[1])
