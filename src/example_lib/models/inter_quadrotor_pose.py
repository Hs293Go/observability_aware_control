"""
Copyright © 2025 Hs293Go

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

import jax
import jax.numpy as jnp

from example_lib import math

NUM_STATES = 10
NUM_INPUTS = 8


@jax.jit
def dynamics(x, u):
    r_lf, q_fl, v_lf = x[:3], x[3:7], x[7:10]

    f_l, omega_l, f_f, omega_f = u[0], u[1:4], u[4], u[5:8]

    w_f = jnp.array([omega_f[0] / 2.0, omega_f[1] / 2.0, omega_f[2] / 2.0, 0.0])
    w_l = jnp.array([omega_l[0] / 2.0, omega_l[1] / 2.0, omega_l[2] / 2.0, 0.0])

    t_l, t_f = jnp.array([0.0, 0.0, f_l]), jnp.array([0.0, 0.0, f_f])
    return jnp.concatenate(
        [
            jnp.cross(r_lf, omega_f) + v_lf,
            -math.quaternion_product(w_f, q_fl) + math.quaternion_product(q_fl, w_l),
            jnp.cross(v_lf, omega_f) + math.quaternion_rotate_point(q_fl, t_l) - t_f,
        ]
    )


def observation(x, _=None, *, use_sqnorm=True):
    r_lf = x[0:3]
    q_fl = x[3:7]

    r_lf_sqnorm = jnp.dot(r_lf, r_lf)
    if use_sqnorm:
        range_meas = r_lf_sqnorm
    else:
        range_meas = jnp.sqrt(r_lf_sqnorm)
    return jnp.concatenate([jnp.array([range_meas]), q_fl])


def interrobot_distance_squared(x0, us, cost):
    xs, _ = cost.eval_integrator(x0, us)
    return jnp.sum(xs[:, 0:3] ** 2, axis=1)
