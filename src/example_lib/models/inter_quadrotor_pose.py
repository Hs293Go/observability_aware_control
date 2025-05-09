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

import functools as fn

import jax.numpy as jnp

from example_lib import math


def dynamics(x, u):
    r_lf, q_fl, v_lf = jnp.split(x, (3, 7))

    f_l, omega_l, f_f, omega_f = jnp.split(u, (1, 4, 5, 4))

    return jnp.concatenate(
        [
            -jnp.cross(r_lf, omega_f) + v_lf,
            math.quaternion_product(q_fl, jnp.concatenate([0.5 * omega_l, 0.0]))
            + math.quaternion_product(jnp.concatenate([0.5 * omega_f, 0.0]), q_fl),
            -jnp.cross(v_lf, omega_f)
            + math.quaternion_rotate_point(q_fl, jnp.array([0.0, 0.0, f_l]))
            - jnp.array([0.0, 0.0, f_f]),
        ]
    )
