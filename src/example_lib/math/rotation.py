"""
Copyright © 2024 H S Helson Go and Ching Lok Chong.

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

import equinox as eqx
import jax
import jax.numpy as jnp


@eqx.filter_jit
def angle_rotate_point(angle, point, invert_rotation=False):
    if invert_rotation:
        angle = -angle
    c = jnp.cos(angle)
    s = jnp.sin(angle)

    return jnp.array(
        [
            c * point[0] - s * point[1],
            s * point[0] + c * point[1],
        ]
    )


@jax.jit
def quaternion_product(lhs, rhs):
    return jnp.array(
        [
            lhs[3] * rhs[0] + lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1],
            lhs[3] * rhs[1] + lhs[1] * rhs[3] + lhs[2] * rhs[0] - lhs[0] * rhs[2],
            lhs[3] * rhs[2] + lhs[2] * rhs[3] + lhs[0] * rhs[1] - lhs[1] * rhs[0],
            lhs[3] * rhs[3] - lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2],
        ]
    )


@eqx.filter_jit
def quaternion_rotate_point(quaternion, point, invert_rotation=False):
    vec = -quaternion[0:3] if invert_rotation else quaternion[0:3]
    uv = jnp.cross(vec, point)
    uv += uv
    return point + quaternion[3] * uv + jnp.cross(vec, uv)


@jax.jit
def quaternion_inverse(quaternion):
    return jnp.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])
