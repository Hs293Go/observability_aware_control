"""
Copyright © 2024 Hs293Go

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
import jax.numpy.linalg as jla
from jax.scipy import special

from . import lie_derivative


class STLOG:
    """
    The Short Time Local Observability Gramian
    """

    def __init__(self, dynamics, observation, order=1, cov=None):
        self._lie_derivative_gradients = map(
            jax.jacobian, lie_derivative.lie_derivative(observation, dynamics, order)
        )
        if cov is None:
            self._inv_cov = None
        else:
            self._inv_cov = jla.inv(cov)[jnp.newaxis, jnp.newaxis]

        self._order = order

        order_seq = jnp.arange(order + 1)
        self._a, self._b, *_ = jnp.ix_(*[order_seq] * 2)
        self._k = self._a + self._b + 1
        facts = special.factorial(order_seq)
        self._den = facts[self._a] * facts[self._b] * self._k

    @property
    def order(self):
        return self._order

    def __call__(self, x, u, dt):
        lie_derivative_gradients = jnp.stack(
            [it(x, u) for it in self._lie_derivative_gradients]
        )

        factor = dt**self._k / self._den

        if self._inv_cov is None:
            return jnp.sum(
                factor
                * lie_derivative_gradients[self._a].mT
                @ lie_derivative_gradients[self._b],
                axis=(0, 1),
            )

        return jnp.sum(
            factor
            * lie_derivative_gradients[self._a].mT
            @ self._inv_cov
            @ lie_derivative_gradients[self._b],
            axis=(0, 1),
        )
