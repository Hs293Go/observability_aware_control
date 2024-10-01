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

import functools
import itertools

import casadi as cs
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import pytest
import test_lib.models.symbolic_simple_robot as sym_bot
from jax.scipy import special

import example_lib.models.simple_robot as bot
from observability_aware_control import elog, stlog

jax.config.update("jax_enable_x64", True)

# pylint: disable=not-callable


def make_symbolic_stlog(order, kind):
    sym = {
        "x": cs.MX.sym("x", 3, 1),  # type: ignore
        "u": cs.MX.sym("u", 2, 1),  # type: ignore
        "dt": cs.MX.sym("dt"),  # type: ignore
        "lm": cs.MX.sym("lm", 2, 1),  # type: ignore
    }

    lfh = sym_bot.observation(sym["x"], sym["u"], sym["lm"], kind=kind)
    lie_derivative_gradients = []
    for _ in range(0, order + 1):
        dlfh = cs.jacobian(lfh, sym["x"])
        lie_derivative_gradients.append(dlfh)
        lfh = cs.jtimes(lfh, sym["x"], sym_bot.dynamics(sym["x"], sym["u"]))

    sym_stlog = sum(
        (
            sym["dt"] ** (i + j + 1)
            / ((i + j + 1) * special.factorial(i) * special.factorial(j))
        )
        * lie_derivative_gradients[i].T
        @ lie_derivative_gradients[j]
        for i, j in itertools.product(range(order + 1), range(order + 1))
    )
    return cs.Function("sym_stlog", sym.values(), [sym_stlog])


ORDER = 2
NUM_TRIALS = 500


@pytest.fixture(name="random_data")
def _():
    key = jax.random.PRNGKey(1000)
    x_key, u_key, lm_key = jax.random.split(key, 3)
    x_batch = jax.random.uniform(
        x_key,
        (NUM_TRIALS, 3),
        minval=jnp.array([-1.0, -1.0, -jnp.pi]),
        maxval=jnp.array([1.0, 1.0, jnp.pi]),
    )
    u_batch = jax.random.uniform(
        u_key, (NUM_TRIALS, 2), minval=jnp.array([0, -5]), maxval=jnp.array([10, 5])
    )
    lm_batch = jax.random.uniform(lm_key, (NUM_TRIALS, 2))
    return x_batch, u_batch, lm_batch


test_params = list(
    itertools.product(
        jnp.arange(1, 6) * 0.1,  # step sizes
        zip(bot.ObservationKind, sym_bot.ObservationKind),
    )
)


@pytest.mark.parametrize("stepsize,kinds", test_params)
def test_stlog(random_data, stepsize, kinds):
    kind, sym_kind = kinds

    sym_gramian = make_symbolic_stlog(ORDER, sym_kind)
    observation_fcn = functools.partial(bot.observation, kind=kind)

    gramian = jax.jit(stlog.STLOG(bot.dynamics, observation_fcn, ORDER))

    for x0, us, lm in zip(*random_data):
        result = gramian(x0, us, stepsize, lm)  # pylint: disable=not-callable
        expected = jnp.asarray(sym_gramian(x0, us, stepsize, lm))
        assert result == pytest.approx(expected)


def normclose(a, b, rtol=1e-2):
    """Even fuzzier, linear-algebra based matrix comparison for Gramians"""
    return jla.norm(a - b) <= jnp.maximum(jla.norm(a), jla.norm(b)) * rtol


test_params_sve = itertools.product([0.01, 0.03, 0.05], list(bot.ObservationKind)[0:2])

ELOG_STEPS = 40


@pytest.mark.parametrize("stepsize,kind", test_params_sve)
def test_stlog_vs_elog(random_data, stepsize, kind):

    observation_fcn = functools.partial(bot.observation, kind=kind)

    stlog_gramian = jax.jit(stlog.STLOG(bot.dynamics, observation_fcn, ORDER))
    elog_gramian = jax.jit(elog.ELOG(bot.dynamics, observation_fcn))

    for x0, u, lm in zip(*random_data):
        # We need to space out the landmarks from the vehicle position as
        # tests will fail due to weird gramian values in blatantly unobservable
        # configurations, i.e. landmark very close to vehicle position
        lm += jnp.array([3.0, 3.0])
        expected = stlog_gramian(x0, u, stepsize, lm)
        us, lms = jnp.array([u] * ELOG_STEPS), jnp.array([lm] * ELOG_STEPS)
        result = elog_gramian(x0, us, stepsize / ELOG_STEPS, lms)
        assert normclose(result, expected)
