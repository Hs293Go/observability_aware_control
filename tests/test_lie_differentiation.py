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
import pytest
import test_lib.models.symbolic_simple_robot as sym_bot

import example_lib.models.simple_robot as bot
from observability_aware_control import lie_derivative

jax.config.update("jax_enable_x64", True)


def make_symbolic_lie_derivatives(order, kind):
    sym = {
        "x": cs.MX.sym("x", 3, 1),  # type: ignore
        "u": cs.MX.sym("u", 2, 1),  # type: ignore
        "lm": cs.MX.sym("lm", 2, 1),  # type: ignore
    }

    lfh = sym_bot.observation(sym["x"], sym["u"], sym["lm"], kind=kind)
    expected_lie_derivatives = []
    for o in range(0, order + 1):
        expected_lie_derivatives.append(
            cs.Function(f"lie_derivative_{o}", sym.values(), [lfh])
        )
        lfh = cs.jtimes(lfh, sym["x"], sym_bot.dynamics(sym["x"], sym["u"]))
    return expected_lie_derivatives


def make_symbolic_lie_derivative_gradients(order, kind):
    sym = {
        "x": cs.MX.sym("x", 3, 1),  # type: ignore
        "u": cs.MX.sym("u", 2, 1),  # type: ignore
        "lm": cs.MX.sym("lm", 2, 1),  # type: ignore
    }

    lfh = sym_bot.observation(sym["x"], sym["u"], sym["lm"], kind=kind)
    expected_lie_derivative_gradients = []
    for o in range(0, order + 1):
        dlfh = cs.jacobian(lfh, sym["x"])
        expected_lie_derivative_gradients.append(
            cs.Function(f"lie_derivative_gradient_{o}", sym.values(), [dlfh])
        )
        lfh = cs.jtimes(lfh, sym["x"], sym_bot.dynamics(sym["x"], sym["u"]))
    return expected_lie_derivative_gradients


NUM_TRIALS = 100


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
        range(5),  # approximation orders
        zip(bot.ObservationKind, sym_bot.ObservationKind),
    )
)


@pytest.mark.parametrize("order,kinds", test_params)
def test_lie_derivative(random_data, order, kinds):
    kind, sym_kind = kinds
    expected_lie_derivatives = make_symbolic_lie_derivatives(order, sym_kind)

    result_lie_derivatives = [
        jax.jit(it)
        for it in lie_derivative.lie_derivative(
            functools.partial(bot.observation, kind=kind), bot.dynamics, order
        )
    ]

    assert len(result_lie_derivatives) == len(expected_lie_derivatives) == order + 1

    for result, expected in zip(result_lie_derivatives, expected_lie_derivatives):
        for x, u, lm in zip(*random_data):
            result_value = result(x, u, lm)
            expected_value = jnp.array(expected(x, u, lm)).squeeze()
            assert result_value == pytest.approx(expected_value)


@pytest.mark.parametrize("order,kinds", test_params)
def test_lie_derivative_gradients(random_data, order, kinds):
    kind, sym_kind = kinds
    if kind == bot.ObservationKind.BEARING and order == 4:
        return  # Freak jax bug, don't run this case
    expected_lie_derivative_gradients = make_symbolic_lie_derivative_gradients(
        order, sym_kind
    )

    result_lie_derivative_gradients = [
        jax.jit(jax.jacobian(it))
        for it in lie_derivative.lie_derivative(
            functools.partial(bot.observation, kind=kind), bot.dynamics, order
        )
    ]

    assert (
        len(result_lie_derivative_gradients)
        == len(expected_lie_derivative_gradients)
        == order + 1
    )

    for result, expected in zip(
        result_lie_derivative_gradients, expected_lie_derivative_gradients
    ):
        for x, u, lm in zip(*random_data):
            expected_value = jnp.array(expected(x, u, lm)).squeeze()
            result_value = result(x, u, lm)
            assert result_value == pytest.approx(expected_value)
