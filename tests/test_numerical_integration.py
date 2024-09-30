import casadi as cs
import jax
import jax.numpy as jnp
import pytest
import test_lib.models.symbolic_simple_robot as sym_bot

import example_lib.models.simple_robot as bot
from observability_aware_control import integrator

jax.config.update("jax_enable_x64", True)


def make_symbolic_integrator(dt):
    sym = {
        "x": cs.MX.sym("x", 3, 1),  # type: ignore
        "u": cs.MX.sym("u", 2, 1),  # type: ignore
    }
    sym["ode"] = sym_bot.dynamics(sym["x"], sym["u"])
    return cs.integrator("simple_robot", "rk", sym, 0.0, dt)


def symbolic_solve_ivp(sym_solve_ivp, x_op, us):

    expected_xs = [jnp.array(x_op)]
    for u in us:
        res = sym_solve_ivp(x0=x_op, u=u)
        x_op = res["xf"]
        expected_xs.append(jnp.array(x_op).ravel())
    return jnp.asarray(expected_xs)


NUM_TRIALS = 20
NUM_STEPS = 100


@pytest.fixture(name="random_data")
def _():
    key = jax.random.PRNGKey(1000)
    x_key, u_key = jax.random.split(key)
    x0 = jax.random.uniform(
        x_key,
        (NUM_TRIALS, 3),
        minval=jnp.array([-1.0, -1.0, -jnp.pi]),
        maxval=jnp.array([1.0, 1.0, jnp.pi]),
    )
    us = jax.random.uniform(
        u_key,
        (NUM_TRIALS, NUM_STEPS, 2),
        minval=jnp.array([0, -5]),
        maxval=jnp.array([10, 5]),
    )

    return x0, us


stepsizes = jnp.arange(1, 6) * 0.01


@pytest.mark.parametrize("dt", stepsizes)
def test_integration(random_data, dt):

    sym_solve_ivp = make_symbolic_integrator(dt)
    solve_ivp = jax.jit(integrator.Integrator(bot.dynamics, stepsize=dt))
    for x0, us in zip(*random_data):

        expected_xs = symbolic_solve_ivp(sym_solve_ivp, x0, us)

        result_xs, _ = solve_ivp(x0, us)  # pylint: disable=not-callable

        assert jnp.allclose(result_xs, expected_xs[:-1, :], rtol=1e-5)
