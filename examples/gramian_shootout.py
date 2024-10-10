import functools as fn
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy import optimize

import example_lib.models.leader_follower_robots as bot
from observability_aware_control import e2log, elog, integrator


@fn.partial(jax.jit, static_argnames=["elog_"])
def elog_cost(u0, u_const, x0, dt, elog_):
    u0 = u0.reshape(-1, 2)
    u0 = jnp.hstack([u_const, u0])
    return -jnp.linalg.norm(elog_(x0, u0, len(u0) * dt), ord=-2)


@fn.partial(jax.jit, static_argnames=["e2log_"])
def e2log_cost(u0, u_const, x0, dt, e2log_, observed_indices):
    u0 = u0.reshape(-1, 2)
    u0 = jnp.hstack([u_const, u0])
    s = jnp.ix_(observed_indices, observed_indices)
    return -jnp.linalg.norm(e2log_(x0, u0, len(u0) * dt)[s], ord=-2)


@fn.partial(jax.jit, static_argnames=["integrator"])
def destination(u0, u_const, x0, expected_xf, integrator):
    u0 = u0.reshape(-1, 2)
    u0 = jnp.hstack([u_const, u0])
    xs, _ = integrator(x0, u0)
    x_fin = xs[-1, :]
    return jnp.sum((x_fin - expected_xf) ** 2)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    gramians = {
        "elog": elog.ELOG(
            bot.dynamics,
            bot.observation,
            perturb_indices=jnp.array([3, 4]),
        ),
        "e2log": e2log.E2LOG(bot.dynamics, bot.observation, order=2),
    }

    n_steps = 600
    u0 = jnp.array([[1.0, 0.0, 1.0, 0.0]] * n_steps)

    x0 = jnp.array([0.5, 0.0, 0.0, 0.0, 5.0, 0.0])

    dt = 0.05
    solve_ode = integrator.Integrator(bot.dynamics, stepsize=dt)

    x_fin = solve_ode(x0, u0)[0][-1, ...]

    costs = {
        "elog": fn.partial(
            elog_cost,
            u_const=u0[:, 0:2],
            x0=x0,
            dt=dt,
            elog_=gramians["elog"],
        ),
        "e2log": fn.partial(
            e2log_cost,
            u_const=u0[:, 0:2],
            x0=x0,
            dt=dt,
            e2log_=gramians["e2log"],
            observed_indices=jnp.array((3, 4)),
        ),
    }

    constr = fn.partial(
        destination, u_const=u0[:, 0:2], x0=x0, expected_xf=x_fin, integrator=solve_ode
    )
    bnds = optimize.Bounds(
        lb=jnp.array([[0.0, -4.0]] * n_steps).ravel(),  # type: ignore
        ub=jnp.array([[8.0, 4.0]] * n_steps).ravel(),  # type: ignore
    )

    nlcon = optimize.NonlinearConstraint(constr, lb=0.0, ub=0.1, jac=jax.grad(constr))

    opts = {"verbose": 2, "disp": True, "maxiter": 400, "xtol": 0.01}
    minimize_fixture = fn.partial(
        optimize.minimize,
        bounds=bnds,
        constraints=nlcon,
        method="trust-constr",
        options=opts,
    )

    solns = {
        "e2log": minimize_fixture(
            costs["e2log"], u0[:, 2:4].ravel(), jac=jax.grad(costs["e2log"])
        ),
        "elog": minimize_fixture(
            costs["elog"], u0[:, 2:4].ravel(), jac=jax.grad(costs["elog"])
        ),
    }

    fig, ax = plt.subplots()
    for k, v in solns.items():
        u_final = jnp.hstack([u0[:, 0:2], jnp.reshape(v.x, (-1, 2))])
        xs_final, _ = solve_ode(x0, u_final)

        position = xs_final.reshape(len(xs_final), -1, 3)[..., 0:2]

        ax.plot(xs_final[:, 0], xs_final[:, 1])
        ax.plot(xs_final[:, 3], xs_final[:, 4])
    plt.show()


if __name__ == "__main__":
    main()
