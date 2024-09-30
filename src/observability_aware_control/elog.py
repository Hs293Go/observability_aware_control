import jax
import jax.numpy as jnp

from . import integrator


class ELOG:

    def __init__(
        self,
        dynamics,
        observation,
        eps=1e-3,
        perturb_axes=None,
        method=integrator.Methods.RK4,
    ):
        self._solve_ode = integrator.Integrator(dynamics, method)
        self._observation = jax.vmap(observation)
        self._eps = eps
        self._perturb_axes = perturb_axes
        self._method = method

    def __call__(self, x0, u, dt):

        dt = jnp.broadcast_to(dt, u.shape[0])

        def _perturb(x0_p, x0_m):
            yi_plus, _ = self._observation(self._solve_ode(x0_p, u, dt))
            yi_minus, _ = self._observation(self._solve_ode(x0_m, u, dt))
            return yi_plus - yi_minus

        perturb_bases = (
            jnp.eye(len(x0))[self._perturb_axes]
            if self._perturb_axes is not None
            else jnp.eye(len(x0))
        )
        x0_plus = x0 + self._eps * perturb_bases
        x0_minus = x0 - self._eps * perturb_bases
        y_all = jax.vmap(_perturb, out_axes=2)(x0_plus, x0_minus) / (2.0 * self._eps)

        dt = dt[..., None, None, None]

        return jnp.sum(dt * y_all.mT * y_all, axis=(0, 1))
