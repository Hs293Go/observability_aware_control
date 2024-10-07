from typing import Any, Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from scipy import optimize

from observability_aware_control import utils

from . import observability_cost, utils


def _recombine_input(u, u_const, id_mut, id_const):
    u = u.reshape(len(u_const), -1)
    return utils.combine_array(u, u_const, id_mut, id_const)


class ObservabilityAwareController:
    def __init__(
        self,
        cost: observability_cost.ObservabilityCost,
        lb: ArrayLike = jnp.inf,
        ub: ArrayLike = -jnp.inf,
        method: Optional[utils.optim_utils.Method] = None,
        optim_options: Optional[Dict[str, Any]] = None,
        constraint=None,
        constraint_bounds=None,
    ):

        self._cost = cost

        self._gradient = jax.grad(lambda us, x, dt: self._cost(x, us, dt).objective)

        self._problem = utils.optim_utils.MinimizeProblem(
            self.objective,
            jac=self.gradient,
            method=method,
            options=optim_options,
        )

        self._lb = jnp.asarray(lb)
        self._ub = jnp.asarray(ub)

        if constraint is not None and constraint_bounds is not None:
            self._constraint = constraint
            self._constraint_lb, self._constraint_ub = constraint_bounds
            self._constraint_jacobian = jax.jacobian(
                lambda us, x, *a, **kw: constraint(x, us, *a, **kw)
            )

    @eqx.filter_jit
    def objective(self, u, u_const, x, dt, minimized_indices, constant_indices):
        u = _recombine_input(u, u_const, minimized_indices, constant_indices)
        return self._cost(x, u, dt).objective

    @eqx.filter_jit
    def gradient(self, u, u_const, x, dt, minimized_indices, constant_indices):
        u = _recombine_input(u, u_const, minimized_indices, constant_indices)
        return self._gradient(u, x, dt)[..., minimized_indices].ravel()

    @eqx.filter_jit
    def constraint(self, u, u_const, x, _, minimized_indices, constant_indices):
        assert (
            self._constraint is not None
        ), "This problem is not configured to have constraints"

        u = _recombine_input(u, u_const, minimized_indices, constant_indices)
        return self._constraint(x, u, cost=self._cost)

    @eqx.filter_jit
    def constraint_jacobian(
        self, u, u_const, x, _, minimized_indices, constant_indices
    ):
        assert (
            self._constraint is not None
        ), "This problem is not configured to have constraints"
        u = _recombine_input(u, u_const, minimized_indices, constant_indices)
        jac = self._constraint_jacobian(u, x, cost=self._cost)
        return jac[..., minimized_indices].reshape(jac.shape[0], -1)

    def minimize(self, x0, u0, t, minimized_indices=()) -> optimize.OptimizeResult:
        num_steps, num_inputs = u0.shape
        if not minimized_indices:
            minimized_indices = jnp.arange(num_inputs)
        else:
            minimized_indices = jnp.asarray(minimized_indices)

        constant_indices = jnp.setdiff1d(jnp.arange(u0.shape[1]), minimized_indices)

        self._lb, self._ub, u0 = jnp.broadcast_arrays(self._lb, self._ub, u0)
        # Broadcast bounds over all time windows, takes mutable components, then flatten
        self._problem.bounds = zip(
            *(it[..., minimized_indices].ravel() for it in (self._lb, self._ub))
        )  # type: ignore

        u_const = u0[..., constant_indices]
        self._problem.x0 = u0[..., minimized_indices].ravel()
        args = (u_const, x0, t, minimized_indices, constant_indices)
        self._problem.args = args

        if self._constraint is not None:
            lb = jnp.broadcast_to(self._constraint_lb, num_steps)
            ub = jnp.broadcast_to(self._constraint_ub, num_steps)
            self._problem.constraints = optimize.NonlinearConstraint(
                lambda u: self.constraint(u, *args),
                lb=lb,
                ub=ub,
                jac=lambda u: self.constraint_jacobian(u, *args),  # type: ignore
            )

        rec = utils.optim_utils.OptimizationRecorder()
        self._problem.callback = rec.update

        prob_dict = vars(self._problem)
        soln = optimize.minimize(**prob_dict)

        soln["x"] = _recombine_input(
            soln.x, u_const, minimized_indices, constant_indices
        )
        soln["fun_hist"] = rec.fun
        return soln
