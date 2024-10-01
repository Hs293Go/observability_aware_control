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

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla

from . import integrator, stlog, typing


class ObservabilityCostValue(NamedTuple):
    objective: jax.Array
    gramians: Optional[jax.Array] = None
    states: Optional[jax.Array] = None
    inputs: Optional[jax.Array] = None


def default_gramian_metric(gramians):
    sigmas = jla.svd(gramians, compute_uv=False, hermitian=True)
    return 1.0 / jnp.log(sigmas.min(axis=1).sum())


class ObservabilityCost:
    """Our observability cost function evaluates observability --- a metric of
    the local observability gramian, suitably approximated --- at each point
    along a model-prediction trajectory, then sums these observability metrics
    up to make for a scalar objective value.
    """

    def __init__(
        self,
        dynamics: typing.DynamicsFunction,
        observation: typing.ObservationFunction,
        integrator_dt: typing.ArrayLike,
        *,
        gramian=stlog.STLOG,
        integration_method=integrator.Methods.RK4,
        gramian_kw=None,
        gramian_metric=default_gramian_metric,
        observed_indices=(),
    ):

        self._solve_ode = integrator.Integrator(
            dynamics, integration_method, stepsize=integrator_dt
        )
        self._gramian_metric = gramian_metric
        if gramian_kw is not None:
            self._stlog = gramian(dynamics, observation, **gramian_kw)
        else:
            self._stlog = gramian(dynamics, observation)

        if observed_indices:
            # Create slice that would extract rows and columns indexed by
            # 'observed_indices'
            self._stlog_slice = jnp.ix_(observed_indices, observed_indices)
        else:
            # Take everything
            self._stlog_slice = ...

    def eval_gramian(self, x, u, dt):
        return self._stlog(x, u, dt)[self._stlog_slice]

    def __call__(
        self,
        x0,
        us,
        dt,
        return_trajectory=False,
        return_gramians=False,
    ) -> ObservabilityCostValue:
        """Evaluates observability cost given an initial state and a sequence of
        control inputs, evaluating a metric of the observability gramian at each
        point of the resultant state trajectory and summing up the results

        Parameters
        ----------
        x0 : Array
            The initial state
        us : Array
            A sequence (array) of control inputs
        dt : Array
            Observation horizon(s)
        return_trajectory : bool, optional
            Toggles if the predicted trajectory (shooting nodes) will be
            returned, by default False
        return_gramians : bool, optional
            Toggles if all the Gramians at each point on the predicted
            trajectory will be returned, by default False

        Returns
        -------
        ObservabilityCostValue
            A tuple consisting of (cost_value [, gramians [, states, inputs]]),
            where gramians, states, and inputs may be None if return_{gramians,
            trajectory} are not toggled on
        """

        xs, _ = self._solve_ode(x0, us)

        gramians = jax.vmap(self.eval_gramian)(xs, us, dt)

        objective = self._gramian_metric(gramians)

        if return_trajectory:
            return ObservabilityCostValue(objective, states=xs, inputs=us)

        if return_gramians:
            return ObservabilityCostValue(objective, gramians)

        if return_trajectory and return_gramians:
            return ObservabilityCostValue(objective, gramians, xs, us)

        return ObservabilityCostValue(objective)
