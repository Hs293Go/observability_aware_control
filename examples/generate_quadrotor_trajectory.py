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

import jax
import jax.numpy as jnp
import minsnap_trajectories as ms
import numpy as np

from example_lib.misc import geometric_controller
from example_lib.models import quadrotor
from observability_aware_control import integrator


def generate_leader_trajectory(timestamps, waypoints, t_sample, quadrotor_mass, ctl_dt):

    polys = ms.generate_trajectory(
        [ms.Waypoint(float(t), p) for t, p in zip(timestamps, waypoints)],
        5,
        idx_minimized_orders=[2, 3],
    )

    traj = ms.compute_quadrotor_trajectory(
        polys, t_sample, quadrotor_mass, yaw="velocity"
    )

    pc = geometric_controller.TrackingController(
        geometric_controller.TrackingControllerParams(
            k_pos=jnp.array([0.8, 0.8, 0.9]),
            k_vel=jnp.array([0.4, 0.4, 0.6]),
            max_z_accel=jnp.inf,
        )
    )
    ac = geometric_controller.AttitudeController(0.25)

    quad = integrator.Integrator(
        quadrotor.dynamics, integrator.Methods.EULER, stepsize=ctl_dt
    )

    @jax.jit
    def loop(state, tup):
        position, attitude, velocity, *_ = jnp.split(state, (3, 7))
        pc_out, _ = pc.run(pc.State(position, attitude, velocity), pc.Reference(*tup))
        ac_out, _ = ac.run(ac.State(attitude), ac.Reference(pc_out.orientation))
        u = jnp.concatenate(
            [jnp.array([pc_out.thrust / quadrotor_mass]), ac_out.body_rate]
        )

        state, _ = quad(state, u)
        return state, (state, u)

    state_in = traj.state[0, :]
    _, (x_leader, u_leader) = jax.lax.scan(
        loop, state_in, (traj.position, traj.velocity)
    )

    return x_leader, u_leader


def generate_trajectory(cfg):
    timestamps = cfg["timestamps"]
    waypoints = np.asarray(cfg["waypoints"])
    if len(timestamps) != len(waypoints):
        raise RuntimeError("Mismatch in number of timestamps and waypoints")

    dt = cfg["timestep"]
    t_refs = np.arange(0, timestamps[-1], dt)
    quadrotor_mass = cfg.get("model_mass", 1.0)
    x_leader, u_leader = generate_leader_trajectory(
        timestamps, waypoints, t_refs, quadrotor_mass, dt
    )

    return dt, t_refs, x_leader, u_leader
