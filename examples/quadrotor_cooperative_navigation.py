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

import sys
import warnings

import jax
import jax.experimental.compilation_cache.compilation_cache as cc
import matplotlib.pyplot as plt
import numpy as np
import tomllib
import tqdm
from generate_quadrotor_trajectory import generate_trajectory

import example_lib.models.inter_quadrotor_pose as mdl
from example_lib import math
from observability_aware_control import (
    integrator,
    observability_aware_controller,
    observability_cost,
    utils,
)

cc.set_cache_dir("./.cache")

jax.config.update("jax_enable_x64", True)


u_eqm = np.r_[9.81, 0.0, 0.0, 0.0]
q_eqm = np.r_[np.zeros(3), 1.0]
v_eqm = np.zeros(3)

N_ROBOTS = 2


warnings.filterwarnings("ignore", category=UserWarning)


def main():
    with open("./config/quadrotor_control_experiment.toml", "rb") as fp:
        cfg = tomllib.load(fp)

    n_robots = cfg["n_robots"]

    # -----------------------Generate initial trajectory------------------------
    dt, t_refs, x_leader, u_leader = generate_trajectory(cfg["leader"]["trajectory"])

    sim_steps = min(cfg["sim"]["steps"], len(t_refs))
    time = t_refs[0 : sim_steps + 1]
    x = np.zeros((sim_steps, mdl.NUM_STATES))
    u = np.zeros((sim_steps, mdl.NUM_INPUTS))

    # pos_var = np.full(3, 1e-2)
    range_var = 1e-2
    att_var = np.full(4, 1e-2)
    vel_var = np.full(3, 1e-2)
    # var = np.concatenate([pos_var, att_var, range_var, vel_var])
    var = np.concatenate(
        [
            np.array([range_var]),
            att_var,
            # vel_var,
        ]
    )
    cost = observability_cost.ObservabilityCost(
        mdl.dynamics,
        mdl.observation,
        dt,
        gramian_kw={"order": cfg["stlog"]["order"], "var": var},
        integration_method=integrator.Methods.EULER,
        observed_indices=cfg["opc"].get("observed_components", ()),
    )

    window = cfg["opc"]["window_size"]
    try:
        u_lb = np.tile(np.array(cfg["optim"]["lb"]), (window, N_ROBOTS))
        u_ub = np.tile(np.array(cfg["optim"]["ub"]), (window, N_ROBOTS))
    except KeyError:
        u_lb = -np.inf
        u_ub = np.inf
    min_problem = observability_aware_controller.ObservabilityAwareController(
        cost,
        lb=u_lb,
        ub=u_ub,
        method=cfg["optim"]["method"],
        optim_options=cfg["optim"]["options"],
        constraint=mdl.interrobot_distance_squared,
        constraint_bounds=(
            cfg["opc"]["min_inter_vehicle_distance"] ** 2,
            cfg["opc"]["max_inter_vehicle_distance"] ** 2,
        ),
    )

    # -----------------Setup initial conditions and data saving-----------------

    x[0, :] = np.concatenate(
        [
            np.array([0.0, -1.0, 1.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.zeros(3),
        ]
    )
    # np.concatenate(
    # [x_leader[0, :], np.asarray(cfg["followers"]["init_state"]).ravel()]
    # )

    input_steps = u_leader.shape[0]
    if sim_steps + window > input_steps:
        u_leader_tmp = np.array(u_leader)
        u_leader = np.zeros((sim_steps + window, u_leader.shape[1]))
        u_leader[:input_steps, :] = u_leader_tmp
        u_leader[input_steps:, :] = u_leader[-1, :]

    u0 = np.r_[u_leader[0, :], np.tile(u_eqm, N_ROBOTS - 1)]
    u[0, :] = u0
    dx = np.zeros_like(x)

    soln_stats = {
        "status": [],
        "nit": [],
        "fun_hist": [],
        "execution_time": [],
        "constr_violation": [],
        "optimality": [],
    }

    # ----------------------------Run the Simulation----------------------------
    success = False
    sim = jax.jit(
        integrator.Integrator(mdl.dynamics, integrator.Methods.EULER, stepsize=dt)
    )

    inputs_all = []
    try:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        anim = utils.animation.AnimatedRobotTrajectory(
            fig, ax, dims=3, animation_kws={"interval": 50, "save_count": 100}
        )

        fig.show()
        p_f = []
        with plt.ion():
            for i in tqdm.trange(1, sim_steps):
                u_leader_0 = u_leader[i : i + window, :]

                u0 = np.hstack([u_leader_0, np.tile(u_eqm, (window, N_ROBOTS - 1))])
                soln = min_problem.minimize(
                    x[i - 1, :], u0, cfg["stlog"]["dt"], (4, 5, 6, 7)
                )
                soln_u = np.concatenate([u_leader[i, :], soln.x[0, 4:]])
                inputs_all.append(soln.x)
                u[i, :] = soln_u
                x[i, :], dx[i, :] = sim(x[i - 1, :], soln_u)
                x[i, 3:7] /= np.linalg.norm(x[i, 3:7])
                fun = soln.fun
                status = soln.get("status", -1)
                nit = soln.get("nit", np.nan)
                execution_time = soln.get("execution_time", np.nan)
                constr_violation = float(soln.get("constr_violation", np.nan))
                optimality = soln.get("optimality", np.nan)

                soln_stats["status"].append(status)
                soln_stats["nit"].append(nit)
                soln_stats["execution_time"].append(execution_time)
                soln_stats["constr_violation"].append(constr_violation)
                soln_stats["optimality"].append(optimality)

                fun_hist = np.full(cfg["optim"]["options"]["maxiter"], fun)
                fun_hist[0 : len(soln.fun_hist)] = np.asarray(soln.fun_hist)
                soln_stats["fun_hist"].append(fun_hist)

                x_abs = mdl.to_absolute_state(x_leader[i, :], x[i, :])

                p_f.append(x_abs[10:13])
                plt_p = np.stack([x_leader[:i, 0:3], np.array(p_f)])
                anim.set_data(plt_p[..., 0], plt_p[..., 1], plt_p[..., 2])
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            success = True
    finally:  # Save the data at all costs
        soln_stats = {k: np.asarray(v) for k, v in soln_stats.items()}
        save_name = str(
            cfg["session"].get("save_name", "optimization_results_extra_large.npz")
        )
        if not success:
            save_name = save_name.replace(".npz", ".failed.npz")
        np.savez(
            save_name,
            states=x,
            inputs=u,
            inputs_all=np.array(inputs_all),
            derivatives=dx,
            time=time,
            **soln_stats,
        )
    plt.show()


if __name__ == "__main__":
    main()
