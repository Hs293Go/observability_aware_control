import functools
import logging
import pathlib
import tomllib

import jax
import jax.experimental.compilation_cache.compilation_cache as cc
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import tqdm

from example_lib.models import leader_follower_robots
import example_lib.visualization.visualization as viz
from observability_aware_control import (
    integrator,
    observability_aware_controller,
    observability_cost,
    utils,
)

cc.set_cache_dir("./.cache")

jax.config.update("jax_enable_x64", True)


u_eqm = np.zeros(2)

SCRIPT_PATH = pathlib.Path(__file__)
CONFIG_DIR = SCRIPT_PATH.parent.parent / "config"

PureYaw = functools.partial(rr.RotationAxisAngle, axis=[0, 0, 1])


def main():
    with (CONFIG_DIR / "planar_robot_control_experiment.toml").open("rb") as fp:
        cfg = tomllib.load(fp)

    n_robots = cfg["model"]["n_robots"]

    position_var = np.full(2, 1e-1)
    bearing_var = 1e-2
    heading_var = np.full(n_robots, 1e-2)
    var = np.r_[position_var, heading_var, bearing_var]

    window = cfg["opc"]["window_size"]
    u_lb = np.array([cfg["optim"]["lb"]] * window)
    u_ub = np.array([cfg["optim"]["ub"]] * window)

    # -----------------------Generate initial trajectory------------------------
    data = np.load("./config/leader_follower_robot_path.npz")

    u_refs_full = np.hstack([data["leader_inputs"], data["follower_inputs"]])
    t_refs = data["time"]
    dt = t_refs[1] - t_refs[0]

    # -----------------Setup initial conditions and data saving-----------------
    sim_steps = min(cfg["sim"]["steps"], len(t_refs))
    time = t_refs[0:sim_steps]
    x = np.zeros((sim_steps, n_robots * leader_follower_robots.NUM_STATES))
    u = np.zeros((sim_steps, n_robots * leader_follower_robots.NUM_INPUTS))

    cost = observability_cost.ObservabilityCost(
        leader_follower_robots.dynamics,
        leader_follower_robots.observation,
        dt,
        gramian_kw={"order": cfg["stlog"]["order"], "var": var},
        observed_indices=cfg["opc"].get("observed_components", ()),
    )

    min_problem = observability_aware_controller.ObservabilityAwareController(
        cost,
        lb=u_lb,
        ub=u_ub,
        method=cfg["optim"]["method"],
        optim_options=cfg["optim"]["options"],
    )

    x[0, :] = np.concatenate(cfg["session"]["initial_positions"])
    u[0, :] = u_refs_full[0, :]

    soln_stats = {
        "status": [],
        "nit": [],
        "fun_hist": [],
        "execution_time": [],
        "constr_violation": [],
        "optimality": [],
    }
    rr.init("quadrotor_control_experiment", spawn=True)
    logging.getLogger().addHandler(rr.LoggingHandler("logs/handler"))
    logging.getLogger().setLevel(logging.INFO)
    rr.set_time("/time", duration=0.0)

    leader_trace = viz.PositionTrace("/leader")
    leader = viz.PoseReferenceFrame("/leader")
    follower_trace = viz.PositionTrace("/follower")
    follower = viz.PoseReferenceFrame("/follower")
    leader.set_pose(np.append(x[0, 0:2], 0.0), PureYaw(angle=x[0, 2]))
    follower.set_pose(np.append(x[0, 3:5], 0.0), PureYaw(angle=x[0, 5]))

    # ----------------------------Run the Simulation----------------------------
    success = False
    sim = jax.jit(
        integrator.Integrator(
            leader_follower_robots.dynamics, integrator.Methods.RK4, stepsize=dt
        )
    )

    try:
        for i in tqdm.trange(1, sim_steps):
            u_refs = np.array([u[i - 1, :]] * window)
            soln = min_problem.minimize(
                x[i - 1, :], u_refs, cfg["stlog"]["dt"], minimized_indices=(2, 3)
            )
            soln_u = soln.x[0, :]
            u[i, :] = soln_u
            x[i, :], _ = sim(x[i - 1, :], soln_u)

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

            logging.info(
                "nit: %d f(x): %.4g from %.4g, optim.: %.4g, viol.: %.4g",
                nit,
                fun,
                fun_hist[0],
                optimality,
                constr_violation,
            )

            rr.set_time("/time", duration=time[i])
            leader_position = np.append(x[i, 0:2], 0.0)
            leader_yaw = x[i, 2]
            leader.set_pose(leader_position, PureYaw(angle=leader_yaw))
            leader_trace.add_position(leader_position)
            follower_position = np.append(x[i, 3:5], 0.0)
            follower_yaw = x[i, 5]
            follower.set_pose(follower_position, PureYaw(angle=follower_yaw))
            follower_trace.add_position(follower_position)

    finally:  # Save the data at all costs
        soln_stats = {k: np.asarray(v) for k, v in soln_stats.items()}
        save_name = str(cfg["session"].get("save_name", "optimization_results.npz"))
        if not success:
            save_name = save_name.replace(".npz", ".failed.npz")
        np.savez(
            save_name, states=x, inputs=u, time=time, allow_pickle=True, **soln_stats
        )


if __name__ == "__main__":
    main()
