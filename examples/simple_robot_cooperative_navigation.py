import functools
import logging
import pathlib
import tomllib

import jax
import jax.experimental.compilation_cache.compilation_cache as cc
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import tqdm

from example_lib.misc import simple_ekf
from example_lib.models import leader_follower_robots
import example_lib.visualization.visualization as viz
from observability_aware_control import (
    integrator,
    observability_aware_controller,
    observability_cost,
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
    logging.getLogger().addHandler(rr.LoggingHandler("/logs/handler"))
    logging.getLogger().setLevel(logging.INFO)
    rr.set_time("/time", duration=0.0)

    leader_trace = viz.PositionTrace("/sim/leader")
    leader = viz.PoseReferenceFrame("/sim/leader")
    follower_trace = viz.PositionTrace("/sim/follower")
    follower = viz.PoseReferenceFrame("/sim/follower")
    leader.set_pose(np.append(x[0, 0:2], 0.0), PureYaw(angle=x[0, 2]))
    follower.set_pose(np.append(x[0, 3:5], 0.0), PureYaw(angle=x[0, 5]))

    for ax, c in zip(["x", "y", "theta"], np.eye(3), strict=True):
        rr.log(
            f"/graphs/estimation/{ax}/variance",
            rr.SeriesLines(
                names=f"3-sigma confidence bound on {ax}", colors=c, widths=2
            ),
        )

    rr.send_blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/sim"),
            rrb.Vertical(
                rrb.TextLogView(origin="/logs"),
                *(
                    rrb.TimeSeriesView(
                        origin=f"/graphs/estimation/{ax}",
                        name=f"Estimation Performance in {ax}",
                    )
                    for ax in ["x", "y", "theta"]
                ),
                row_shares=[0.5, 1, 1, 1],
            ),
        )
    )

    # ----------------------------Run the Simulation----------------------------
    success = False
    sim = jax.jit(
        integrator.Integrator(
            leader_follower_robots.dynamics, integrator.Methods.RK4, stepsize=dt
        )
    )

    input_var = np.tile(np.array([1e-2, 1e-2]), 2)
    ekf = simple_ekf.SimpleEKF(
        lambda x, u, dt: x + dt * leader_follower_robots.dynamics(x, u),
        lambda x: leader_follower_robots.observation(x, 0),
        in_cov=np.diag(input_var),
        obs_cov=np.diag(var),
    )
    ekf_cov = np.eye(x.shape[1]) * 1e-3

    rng = np.random.default_rng()
    try:
        for i in tqdm.trange(1, sim_steps):
            u_refs = np.array([u[i - 1, :]] * window)
            x_op = x[i - 1, :]
            x_op, ekf_cov = ekf.predict(x_op, ekf_cov, u[i - 1, :], dt)

            feedback = leader_follower_robots.observation(
                x_op, u[i - 1, :]
            ) + rng.normal(0.0, np.sqrt(var), size=var.shape)

            x_op, ekf_cov = ekf.update(x_op, ekf_cov, feedback)
            three_sigmas = 3 * np.sqrt(np.diag(ekf_cov))
            for j, ax in enumerate(["x", "y", "theta"]):
                rr.log(
                    f"/graphs/estimation/{ax}/variance", rr.Scalars(three_sigmas[3 + j])
                )

            if np.any(np.isnan(x_op)):
                raise ValueError("EKF diverged")
            soln = min_problem.minimize(
                x_op, u_refs, cfg["stlog"]["dt"], minimized_indices=(2, 3)
            )
            soln_u = soln.x[0, :]
            soln_u = soln_u.at[3:6].add(
                rng.normal(0.0, np.sqrt(input_var[3:6]))
            )  # Input noise to follower
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
