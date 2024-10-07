import functools
import tomllib

import jax
import jax.experimental.compilation_cache.compilation_cache as cc
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from example_lib.models import leader_follower_robots
from observability_aware_control import (
    integrator,
    observability_aware_controller,
    observability_cost,
    utils,
)

cc.set_cache_dir("./.cache")

jax.config.update("jax_enable_x64", True)


u_eqm = np.zeros(2)


def main():
    with open("./config/planar_robot_control_experiment.toml", "rb") as fp:
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
        gramian_metric=functools.partial(
            observability_cost.default_gramian_metric, log_scale=False
        ),
    )

    min_problem = observability_aware_controller.ObservabilityAwareController(
        cost,
        lb=u_lb,
        ub=u_ub,
        method=cfg["optim"]["method"],
        optim_options=cfg["optim"]["options"],
        constraint=leader_follower_robots.interrobot_distance,
        constraint_bounds=(
            cfg["opc"]["min_inter_vehicle_distance"],
            cfg["opc"]["max_inter_vehicle_distance"],
        ),
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

    # ----------------------------Run the Simulation----------------------------
    success = False
    sim = jax.jit(
        integrator.Integrator(
            leader_follower_robots.dynamics, integrator.Methods.RK4, stepsize=dt
        )
    )

    try:
        fig, ax = plt.subplots()
        anim = utils.animation.AnimatedRobotTrajectory(
            fig, ax, animation_kws={"interval": 50, "save_count": 100}
        )
        with plt.ion():
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

                anim.annotation = (
                    f"nit: {nit} f(x): {fun:.4}\n $\\Delta$ f(x):"
                    f" {(fun - fun_hist[0]):4g}\nOptimality:"
                    f" {optimality:.4}\nviolation: {constr_violation:.4}"
                )
                plt_x, plt_y, *_ = np.dsplit(
                    np.moveaxis(np.reshape(x[0:i, :], (i, -1, 3))[..., 0:2], 1, 0), 2
                )

                anim.set_data(plt_x, plt_y)

                plt.pause(1e-3)
            success = True
    finally:  # Save the data at all costs
        anim.anime.save(cfg["session"].get("video_name", "optimization.mp4"))
        soln_stats = {k: np.asarray(v) for k, v in soln_stats.items()}
        save_name = str(cfg["session"].get("save_name", "optimization_results.npz"))
        if not success:
            save_name = save_name.replace(".npz", ".failed.npz")
        np.savez(save_name, states=x, inputs=u, time=time, **soln_stats)

    figs = {}
    figs[0], ax = plt.subplots()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    plt_data = np.reshape(x, (-1, n_robots, 3))
    for idx in range(n_robots):  # Vary vehicles
        ax.plot(plt_data[:, idx, 0], plt_data[:, idx, 1], f"C{idx}")

    plt.show()


if __name__ == "__main__":
    main()
