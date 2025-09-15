# Observability Aware Control

This repository implements the algorithms in our paper **Observability-Aware
Control for Cooperatively Localizing Multirotor Vehicles**.

```bibtex
@unpublished{
  title={observability2024go},
  author={Go, H S Helson, Chong, Ching Lok, and Liu, H.-T},
  journal={Journal of Guidance, Control, and Dynamics},
  year={2025},
}
```

> We built upon _Observability-Aware Trajectory Optimization_ techniques
> (Hausmann et. al. [2017](https://ieeexplore.ieee.org/document/7805145), Grebe
> and Kelly., [2022](https://hdl.handle.net/1807/110811)), but we are the first
> to make our implementation public!

## Prerequisites

We use the `uv` package manager to manage dependencies and run our programs. To
install `uv`, please follow the instructions:

We use the `pipx` application manager to install `uv` in an isolated
environment. On debian-based systems, you can install `pipx` using the following
command:

```bash
sudo apt-get install pipx
```

Then, install `uv` using `pipx`:

```bash
pipx install uv
```

## Running the code

Finally, you can begin to run our scripts using `uv run <script_name>`.

To run our simple robot cooperative navigation example, run:

```bash
uv run examples/simple_robot_cooperative_navigation.py
```

To run our quadrotor cooperative navigation example, run:

```bash
uv run examples/quadrotor_cooperative_navigation.py
```

## Troubleshooting

Unless you install this package, you need to put `src` on the `PYTHONPATH` for
the module `observability_aware_control` to be found. This can be done by
running the following command at the project root:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

Furthermore,

We use the `jax[cuda12]` variant of `jax`, which contains its own set of CUDA
libraries. This may conflict with other CUDA installations on your system. Clear
`LD_LIBRARY_PATH` to make system CUDA libraries invisible to `jax`.

```bash
unset LD_LIBRARY_PATH
```

> [!NOTE]
>
> If you installed `direnv` and enabled its `load_dotenv` option, inspect our
> `.env` then run `direnv allow` to automatically set the `PYTHONPATH` and
> `LD_LIBRARY_PATH` when you `cd` into this directory.
