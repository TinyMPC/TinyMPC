# TinyMPC

Model-predictive control on resource-constrained microcontrollers

If you have any questions related to the solver, visit the [GitHub Discussions](https://github.com/orgs/TinyMPC/discussions) page. This guarantees accessibility for everyone.

The documentation is available at [tinympc.org](https://tinympc.org/)

## Building

1. On terminal, clone this repo

```bash
git clone https://github.com/TinyMPC/TinyMPC.git
```

2. Navigate to root directory and run

```bash
mkdir build && cd build
```

3. Run CMake configure step

```bash
cmake ../
```

4. Build TinyMPC

```bash
make 
```

## Examples

* Run the `quadrotor_hovering` example

```bash
./examples/quadrotor_hovering
```

* Run the `codegen_cartpole` example then follow the same building steps inside that directory

```bash
./examples/codegen_cartpole
```

## Citing TinyMPC

If you are using TinyMPC, we encourage you to

* [Cite the related papers](https://tinympc.org/docs/citing/),
* Put a star on this repository.

**Excited to hear your TinyMPC success stories—share them with us!**

## Bug reports and support

Please report any issues via the [Github issue tracker](https://github.com/TinyMPC/TinyMPC/issues). All types of issues are welcome including bug reports, documentation typos, feature requests and so on.

## Running on MCUs

Numerical benchmarks against other solvers on MCUs are available [here](https://github.com/RoboticExplorationLab/mcu-solver-benchmarks).

## Notes
