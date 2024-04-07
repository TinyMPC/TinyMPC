# TinyMPC: Model-predictive control on resource-constrained microcontrollers

<img src="https://github.com/TinyMPC/TinyMPC.github.io/blob/main/docs/media/lightmode-banner.png" width=50%>

If you have any questions related to the solver, visit the [GitHub Discussions](https://github.com/orgs/TinyMPC/discussions) page. This guarantees accessibility for everyone.

The documentation is available at [tinympc.org](https://tinympc.org/)

High-level language interfaces with detailed examples and code generation are available in [Python](https://github.com/TinyMPC/tinympc-python), [Julia](https://github.com/TinyMPC/tinympc-julia), and [MATLAB](https://github.com/TinyMPC/tinympc-matlab).

## Building

1. On terminal, clone this repo

```bash
git clone https://github.com/TinyMPC/TinyMPC.git
```

2. Navigate to root directory and run

```bash
cd TinyMPC && mkdir build && cd build
```

3. Run CMake configure step

```bash
cmake ..
```

4. Build TinyMPC

```bash
cmake --build .
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

* Cite the related papers:
```
@inproceedings{tinympc,
      title={TinyMPC: Model-Predictive Control on Resource-Constrained Microcontrollers}, 
      author={Khai Nguyen and Sam Schoedel and Anoushka Alavilli and Brian Plancher and Zachary Manchester},
      booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
      year = {2024}
}
```
```
@misc{tinympc2,
      title={Code Generation for Conic Model-Predictive Control on Microcontrollers with TinyMPC}, 
      author={Sam Schoedel and Khai Nguyen and Elakhya Nedumaran and Brian Plancher and Zachary Manchester},
      year={2024},
      eprint={2403.18149},
      archivePrefix={arXiv},
}

```
* Put a star ⭐ on this repository.

**Excited to hear your TinyMPC success stories—share them with us!**

## Bug reports and support

Please report any issues via the [Github issue tracker](https://github.com/TinyMPC/TinyMPC/issues). All types of issues are welcome including bug reports, documentation typos, feature requests and so on.

## Running on MCUs

Numerical benchmarks against other solvers on MCUs are available [here](https://github.com/RoboticExplorationLab/mcu-solver-benchmarks).

## Notes
