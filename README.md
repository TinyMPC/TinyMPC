# TinyMPC: Model-Predictive Control on Resource-Constrained Microcontrollers

<p align="center">
<img align="center" src="https://github.com/TinyMPC/TinyMPC.github.io/blob/main/docs/media/tinympc-dark-logo.png" width=50%>
</p>

If you have any questions related to the solver, visit the [GitHub Discussions](https://github.com/orgs/TinyMPC/discussions) page. This guarantees accessibility for everyone. 

The documentation is available at [tinympc.org](https://tinympc.org/)

High-level language interfaces with detailed examples and code generation are available in [Python](https://github.com/TinyMPC/tinympc-python), [Julia](https://github.com/TinyMPC/tinympc-julia), and [MATLAB](https://github.com/TinyMPC/tinympc-matlab).

**Excited to hear your TinyMPC success stories, share them with us!**

## Building
For Windows - Enable ```wsl``` before following prompts below

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

* Put a star ⭐ on this repository.
  
* Cite the related papers:
```
@inproceedings{tinympc,
      title={TinyMPC: Model-Predictive Control on Resource-Constrained Microcontrollers}, 
      author={Khai Nguyen and Sam Schoedel and Anoushka Alavilli and Brian Plancher and Zachary Manchester},
      booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
      year={2024},
}
```
```
@inproceedings{tinympc-adaptive,
      title={Robust and Efficient Embedded Convex Optimization through First-Order Adaptive Caching}, 
      author={Ishaan Mahajan and Brian Plancher},
      booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      year={2025}
}
```
```
@inproceedings{conic-tinympc,
      title={Code Generation and Conic Constraints for Model-Predictive Control on Microcontrollers with Conic-TinyMPC}, 
      author={Ishaan Mahajan and Khai Nguyen and Sam Schoedel and Elakhya Nedumaran and Moises Mata and Brian Plancher and Zachary Manchester},
      year={2026},
      booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
}
```


## Bug reports and support

Please report any issues via the [Github issue tracker](https://github.com/TinyMPC/TinyMPC/issues). All types of issues are welcome including bug reports, documentation typos, feature requests and so on.

## Running on MCUs

Numerical benchmarks against other solvers on MCUs are available [here](https://github.com/RoboticExplorationLab/mcu-solver-benchmarks).

## TinyMPC on the Crazyflie

TinyMPC-integrated firmware on the Crazyflie nano-quadrotor is available [here](https://github.com/RoboticExplorationLab/tinympc-crazyflie-firmware).

## TinyMPC-AL

An earlier version (not optimized) which can deal with nonlinear dynamics (e.g., bicycle models) is available [here](https://github.com/RoboticExplorationLab/TinyMPC-AL).

## Notes
