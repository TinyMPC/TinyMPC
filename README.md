# TinyMPC

Model-predictive control on resource-constrained microcontrollers

## Building on Ubuntu

1. On terminal, clone this repo

```bash
git clone git@github.com:TinyMPC/TinyMPC.git
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

* Run the quadrotor hovering example

```bash
./examples/example_quadrotor_hovering
```

* Run the codegen example then follow the same building steps inside that directory

```bash
./examples/example_codegen
```

## Notes

* You should customize your own Eigen compile flags based on your applications.
