#pragma once
#include <cstdint>
#include "types.hpp"

struct RhoAdapter {
    tinytype rho_min;
    tinytype rho_max;
    bool clip;
};

struct RhoBenchmarkResult {
    uint32_t time_us;
    tinytype initial_rho;
    tinytype final_rho;
    tinytype pri_res;
    tinytype dual_res;
};

void benchmark_rho_adaptation(
    const tinytype* x_k,
    const tinytype* u_k,
    const tinytype* v_k,
    tinytype pri_res,
    tinytype dual_res,
    RhoBenchmarkResult* result,
    RhoAdapter* adapter,
    tinytype current_rho
);