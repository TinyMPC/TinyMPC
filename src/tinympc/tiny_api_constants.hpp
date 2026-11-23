#pragma once


// Default settings
#define TINY_DEFAULT_ABS_PRI_TOL        (1e-03)
#define TINY_DEFAULT_ABS_DUA_TOL        (1e-03)
#define TINY_DEFAULT_MAX_ITER           (1000)
#define TINY_DEFAULT_CHECK_TERMINATION  (1)
#define TINY_DEFAULT_EN_STATE_BOUND     (1)
#define TINY_DEFAULT_EN_INPUT_BOUND     (1)
#define TINY_DEFAULT_EN_STATE_SOC       (0)
#define TINY_DEFAULT_EN_INPUT_SOC       (0)
#define TINY_DEFAULT_EN_STATE_LINEAR    (0)
#define TINY_DEFAULT_EN_INPUT_LINEAR    (0)
#define TINY_DEFAULT_EN_TV_STATE_LINEAR (0)
#define TINY_DEFAULT_EN_TV_INPUT_LINEAR (0)

// Adaptive rho: the first-order Taylor cache update is only accurate near the rho
// the sensitivities were computed at, so by default rho is kept in a band around it
// (cf. [60, 100] around 85 in the FoAC paper, arXiv:2507.03231)
#define TINY_ADAPTIVE_RHO_MIN_SCALE     (0.7)
#define TINY_ADAPTIVE_RHO_MAX_SCALE     (1.2)
