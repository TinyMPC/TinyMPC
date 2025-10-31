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

// SDP projection toggles (augmented formulations)
#define TINY_DEFAULT_EN_STATE_SDP       (0)
#define TINY_DEFAULT_EN_INPUT_SDP       (0)

// Forward rollout blending between slack projections and LQR
#define TINY_DEFAULT_FORWARD_BLEND_ALPHA (0.9)
