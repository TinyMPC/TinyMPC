#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// #if defined(__linux__) || defined(__unix__)// Check if Linux
// #include <error.h>
// #define ERROR_MSG(exit_code, format, ...) error(exit_code, errno, format, ##__VA_ARGS__)

// #elif defined(__APPLE__) || defined(__MACH__) // Check if macOS
#define ERROR_MSG(exit_code, format, ...) \
        { \
        fprintf(stderr, format ": %s\n", ##__VA_ARGS__, strerror(errno)); \
        exit(exit_code); \
        }

// #else
// #error "Unsupported operating system"
// #endif

#ifdef __cplusplus
}
#endif