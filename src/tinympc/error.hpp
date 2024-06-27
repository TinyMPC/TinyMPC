#ifndef ERROR_HPP
#define ERROR_HPP
#include <cstdio>   
#include <cstdlib>  
#include <cstring> 

#if defined(__linux__) || defined(__unix__)// Check if Linux
#include <error.h>
#define ERROR_MSG(exit_code, format, ...) error(exit_code, errno, format, ##__VA_ARGS__)

#elif defined(__APPLE__) || defined(__MACH__) // Check if macOS
#include <errno.h>
#define ERROR_MSG(exit_code, format, ...) \
    do { \
        fprintf(stderr, format ": %s\n", ##__VA_ARGS__, strerror(errno)); \
        exit(exit_code); \
    } while (0)

#else
#error "Unsupported operating system"
#endif

#endif 
