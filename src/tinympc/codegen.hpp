#pragma once

#include "types.hpp"

#ifdef __cplusplus
extern "C" {
#endif

    int tiny_codegen(TinySolver* solver, const char* output_dir, int verbose);
    
    int codegen_create_directories(const char* output_dir, int verbose);
    int codegen_data_header(const char* output_dir, int verbose);
    int codegen_data_source(TinySolver* solver, const char* output_dir, int verbose);
    int codegen_example(const char* output_dir, int verbose);
    

#ifdef __cplusplus
}
#endif