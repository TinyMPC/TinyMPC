// Deployable on Crazyflie: generate the 100 Hz quadrotor cache used by
// controller_tinympc_eigen in tinympc-crazyflie.

#include <iostream>
#ifdef __MINGW32__
#include <experimental/filesystem>
namespace std_fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace std_fs = std::filesystem;
#endif
#include <stdexcept>
#include <system_error>

#include <tinympc/tiny_api.hpp>
#include <tinympc/codegen.hpp>

#ifndef NSTATES
#define NSTATES 12
#endif
#ifndef NINPUTS
#define NINPUTS 4
#endif
#ifndef NHORIZON
#define NHORIZON 25
#endif

#include "problem_data/quadrotor_100hz_params.hpp"

namespace {

void deploy_if_present(const std::string& output_dir)
{
#ifdef TINYMPC_SOURCE_DIR
    const std_fs::path tinympc_root = std_fs::path(TINYMPC_SOURCE_DIR);
    const std_fs::path app_src_dir =
        tinympc_root.parent_path() / "apps" / "controller_tinympc_eigen" / "src";
    const std_fs::path generated_dir = app_src_dir / "generated";

    if (!std_fs::exists(app_src_dir)) {
        return;
    }

    const std_fs::path generated_root = std_fs::path(output_dir);
    const std_fs::path generated_params =
        generated_root / "crazyflie" / "params_100hz.h";
    const std_fs::path generated_problem =
        generated_root / "problem_data" / "quadrotor_100hz_generated.hpp";

    std::error_code ec;
    std_fs::create_directories(generated_dir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create generated deploy directory");
    }

    std_fs::copy_file(
        generated_params, app_src_dir / "params_100hz.h",
        std_fs::copy_options::overwrite_existing, ec);
    if (ec) {
        throw std::runtime_error("Failed to deploy params_100hz.h");
    }

    ec.clear();
    std_fs::copy_file(
        generated_problem, generated_dir / "quadrotor_100hz_generated.hpp",
        std_fs::copy_options::overwrite_existing, ec);
    if (ec) {
        throw std::runtime_error("Failed to deploy quadrotor_100hz_generated.hpp");
    }

    std::cout << "Deployed generated Crazyflie headers into "
              << app_src_dir.string() << std::endl;
#endif
}

}  // namespace

int main()
{
    TinySolver* solver = nullptr;

    tinyMatrix Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
    tinyMatrix Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS, RowMajor>>(Bdyn_data);
    tinyVector Q = Map<Matrix<tinytype, NSTATES, 1>>(Q_data);
    tinyVector R = Map<Matrix<tinytype, NINPUTS, 1>>(R_data);

    tinyVector fdyn = tinyVector::Zero(NSTATES);

    int verbose = 0;
    int status = tiny_setup(&solver,
                            Adyn, Bdyn, fdyn, Q.asDiagonal(), R.asDiagonal(),
                            rho_value, NSTATES, NINPUTS, NHORIZON, verbose);
    if (status) {
        return status;
    }

    const std::string output_dir =
        std_fs::absolute("tinympc_generated_code_quadrotor_100hz").string();

    tiny_codegen_problem_data(
        solver, output_dir.c_str(), "quadrotor_100hz_generated", verbose);
    tiny_codegen_crazyflie_params(
        solver, output_dir.c_str(), "params_100hz", verbose);
    deploy_if_present(output_dir);

    return 0;
}
