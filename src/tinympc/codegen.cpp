#include <stdio.h>
#include <ctype.h> /* -> toupper */
#include <iostream>

#include "codegen.hpp"

using namespace Eigen;
IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

int tiny_codegen(int nx, int nu, int N,
                 tinytype *Adyn, tinytype *Bdyn, tinytype *Q, tinytype *Qf, tinytype *R,
                 tinytype *x_min, tinytype *x_max, tinytype *u_min, tinytype *u_max,
                 tinytype rho, tinytype abs_pri_tol, tinytype abs_dua_tol, int max_iters, int check_termination)
{
  TinyCache cache;
  TinySettings settings;
  TinyWorkspace work;
  TinySolver solver{&settings, &cache, &work};

  settings.abs_dua_tol = abs_dua_tol;
  settings.abs_pri_tol = abs_pri_tol;
  settings.check_termination = check_termination;
  settings.max_iter = max_iters;
  if (x_min != nullptr && x_max != nullptr)
  {
    settings.en_state_bound = 1;
  }
  else
  {
    settings.en_state_bound = 0;
  }
  if (u_min != nullptr && u_max != nullptr)
  {
    settings.en_input_bound = 1;
  }
  else
  {
    settings.en_input_bound = 0;
  }

  cache.rho = rho;
  work.Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn);
  work.Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS, RowMajor>>(Bdyn);
  work.Q = Map<tiny_VectorNx>(Q);
  work.Qf = Map<tiny_VectorNx>(Qf);
  work.R = Map<tiny_VectorNu>(R);
  work.x_min = Map<tiny_MatrixNxNh>(x_min); // x_min is col-major
  work.x_max = Map<tiny_MatrixNxNh>(x_max);
  work.u_min = Map<tiny_MatrixNuNhm1>(u_min); // u_min is col-major
  work.u_max = Map<tiny_MatrixNuNhm1>(u_max);

  // Update by adding rho * identity matrix to Q, Qf, R
  work.Q = work.Q + rho * tiny_VectorNx::Ones();
  work.Qf = work.Qf + rho * tiny_VectorNx::Ones();
  work.R = work.R + rho * tiny_VectorNu::Ones();
  tiny_MatrixNxNx Q1 = work.Q.array().matrix().asDiagonal();
  tiny_MatrixNxNx Qf1 = work.Qf.array().matrix().asDiagonal();
  tiny_MatrixNuNu R1 = work.R.array().matrix().asDiagonal();

  // Printing
  std::cout << "A = " << work.Adyn.format(CleanFmt) << std::endl;
  std::cout << "B = " << work.Bdyn.format(CleanFmt) << std::endl;
  std::cout << "Q = " << Q1.format(CleanFmt) << std::endl;
  std::cout << "Qf = " << Qf1.format(CleanFmt) << std::endl;
  std::cout << "R = " << R1.format(CleanFmt) << std::endl;
  std::cout << "rho = " << cache.rho << std::endl;

  // Riccati recursion to get Kinf, Pinf 
  tiny_MatrixNuNx Ktp1;
  tiny_MatrixNxNx Ptp1 = Qf1;
  Ktp1 = tiny_MatrixNuNx::Zero();

  for (int i = 0; i < 100; i++)
  {
    cache.Kinf = (R1 + work.Bdyn.transpose() * Ptp1 * work.Bdyn).inverse() * work.Bdyn.transpose() * Ptp1 * work.Adyn;
    cache.Pinf = Q1 + work.Adyn.transpose() * Ptp1 * (work.Adyn - work.Bdyn * cache.Kinf);
    // if Kinf converges, break
    if ((cache.Kinf - Ktp1).cwiseAbs().maxCoeff() < 1e-6)
    {
      std::cout << "Kinf converged after " << i+1 << " iterations" << std::endl;
      break;
    }
    Ktp1 = cache.Kinf;
    Ptp1 = cache.Pinf;
  }

  // compute caches
  cache.Quu_inv = (R1 + work.Bdyn.transpose() * cache.Pinf * work.Bdyn).inverse();
  cache.AmBKt = (work.Adyn - work.Bdyn * cache.Kinf).transpose();
  cache.coeff_d2p = cache.Kinf.transpose() * R1 - cache.AmBKt * cache.Pinf * work.Bdyn;

  std::cout << "Kinf = " << cache.Kinf.format(CleanFmt) << std::endl;
  std::cout << "Pinf = " << cache.Pinf.format(CleanFmt) << std::endl;
  std::cout << "Quu_inv = " << cache.Quu_inv.format(CleanFmt) << std::endl;
  std::cout << "AmBKt = " << cache.AmBKt.format(CleanFmt) << std::endl;
  std::cout << "coeff_d2p = " << cache.coeff_d2p.format(CleanFmt) << std::endl;

  // Write to files
  // Write caches
  // Write settings
  // Write workspace
  // Write solver

  return 1;
}