#include "tinympc/tiny_wrapper.hpp"

extern "C"
{
    void set_x0(float *x0, int verbose)
    {
        for (int i = 0; i < NSTATES; i++)
        {
            tiny_data_solver.work->x(i, 0) = x0[i];
        }

        if (verbose != 0)
        {
            for (int i = 0; i < NSTATES; i++)
            {
                printf("set_x0 result:  %f\n", tiny_data_solver.work->x(i, 0));
            }
        }
    }

    void set_xref(float *xref, int verbose)
    {
        for (int j = 0; j < NHORIZON; j++)
        {
            for (int i = 0; i < NSTATES; i++)
            {
                tiny_data_solver.work->Xref(i, j) = xref[j * NSTATES + i];
            }
        }

        if (verbose != 0)
        {
        for (int j = 0; j < NHORIZON; j++)
        {
            for (int i = 0; i < NSTATES; i++)
            {
                    printf("set_xref result:  %f\n", tiny_data_solver.work->Xref(i, j));
                }
            }
        }
    }

    void set_umin(float *umin, int verbose)
    {
        for (int i = 0; i < NINPUTS; i++)
        {
            for (int j = 0; j < NHORIZON - 1; j++)
            {
                tiny_data_solver.work->u_min(i, j) = umin[i];
            }
        }

        if (verbose != 0)
        {
            for (int j = 0; j < NHORIZON - 1; j++)
            {
                for (int i = 0; i < NINPUTS; i++)
                {
                    printf("set_umin result:  %f\n", tiny_data_solver.work->u_min(i, j));
                }
            }
        }
    }

    void set_umax(float *umax, int verbose)
    {
        for (int i = 0; i < NINPUTS; i++)
        {
            for (int j = 0; j < NHORIZON - 1; j++)
            {
                tiny_data_solver.work->u_max(i, j) = umax[i];
            }
        }

        if (verbose != 0)
        {
            for (int j = 0; j < NHORIZON - 1; j++)
            {
                for (int i = 0; i < NINPUTS; i++)
                {
                    printf("set_umax result:  %f\n", tiny_data_solver.work->u_max(i, j));
                }
            }
        }
    }

    void set_xmin(float *xmin, int verbose)
    {
        for (int i = 0; i < NSTATES; i++)
        {
            for (int j = 0; j < NHORIZON; j++)
            {
                tiny_data_solver.work->x_min(i, j) = xmin[i];
            }
        }

        if (verbose != 0)
        {
            for (int j = 0; j < NHORIZON; j++)
            {
                for (int i = 0; i < NSTATES; i++)
                {
                    printf("set_xmin result:  %f\n", tiny_data_solver.work->x_min(i, j));
                }
            }
        }
    }

    void set_xmax(float *xmax, int verbose)
    {
        for (int i = 0; i < NSTATES; i++)
        {
            for (int j = 0; j < NHORIZON; j++)
            {
                tiny_data_solver.work->x_max(i, j) = xmax[i];
            }
        }

        if (verbose != 0)
        {
            for (int j = 0; j < NHORIZON; j++)
            {
                for (int i = 0; i < NSTATES; i++)
                {
                    printf("set_xmax result:  %f\n", tiny_data_solver.work->x_max(i, j));
                }
            }
        }
    }

    void reset_dual_variables(int verbose)
    {
        tiny_data_solver.work->y = tiny_MatrixNuNhm1::Zero();
        tiny_data_solver.work->g = tiny_MatrixNxNh::Zero();

        if (verbose != 0)
        {
            std::cout << "reset duals finished" << std::endl;
        }
    }

    void call_tiny_solve(int verbose)
    {
        tiny_solve(&tiny_data_solver);

        if (verbose != 0)
        {
            std::cout << "tiny solve finished" << std::endl;
        }
    }

    void get_x(float *x_soln, int verbose)
    {
        Eigen::Map<tiny_MatrixNxNh>(x_soln, tiny_data_solver.work->x.rows(), tiny_data_solver.work->x.cols()) = tiny_data_solver.work->x;

        if (verbose != 0)
        {
            for (int i = 0; i < NHORIZON; i++)
            {
                printf("x_soln:  %f\n", x_soln[i]);
            }
        }
    }

    void get_xref(float *xref, int verbose)
    {
        Eigen::Map<tiny_MatrixNxNh>(xref, tiny_data_solver.work->Xref.rows(), tiny_data_solver.work->Xref.cols()) = tiny_data_solver.work->Xref;

        if (verbose != 0)
        {
            for (int i = 0; i < NHORIZON; i++)
            {
                printf("x_soln:  %f\n", xref[i]);
            }
        }
    }

    void get_u(float *u_soln, int verbose)
    {
        Eigen::Map<tiny_MatrixNuNhm1>(u_soln, tiny_data_solver.work->u.rows(), tiny_data_solver.work->u.cols()) = tiny_data_solver.work->u;

        if (verbose != 0)
        {
            for (int i = 0; i < NHORIZON - 1; i++)
            {
                printf("u_soln:  %f\n", u_soln[i]);
            }
        }
    }
}
