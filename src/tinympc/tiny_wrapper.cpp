#include "tinympc/tiny_wrapper.hpp"

extern "C"{
    void set_x(float *x0){
        for (int i=0; i < NSTATES; i++){
            tiny_data_solver.work->x(i,0) = x0[i];
            printf("set_x result:  %f\n", tiny_data_solver.work->x(i,0));
        }
    }

    void set_xref(float *xref){
        for (int i=0; i < NSTATES; i++){
            for (int j=0; j < NHORIZON; j++){
                tiny_data_solver.work->Xref(i,j) = xref[i];
                printf("set_xref result:  %f\n", tiny_data_solver.work->Xref(i,j));
            }
        }
    }

    void reset_dual_variables(){
        tiny_data_solver.work->y = tiny_MatrixNuNhm1::Zero();
        tiny_data_solver.work->g = tiny_MatrixNxNh::Zero();
        std::cout << "reset duals finished" << std::endl;
    }

    void call_tiny_solve(){
        tiny_solve(&tiny_data_solver);
        std::cout << "tiny solve finished" << std::endl;
    }

    void get_x(float *x_soln){
        Eigen::Map<tiny_MatrixNxNh>(x_soln, tiny_data_solver.work->x.rows(), tiny_data_solver.work->x.cols()) = tiny_data_solver.work->x;
        for (int i=0; i < NHORIZON; i++){
            printf("x_soln:  %f\n", x_soln[i] );
        }
    }

    void get_u(float *u_soln){
        Eigen::Map<tiny_MatrixNuNhm1>(u_soln, tiny_data_solver.work->u.rows(), tiny_data_solver.work->u.cols()) = tiny_data_solver.work->u;
        for (int i=0; i < NHORIZON-1; i++){
            printf("u_soln:  %f\n", u_soln[i] );
        }
    }

    void edit_x(float *x){
        printf("num rows:  %ld\n", tiny_data_solver.work->x.rows());
        printf("num cols:  %ld\n", tiny_data_solver.work->x.cols());
    }
}
