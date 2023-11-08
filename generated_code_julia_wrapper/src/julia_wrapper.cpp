#include <tinympc/tiny_data_workspace.hpp>
#include <tinympc/admm.hpp>

extern "C"{


    void call_tiny_solve(){
        tiny_solve(&tiny_data_solver);

    }

    void set_initial_position(float *x, float *init_cond){
        for (int i=0; i < NSTATES; i++){
            for (int j=0; j < NHORIZON; j++){
                if (j==0){
                    tiny_data_solver.work->x(i,j) = init_cond[i];
                }
                printf("x0: %f\n", tiny_data_solver.work->x(i,j));
            }
        }
    }

    void get_solution(float *x_soln){
        tiny_solve(&tiny_data_solver);
        printf("Rows:  %ld\n", tiny_data_solver.work->x.rows());
        printf("Cols:  %ld\n", tiny_data_solver.work->x.cols());
        printf("contents1:  %f\n", tiny_data_solver.work->x(0,0));
        printf("contents2:  %f\n", tiny_data_solver.work->x(0,1));

        Eigen::Map<tiny_MatrixNxNh>(x_soln, tiny_data_solver.work->x.rows(), tiny_data_solver.work->x.cols()) = tiny_data_solver.work->x;
        for (int i=0; i < NHORIZON*NSTATES; i++){
            printf("x_soln:  %f\n", x_soln[i]); // store soln in x_soln to pass back to Julia
        }
    }
}
