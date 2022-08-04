#include <cpg_solve.h>
#include <cpg_workspace.h>
#include "scs.h"
#include <stdbool.h>

typedef struct {
    double A[3];
    double b[3];
} CPG_Params_cpp_t;

// Flags for updated user-defined parameters
typedef struct {
    bool A;
    bool b;
} CPG_Updated_cpp_t;

// Primal solution
typedef struct {
    double x[2];
} CPG_Prim_cpp_t;

// Dual solution
typedef struct {
    double d0[2];
} CPG_Dual_cpp_t;

// Solver information
typedef struct {
    double obj_val;
    int iter;
//  char* status;
    double pri_res;
    double dua_res;
    double time;
} CPG_Info_cpp_t;

// Solution and solver information
typedef struct {
    CPG_Prim_cpp_t prim;
    CPG_Dual_cpp_t dual;
    CPG_Info_cpp_t info;
} CPG_Result_cpp_t;

typedef struct {
    double obj_val;
    int iter;
    double pri_res;
    double dua_res;
    double* time;
} new_struct;

static int i;
// Main solve function
void solve_cpp(CPG_Updated_cpp_t *CPG_Updated_cpp, CPG_Params_cpp_t *CPG_Params_cpp, CPG_Result_cpp_t *CPG_Result_cpp) {

    // Pass changed user-defined parameter values to the solver
    if (CPG_Updated_cpp->A) {
        for(i=0; i<3; i++) {
            cpg_update_A(i, CPG_Params_cpp->A[i]);
        }
    }
    if (CPG_Updated_cpp->b) {
        for(i=0; i<3; i++) {
            cpg_update_b(i, CPG_Params_cpp->b[i]);
        }
    }

    // Solve
    // std::clock_t ASA_start = std::clock();
    cpg_solve();
    // std::clock_t ASA_end = std::clock();

    // Arrange and return results
    CPG_Prim_cpp_t CPG_Prim_cpp;
    for(i=0; i<2; i++) {
        CPG_Prim_cpp.x[i] = CPG_Prim.x[i];
    }
    CPG_Dual_cpp_t CPG_Dual_cpp;
    for(i=0; i<2; i++) {
        CPG_Dual_cpp.d0[i] = CPG_Dual.d0[i];
    }
    CPG_Info_cpp_t CPG_Info_cpp;
    CPG_Info_cpp.obj_val = CPG_Info.obj_val;
    CPG_Info_cpp.iter = CPG_Info.iter;
    
    //CPG_Info_cpp.status = CPG_Info.status;

    CPG_Info_cpp.pri_res = CPG_Info.pri_res;
    CPG_Info_cpp.dua_res = CPG_Info.dua_res;
    CPG_Info_cpp.time = 0.0; //1.0*(ASA_end-ASA_start) / CLOCKS_PER_SEC;

    // CPG_Result_cpp_t CPG_Result_cpp;
    CPG_Result_cpp->prim = CPG_Prim_cpp;
    CPG_Result_cpp->dual = CPG_Dual_cpp;
    CPG_Result_cpp->info = CPG_Info_cpp;
    //return CPG_Result_cpp;
}

CPG_Updated_cpp_t test_func(CPG_Updated_cpp_t *CPG_Updated_cpp, CPG_Params_cpp_t *CPG_Params_cpp) {
    CPG_Updated_cpp_t tmp = {};
    return tmp;
}