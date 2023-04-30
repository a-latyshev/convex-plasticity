typedef int c_int;
typedef double c_float;

// Update user-defined parameter values
extern void cpg_update_sig_old(c_int idx, c_float val);
extern void cpg_update_deps(c_int idx, c_float val);

// Map user-defined to canonical parameters
extern void cpg_canonicalize_b();

// Retrieve dual solution in terms of user-defined constraints
extern void cpg_retrieve_dual();

// Retrieve solver information
extern void cpg_retrieve_info();

// Solve via canonicalization, canonical solve, retrieval
extern void cpg_solve();

// Update solver settings
extern void cpg_set_solver_default_settings();
extern void cpg_set_solver_normalize(c_int normalize_new);
extern void cpg_set_solver_scale(c_float scale_new);
extern void cpg_set_solver_adaptive_scale(c_int adaptive_scale_new);
extern void cpg_set_solver_rho_x(c_float rho_x_new);
extern void cpg_set_solver_max_iters(c_int max_iters_new);
extern void cpg_set_solver_eps_abs(c_float eps_abs_new);
extern void cpg_set_solver_eps_rel(c_float eps_rel_new);
extern void cpg_set_solver_eps_infeas(c_float eps_infeas_new);
extern void cpg_set_solver_alpha(c_float alpha_new);
extern void cpg_set_solver_time_limit_secs(c_float time_limit_secs_new);
extern void cpg_set_solver_verbose(c_int verbose_new);
extern void cpg_set_solver_warm_start(c_int warm_start_new);
extern void cpg_set_solver_acceleration_lookback(c_int acceleration_lookback_new);
extern void cpg_set_solver_acceleration_interval(c_int acceleration_interval_new);
extern void cpg_set_solver_write_data_filename(const char* write_data_filename_new);
extern void cpg_set_solver_log_csv_filename(const char* log_csv_filename_new);

// User-defined parameters
typedef struct {
    double sig_old[4];
    double deps[4];
} CPG_Params_cpp_t;

// Flags for updated user-defined parameters
typedef struct {
    bool sig_old;
    bool deps;
} CPG_Updated_cpp_t;

// Primal solution
typedef struct {
    double sig[4];
} CPG_Prim_cpp_t;

// Dual solution
typedef struct {
    double d0;
} CPG_Dual_cpp_t;

// Solver information
typedef struct {
    double obj_val;
    int iter;
    // char* status;
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

// Main solve function
void solve_cpp(CPG_Updated_cpp_t *CPG_Updated_cpp, CPG_Params_cpp_t *CPG_Params_cpp, CPG_Result_cpp_t *CPG_Result_cpp);
