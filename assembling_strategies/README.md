# Custom assembling system 

## Context

We consider the following typical non-linear problem in our study:

Find $\underline{u} \in V$, such that

$$ F(u) = \int_\Omega \underline{\underline{\sigma}}(\underline{u}) : \underline{\underline{\varepsilon}}(\underline{v}) dx - \int_\Omega \underline{f} \underline{v} dx = 0 \quad \forall v \in V,$$
where an expression of $\underline{\underline{\sigma}}(\underline{u})$ is not linear and cannot be defined via UFLx. Let us show you some examples

- $\underline{\underline{\sigma}}(\underline{u}) = f(\varepsilon_\text{I}, \varepsilon_\text{II}, \varepsilon_\text{III})$,
- $\underline{\underline{\sigma}}(\underline{u}) = \min\limits_\alpha g(\underline{\underline{\varepsilon}}(\underline{u}),  \alpha)$,

where $\varepsilon_\text{I}, \varepsilon_\text{II}, \varepsilon_\text{III}$ are eigen values of $\underline{\underline{\varepsilon}}$.

Furthermore, $\underline{\underline{\sigma}}(\underline{u})$ can be depended on other scalar, vector or even tensor fields, which we would like to prevent from memory allocation. They shell to be calculated during the assembling procedure.

For example here
- $\underline{\underline{\sigma}}(\underline{u}) = h(\underline{\underline{\varepsilon}}(\underline{u}), p, \beta, \underline{\underline{n}}, \dots)$,

we don't use internal variables $\beta$ and $\underline{\underline{n}}$ as global fields, so it would be a waste of memory if we allocated memory for their values.

In addition, as we use a standard Newton method to find the solution, we need to calculate the derivative of $\underline{\underline{\sigma}}$ also. In our cases it's the forth rank tensor, which values depend on some internal variables of $\underline{\underline{\sigma}}$. Thus, it's necessary to have a fonctionality which will allow to do coupled calculations of $\underline{\underline{\sigma}}$ and its derivative during assembling procedure.

As the result we require the following features:

1. To express functions of variational problem in more complex way than it's allowed by UFLx
2. To not allocate the memory for unnecessary fields 
3. To couple calculations of different parts of variational problem during assembling

The following text describes our own view of these features implementation.

## Custom assembling 

Following the concept of the [custom assembler](https://github.com/FEniCS/dolfinx/blob/main/python/test/unit/fem/test_custom_assembler.py), which uses the power of `numba` and  `cffi` python libraries, we implemented our own version of custom assembler, where we can change the main loop of the assembling procedure.

<!-- There are several essential elements of the algorithm to be mentioned. First of all,  -->

## CustomFunction 

We would like to introduce a concept of `CustomFunction`, which is essential for our study. Let us consider the next simple variational problem 

$$ \int_\Omega g \cdot uv dx, \quad \forall v \in V, $$

where $u$ is a trial function, $v$ is a test function and the function $g$ is an expression. For this moment we must use `fem.Function` class to implement this variational form. Knowing the exact UFLx expression of $g$ we can calculate its values on every element using the interpolation procedure of `fem.Expression` class. So we save all values of $g$ in one global vector. The goal is to have a possibility to calculate $g$ expression, no matter how difficult it is, in every element node (for instance, in every gauss point, if we define $g$ on a quadrature element) during the assembling procedure. 

We introduce a new entity named as `CustomFunction` (or `CustomExpression`). It
1. inherits `fem.Function`
2. contains a method `eval`, which will be called inside of the assemble loop and calculates the function local values

## DummyFunction 

Besides `CustomFunction` we need an other entity. Every `fem.Function` object stores its values globally, but we would like to avoid such a waste of memory updating the function value during the assembling procedure. Let us consider the previous variational form, where $g$ contains its local-element values now. If there is one local value of $g$ (only 1 gauss point), $g$ will be  `fem.Constant`, but we need to store different values of $g$ in every element node (gauss point). So we introduce a concept of `DummyFunction` (or `ElementaryFunction`?), which 
1. inherits `fem.Function`
2. allocates the memory for local values only
3. can be updated during assembling procedure

## Examples 

We implemented elasticity and plasticity problems to explain our ideas by examples.

### Elasticity problem

Let's consider a beam stretching with the left side fixed. On the other side we apply displacements. 

$$\int_\Omega (\mathbf{C}:\underline{\underline{\varepsilon}}(\underline{du})) : \underline{\underline{\varepsilon}}(\underline{v}) dx - \int_\Omega \underline{\underline{\sigma}}^n : \underline{\underline{\varepsilon}}(\underline{v}) dx = 0 \quad \forall v \in V,$$

$$ \partial\Omega_\text{left} : u_x = 0, $$
$$ (0, 0) : u_y = 0, $$
$$ \partial\Omega_\text{right} : u_x = t \cdot u_\text{bc},$$
where $u_\text{bc}$ is a maximal displacement on the right side of the beam, $t$ is a parameter varying from 0 to 1, $\underline{\underline{\sigma}}^n$ is equal to a stress on a previous loading step and $\mathbf{C}$ is the 4th rank stiffness tensor.

Let's focus on the key points. In this "naif" example the tensor $\mathbf{C}$ is constant, but it's often a variable in a majority of non-linear models. We would like to change it on every assembling step. In our therms it is a `DummyFunction`. Obviosly $\sigma^n$ serves as `CustomFunction`, which depends on $\mathbf{C}$ and we should take it into account. 
```python
q_dsigma = ca.DummyFunction(VQT, name='stiffness') # tensor C
q_sigma = ca.CustomFunction(VQV, eps(u), [q_dsigma], get_eval)  # sigma^n
```
In the `CustomFunction` constructor we observe three arguments. The first one is a ufl-expression of a basic function participating in the `q_sigma` expression. It will be compiled via ffcx and will be sent as "tabulated" expression to a numba function, which performs the calculation of `q_sigma`. The second argument is a list of `q_sigma` coefficients (`fem.Function` or `DummyFunction`), which take a part in calculations of `q_sigma`. The third argument contains a function generating a `CustomFunction` method `eval`, which will be called during the assembling. It describes every step of local calculation of $\sigma^n$.

Besides the local implementation of new entities we need to change the assembling procedure loop to describe explicitly the interaction between different coefficients of linear and bilinear forms. It allows us to write a quite general custom assembler, which will work for any kind non-linear problem. Thus we have to define two additional numba functions to calculate local values of forms kernels coefficients (see the code below).

```python
@numba.njit(fastmath=True)
def local_assembling_b(cell, coeffs_values_global_b, coeffs_coeff_values_b, coeffs_dummy_values_b, coeffs_eval_b, u_local, coeffs_constants_b, geometry, entity_local_index, perm):
    sigma_local = coeffs_values_global_b[0][cell]
    
    output_values = coeffs_eval_b[0](sigma_local, 
                                     u_local, 
                                     coeffs_constants_b[0], 
                                     geometry, entity_local_index, perm)

    coeffs_b = sigma_local

    for i in range(len(coeffs_dummy_values_b)):
        coeffs_dummy_values_b[i][:] = output_values[i] #C update

    return coeffs_b

@numba.njit(fastmath=True)
def local_assembling_A(coeffs_dummy_values_b):
    coeffs_A = coeffs_dummy_values_b[0]
    return coeffs_A
```

### Plasticity problem

The elasticity case is trivial and doesn't show clearly our demands by the described above features. Therefore we present here a standard non-linear problem from our scientific domain - a plasticity problem.

The full description of the problem and its implementation on a legacy version of Fenics is introduced [here](https://comet-fenics.readthedocs.io/en/latest/demo/2D_plasticity/vonMises_plasticity.py.html). We focus on the following variational problem only 
$$ \int\limits_\Omega \left( \mathbf{C}^\text{tang} : \underline{\underline{\varepsilon}}(\underline{du})  \right) : \underline{\underline{\varepsilon}}(\underline{v}) dx + \int\limits_\Omega \underline{\underline{\sigma_n}} : \underline{\underline{\varepsilon}}(\underline{v}) dx - q \int\limits_{\partial\Omega_{\text{inside}}} \underline{n} \cdot \underline{v} dx = 0, \quad \forall \underline{v} \in V, $$

where $\underline{\underline{\sigma}}_n$ is the stress tensor calculated on the previous loading step and the 4th rank tensor $\mathbf{C}^\text{tang}$ is defined as follows:
$$ \mathbf{C}^\text{tang} = \mathbf{C} - 3\mu \left( \frac{3\mu}{3\mu + H} -\beta \right) \underline{\underline{n}} \otimes \underline{\underline{n}} - 2\mu\beta \mathbf{DEV} $$

In contrast to the elasticity problem the 4th rank tensor here is a variable and has different values in every gauss point. At the same time we would like to avoid an allocation of tensor with that rank. Now it should be more evident to use the concept of `DummyFunction` for $\mathbf{C}^\text{tang}$.
Other necessary expressions are presented below, but we won't get into details 

$$\underline{\underline{\sigma_\text{elas}}} = \underline{\underline{\sigma_n}} + \mathbf{C} : \Delta \underline{\underline{\varepsilon}}, \quad \sigma^\text{eq}_\text{elas} = \sqrt{\frac{3}{2} \underline{\underline{s}} : \underline{\underline{s}}}$$ 

$$\underline{\underline{s}} = \mathsf{dev} \underline{\underline{\sigma_\text{elas}}} $$

$$ f_\text{elas} = \sigma^\text{eq}_\text{elas} - \sigma_0 - H p_n $$  

$$\Delta p = \frac{< f_\text{elas} >_+}{3\mu + H},$$

$$\beta = \frac{3\mu}{\sigma^\text{eq}_\text{elas}}\Delta p$$

$$\underline{\underline{n}} = \frac{\underline{\underline{s}}}{\sigma^\text{eq}_\text{elas}}$$

$$\underline{\underline{\sigma_{n+1}}} = \underline{\underline{\sigma_\text{elas}}} - \beta \underline{\underline{s}}$$

$$
    < f>_+ = 
        \begin{cases}
            f, \quad f > 0, \\
            0, \quad \text{otherwise}
        \end{cases}
$$

We can conclude, that the fields $\underline{\underline{\sigma_{n+1}}} = \underline{\underline{\sigma_{n+1}}}(\Delta \underline{\underline{\varepsilon}}, \beta, \underline{\underline{n}}, dp, p_n, \underline{\underline{\sigma_n}})$ and $\mathbf{C}^\text{tang} = \mathbf{C}^\text{tang}(\beta, \underline{\underline{n}})$ depend on the common variables $\beta$ and $\underline{\underline{n}}$. So before, it would be necessary to allocate additional space for them and calculate $\underline{\underline{\sigma_{n+1}}}$ and $\mathbf{C}^\text{tang}$ separately. 

In comparison with the elasticity case the `CustomFunction` `sig` has more dependent fields. Look at the code below
```python
C_tang = ca.DummyFunction(QT, name='tangent') # tensor C_tang
sig = ca.CustomFunction(W, eps(Du), [C_tang, p, dp, sig_old], get_eval) # sigma_n
```
As it was expected, the local evaluation of the `CustomFunction` becomes more complex 

```python
def get_eval(self:ca.CustomFunction):
    tabulated_eps = self.tabulated_input_expression
    n_gauss_points = len(self.input_expression.X)
    local_shape = self.local_shape
    C_tang_shape = self.tangent.shape

    @numba.njit(fastmath=True)
    def eval(sigma_current_local, sigma_old_local, p_old_local, dp_local, coeffs_values, constants_values, coordinates, local_index, orientation):
        deps_local = np.zeros(n_gauss_points*3*3, dtype=PETSc.ScalarType)
        
        C_tang_local = np.zeros((n_gauss_points, *C_tang_shape), dtype=PETSc.ScalarType)
        
        sigma_old = sigma_old_local.reshape((n_gauss_points, *local_shape))
        sigma_new = sigma_current_local.reshape((n_gauss_points, *local_shape))

        tabulated_eps(ca.ffi.from_buffer(deps_local), 
                      ca.ffi.from_buffer(coeffs_values), 
                      ca.ffi.from_buffer(constants_values), 
                      ca.ffi.from_buffer(coordinates), ca.ffi.from_buffer(local_index), ca.ffi.from_buffer(orientation))
        
        deps_local = deps_local.reshape((n_gauss_points, 3, 3))

        n_elas = np.zeros((3, 3), dtype=PETSc.ScalarType) 
        beta = np.zeros(1, dtype=PETSc.ScalarType) 
        dp = np.zeros(1, dtype=PETSc.ScalarType) 

        for q in range(n_gauss_points):
            sig_n = as_3D_array(sigma_old[q])
            sig_elas = sig_n + sigma(deps_local[q])
            s = sig_elas - np.trace(sig_elas)*I/3.
            sig_eq = np.sqrt(3./2. * inner(s, s))
            f_elas = sig_eq - sig0 - H*p_old_local[q]
            f_elas_plus = ppos(f_elas)
            dp[:] = f_elas_plus/(3*mu_+H)
            
            sig_eq += TPV # for the case when sig_eq is equal to 0.0
            n_elas[:,:] = s/sig_eq*f_elas_plus/f_elas
            beta[:] = 3*mu_*dp/sig_eq
      
            new_sig = sig_elas - beta*s
            sigma_new[q][:] = np.asarray([new_sig[0, 0], new_sig[1, 1], new_sig[2, 2], new_sig[0, 1]])
            dp_local[q] = dp[0]
            
            C_tang_local[q][:] = get_C_tang(beta, n_elas)
        
        return [C_tang_local.flatten()] 
    return eval
```

Thus it can been seen more clearly the dependance of the tensor $\mathbf{C}^\text{tang}$ on the calculation of the tensor $\underline{\underline{\sigma}}_{n+1}$.

## Summarize

