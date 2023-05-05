# Benchmark for compilation high-dimensional convex problems via cvxpy

Here you find code analysing an impact of a dimension of a conic optimisation problem on compilation time via the cvxpy library. The compilation time is spent on a transformation of an original problem into its canonic form that is required for the majority of modern conic solvers. This benchmark shows that the compilation time drastically increases for high-dimension conic optimisation problems (more than 2000 variables). The main goal of the presented here script is to provide the cvxpy community with a stress test example that could help to accelerate the compilation of high-dimension conic optimisation problems.

## In brief

The vectorized conic optimisation problem is formulated in the following way

$$
    \min F(\mathbf{\Sigma}, \mathbf{P}), \\
    f (\mathbf{\Sigma}, \mathbf{P}) \leq 0,
$$

where $F$ is the objective function, the vectors $\mathbf{\Sigma}$ and $\mathbf{P}$ have the size of 4 ∗ N and N elements respectively and represent main variables of the problem, where N is a certain integer parameter, on which a dimension of the problem is dependent. The elementwise function $f$ represents a conic semi-definitive constrain, which takes 4 components of the vector $\mathbf{\Sigma}$ and 1 component of the vector $\mathbf{P}$. Let us denote, for example, the first 4 components of the vector $\mathbf{\Sigma}$ and the first component of the vector $\mathbf{P}$ as $\sigma$ and $p$ respectively. The expression of the function $f$ is following

$$
    f(\sigma, p) = \sqrt{\frac{3}{2}\mathbf{s}^T\mathbf{s}} - \sigma_0 - pH,
$$

where $\mathbf{s} = \mathbf{DEV} \cdot \sigma$ is a deviatoric part of the variable $\sigma$, the 4x4 matrix $\mathbf{DEV}$ and scalars $\sigma_0$ and $H$ are certain constants.

The explicit expression of the objective function F is written in this way 

$$
    F(\mathbf{\Sigma}, \mathbf{P}) = \frac{1}{2}(\mathbf{\Sigma}^\text{elas}_{n+1} - \mathbf{\Sigma})^T\mathbb{S}(\mathbf{\Sigma}^\text{elas}_{n+1} - \mathbf{\Sigma}) + \frac{1}{2}H(\mathbf{P}^\text{elas}_{n+1} - \mathbf{P})^T(\mathbf{P}^\text{elas}_{n+1} - \mathbf{P})
$$

where the matrix $\mathbb{S}$ of the size 4∗N×4∗N is the block-diagonal matrix with a positive semi-definitive 4x4 matrix on its diagonal, $\mathbf{\Sigma}^\text{elas}_{n+1}  = \mathbf{\Sigma}_{n}  + \mathbf{C} \cdot d\varepsilon$ and $\mathbf{P}^\text{elas}_{n+1} = \mathbf{P}_n$ are problem parameters, where, in this particular benchmark, it is stated that $\mathbf{P}_n$ and $\mathbf{\Sigma}_{n}$ are identical to zero, $\mathbf{C}$ is a 4x4 constant matrix and $d\varepsilon$ is a vector of parameters, values of which are provided via the standard binary file format in NumPy (.npy).

The parameter vector $d\varepsilon$ has the same size as the variable $\mathbf{\Sigma}$, i.e. 4∗N elements. If we choose a certain value of N, we can state that the problem has 4∗N + N ($\mathbf{\Sigma}$ & $\mathbf{P}$) variables, N constraints and 4∗N + 4∗N + N ($d\varepsilon$, $\mathbf{\Sigma}_n$ & $\mathbf{P}_n$) parameters. Changing the value of N allows easily analysing how solving the problem via cvxpy depends on the dimension of the problem. At the moment, there is provided only one vector $d\varepsilon$ of the size 17736, so the maximal value of N is equal to 4434.

## More context
Coming soon