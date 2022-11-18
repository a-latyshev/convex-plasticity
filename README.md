# Convex-plasticity

The main goal of the project is to propose an efficient and generic implementation which can tackle both classical and softening plasticity models. The implementation will be then extended to the resolution of the local plasticity problem using convex optimization solvers and assess its efficiency compared to standard return mapping algorithms. Finally, softening plasticity will be considered and regularization strategies in order to prevent mesh dependency will be explored. The implementation is based on the next generation FEniCS problem solving environment.


The project aims at exploring a finite-element formulation of plasticity in the next generation FEniCSx problem solving environment. The main goal is to propose an efficient and generic implementation which can tackle hardening plasticity models taking into account non-smooth yield criteria. The latter is achieved through the use of convex optimization in the context of plasticity theory.

The repository contains a framework in Python which aims to solve plasticity problems in the FEniCSx environment. The implementation finds the solution of the local plasticity problem using convex optimization solvers. In this framework, both a common approach to modeling plasticity through the return-mapping algorithm and using conic optimization can be found. The latter works for a wide range of yield criteria while the classical approach is well-suited for smooth yield surface such as the von Mises and Drucker-Prager surfaces, but it's more difficult to manage for non-smooth ones such as, for instance, the Rankine yield surface. The research was performed under the assumptions of plane strain, an associative plasticity law and a linear isotropic hardening. Several numerical tests were carried out for different conic solvers where the effect of the size of the vectorized conic optimization problem was analyzed. In addition, the idea of custom assembly was developed to change the assembly process of the FEniCSx library easily. It is based on the concept of just-in-time compilation, which allows us to improve the time performance of some parts of the framework.

Keywords: plasticity, isotropic hardening, return-mapping algorithm, convex optimization, conic solver, FEniCS Project, cvxpy, python, just-in-time compilation.

## How to use

The framework is conditionally divided into 2 parts: convex plasticity and custom assembling.

We call the *convex plasticity* or the convex optimization approach the one where the return-mapping algorithm is implemented by solving the optimization problem. 

The *custom assembling* part of the project means our own version of custom assembler, where we change the main loop of the assembling procedure via FEniCSx environnement.

We are welcome you to get started from [tutorials folder](tutorials/), where you find descriptions of our ideas and code examples solving a plasticity problem using different approaches.

All sources of the framework are situated in the [src folder](src/). In other folders we test different approaches and analyse results. 

## Requirements

Tested for 

* dolfinx (v0.3.1.0)
* cvxpy (v1.2.2)
* numba (v0.56.4)