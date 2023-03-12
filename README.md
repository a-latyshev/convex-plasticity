# Convex-plasticity

The project aims at exploring a finite-element formulation of plasticity in the next generation [FEniCSx](https://fenicsproject.org/) problem-solving environment. The main goal is to propose an efficient and generic implementation, which can tackle plasticity models with hardening taking into account non-smooth yield criteria. In order to achieve this goal “classical plasticity”, “convex plasticity” and “custom assembly” approaches have been used and further developed. They are focused on the implementation of plasticity modelling with the help of FEniCSx, the application of convex optimization in the context of plasticity theory and the use of the concept of just-in-time (JIT) compilation.

The repository contains a framework in Python that aims to solve plasticity problems in the FEniCSx environment. In this framework, two common approaches of modelling plasticity are implemented. The first one is called “classical plasticity” to pay tribute to [the original implementation](https://comet-fenics.readthedocs.io/en/latest/demo/2D_plasticity/vonMises_plasticity.py.html) of solving plasticity problems using a common return-mapping algorithm. The classical approach is well-suited for smooth yield criteria such as the von Mises and Drucker-Prager yield criteria. The second one is called “convex plasticity” because of the application of convex optimization in the context of plasticity problem-solving. The convex plasticity approach works for a wide range of yield criteria including non-smooth ones, for example for the Rankine yield criterion. Application of this approach makes the code more generic. 
 
The framework draws attention to efficiency of numerical calculations in the context of writing scientific code in Python. In order to increase code performance, the concept of just-in-time compilation is applied. The project proposes an approach according to which the most critical parts of the assembly process of finite-element modelling are replaced by more performant JIT code. This approach is called “custom assembly” and inspired by the authors of [DOLFINx](https://github.com/FEniCS/dolfinx/blob/main/python/test/unit/fem/test_custom_assembler.py). 

This repository proposes implementations of three approaches to modelling plasticity: “classical plasticity”, “convex plasticity” and “custom assembly”. The first two show how to solve plasticity problems via the FEniCSx environment by implementing a return-mapping algorithm and how to extend such an implementation to a wider range of plasticity models using convex optimisation theory. The third approach aims at demonstrating how the efficiency of plasticity modelling via FEniCSx can be improved by applying JIT compilation and changing the assembly procedure. 

Keywords: plasticity, isotropic hardening, return-mapping algorithm, convex optimization, conic solver, FEniCS Project, cvxpy, python, just-in-time compilation.

## How to use

We welcome you to get started from [tutorials folder](tutorials/), where you find descriptions of our ideas and code examples solving a plasticity problem using different approaches.

In order to deeply understand theoretical aspects of this project, we recommend you to get familiarised with the [rapport](rapport/Andrey_Latyshev_rapport.pdf) in the [rapport folder](rapport/). You find there mathematical formulations of the elasto-plastic problem, description of numerical methods to solve it as well as numerical experiments testing the proposed approaches.

All sources of the framework are stored in the [src folder](src/). 

The [demo folder](demo/) contains different examples implementing a specific feature, which is used or can be useful in this framework.

Other folders are intended to develop, test and analyse new features of the framework. 
 

## Dependencies 

[![dolfinx](https://badgen.net/badge/DOLFINx/0.3.1/blue)](https://github.com/FEniCS/dolfinx)
[![cvxpy](https://badgen.net/badge/CVXPY/1.2.2/blue)](https://github.com/cvxpy/cvxpy)
[![numba](https://badgen.net/badge/numba/0.56.4/blue)](https://github.com/numba/numba)
