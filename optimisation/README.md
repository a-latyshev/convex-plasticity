# Benchmarks

We implemented here several comparative tests to show the difference between the classical return-mapping approach and the one, where the return step is replaced by resolving of a supplementary convex optimization problem in the case of von Mises plasticity problem. Its full description can be found [here](https://comet-fenics.readthedocs.io/en/latest/demo/2D_plasticity/vonMises_plasticity.py.html).

## Classical Fenics implementation with interpolation approach vs convex optimization

### Interpolation approach

| Mesh | Total time (s) | Return-mapping step <br />(mean value in s) | Elements nb. | Nodes nb. |
| :---: | :---: | :----: | :----: | :---: | 
| Coarse | 2.344 | 0.04 | 69 | 50 | 
| Medium | 3.794 | 0.041 | 1478 | 811 | 

### Convex optimization approach

| Mesh | Total time (s) | Return-mapping step <br />(mean value in s) | Elements nb. | Nodes nb. |
| :---: | :---: | :----: | :----: | :---: | 
| Coarse | 124.206 | 2.182 | 69 | 50 | 
| Medium | 3000 | 50 | 1478 | 811 | 

## Vectorized convex optimization problems 

We can solve a vectorized convex optimization problem by stacking target function in a vector or matrix. Here we gather them in the order of quadrature points.

| Mesh | Elements nb. | Nodes nb. | Quadrature points nb. |
| :---: | :----: | :----: | :---: | 
| Coarse | 69 | 50 | 207 | 
| Medium | 1478 | 811 |  | 
| Fine |  |  |  | 

N is a number of quadrature points in a group

### Coarse mesh
| N | Total time (s) | Return-mapping <br />(total in s) | Differentiation <br />(total in s) | Convex solving <br />(total in s) |
| :---: | :---: | :----: | :----: | :---: | 
| 1 | 138.393 | 48.894 | 33.405 | 14.667 | 
| 3 | 99.641 | 41.961 | 35.222 | 6.366 | 
| 9 | 99.067 | 40.830 | 37.215 | 3.423 | 
| 207 | 453.724 | 193.869 | 100.799 | 92.952 | 