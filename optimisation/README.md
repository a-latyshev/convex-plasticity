# Numerical experiments 

We implemented here several comparative tests to show the difference between the classical return-mapping approach and the one, where the return step is replaced by resolving of a supplementary convex optimization problem in the case of von Mises plasticity problem. Its full description can be found [here](https://comet-fenics.readthedocs.io/en/latest/demo/2D_plasticity/vonMises_plasticity.py.html).


## Fenics implementation of plasticity problems with the interpolation technic

| Mesh | Total time (s) | Return-mapping step <br />(mean value in s) | Elements nb. | Nodes nb. |
| :---: | :---: | :----: | :----: | :---: | 
| Coarse | 2.344 | 0.04 | 69 | 50 | 
| Medium | 3.794 | 0.041 | 1478 | 811 | 

##  Fenics implementation of plasticity problems with the convex optimization technic

| Mesh | Total time (s) | Return-mapping step <br />(mean value in s) | Elements nb. | Nodes nb. |
| :---: | :---: | :----: | :----: | :---: | 
| Coarse | 124.206 | 2.182 | 69 | 50 | 
| Medium | 3000 | 50 | 1478 | 811 | 
