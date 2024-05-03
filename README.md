# Task
Several practical implementation of optimization algorithms are demonstrated. 
The task is the spectrum deconvolution to the set of pseudo-Voigt curves whose sum approximates the original line shape.

# Formulation
![image](https://github.com/leostre/modernopt/assets/103892559/07cd1a5e-5da7-4d9a-90f4-1e00688f13d8)

The optimization task aims to minimize the MSE
while maintaining 
![image](https://github.com/leostre/modernopt/assets/103892559/262f40e0-8f5b-4b98-9a93-a46b5773d69d)

# Algos
Researched algorithms are as follows:
* Stochastic Gradient Descent - classic stochastic algorithm for differentiable functions
* Fish School - swarm algorithm
* Evolutionary optimization

The code is placed in src folder
Some experiments in experiments
Raw real spectra are in data

