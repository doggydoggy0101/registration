# Robust registration solvers

[![Python Formatter](https://img.shields.io/badge/Python_Formatter-ruff-black?style=flat-square)](https://github.com/astral-sh/ruff)

Install [Python](https://www.python.org/downloads/). 
```shell
pip install numpy scipy
pip install cvxpy "cvxpy[MOSEK]"
```
Least square solvers `Sdp` and `Smr` requires cvxpy's mosek.

|                    | Robust solver | Least squares solver        | Robust function |
|--------------------|---------------|-----------------------------|-----------------|
| IrlsHornSolver     | IRLS          | Horn                        | TLS, GM         |
| IrlsLinearSolver   | IRLS          | Linear relaxation           | TLS, GM         |
| IrlsSdpSolver      | IRLS          | Semidefinite program        | TLS, GM         |
| IrlsRgdSolver      | IRLS          | Riemannian gradient descent | TLS, GM         |
| IrlsSmrSolver      | IRLS          | Stiefel manifold relaxation | TLS, GM         |
| GncHornSolver      | GNC           | Horn                        | TLS, GM         |
| GncLinearSolver    | GNC           | Linear relaxation           | TLS, GM         |
| GncSdpSolver       | GNC           | Semidefinite program        | TLS, GM         |
| GncRgdSolver       | GNC           | Riemannian gradient descent | TLS, GM         |
| GncSmrSolver       | GNC           | Stiefel manifold relaxation | TLS, GM         |
| FracgmLinearSolver | FracGM        | Linear relaxation           | GM              |
| FracgmSdpSolver    | FracGM        | Semidefinite program        | GM              |
| FracgmRgdSolver    | FracGM        | Riemannian gradient descent | GM              |
| FracgmSmrSolver    | FracGM        | Stiefel manifold relaxation | GM              |