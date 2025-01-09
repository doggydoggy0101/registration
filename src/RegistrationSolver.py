# Copyright 2024 Bang-Shien Chen.
# All rights reserved. See LICENSE for the license information.

import numpy as np

from src.registration.solver import AbstractSolver
from src.registration.linear import LinearRelaxationSolver
from src.registration.sdp import SdpSolver
from src.registration.rgd import RgdSolver
from src.registration.smr import SmrSolver
from src.registration.gnc import GncSolver
from src.registration.fracgm import FracgmSolver
from src.registration.utils import rot_and_t_to_vec, vec_to_rot_and_t
from src.utils import project, registrationHorn


class IrlsHornSolver(AbstractSolver):
    def __init__(self, max_iteration=1000, tolerance=1e-6, c=1, robust_type="GM"):
        assert robust_type == "TLS" or "GM", "Robust type must be TLS/GM."

        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c
        self.robust_type = robust_type

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Horn's approach
        rot, t = registrationHorn(pcd1, pcd2)
        res = self.compute_residuals(pcd1, pcd2, rot, t, noise_bound)
        prev_cost = np.sum(res * res)
        cost = 0.0

        for _ in range(self.max_iter):
            # weight update
            vec_w = self.compute_weights(res, self.c, self.robust_type)
            # variable update
            rot, t = registrationHorn(pcd1, pcd2, weight=vec_w)
            res = self.compute_residuals(pcd1, pcd2, rot, t, noise_bound)
            # stopping criteria
            cost = np.sum(res * res)
            if self.check_cost_convergence(cost, prev_cost):
                break
            prev_cost = cost

        return rot, t


class IrlsLinearSolver(LinearRelaxationSolver, AbstractSolver):
    def __init__(self, max_iteration=1000, tolerance=1e-6, c=1, robust_type="GM"):
        assert robust_type == "TLS" or "GM", "Robust type must be TLS/GM."

        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c
        self.robust_type = robust_type

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Horn's approach
        rot, t = registrationHorn(pcd1, pcd2)
        res = self.compute_residuals(pcd1, pcd2, rot, t, noise_bound)
        prev_cost = np.sum(res * res)
        cost = 0.0

        terms = self.compute_terms(pcd1, pcd2, noise_bound)
        x = rot_and_t_to_vec(rot, t)

        for _ in range(self.max_iter):
            # weight update
            mat_w, vec_w = self.compute_weighted_term(
                terms, x, self.c, self.robust_type
            )

            # NOTE: TLS extreme outlier cases when all the data are outliers.
            if sum(vec_w) == 0:
                break

            # variable update
            x = self.solve_quadratic_program(mat_w)
            # stopping criteria
            cost = x.T @ mat_w @ x
            if self.check_cost_convergence(cost, prev_cost):
                break
            prev_cost = cost

        rot, t = vec_to_rot_and_t(x)
        rot = project(rot)

        return rot, t


class IrlsSdpSolver(SdpSolver, AbstractSolver):
    def __init__(self, max_iteration=1000, tolerance=1e-6, c=1, robust_type="GM"):
        assert robust_type == "TLS" or "GM", "Robust type must be TLS/GM."

        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c
        self.robust_type = robust_type

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Horn's approach
        rot, t = registrationHorn(pcd1, pcd2)
        res = self.compute_residuals(pcd1, pcd2, rot, t, noise_bound)
        prev_cost = np.sum(res * res)
        cost = 0.0

        terms = self.compute_terms(pcd1, pcd2, noise_bound)
        x = rot_and_t_to_vec(rot, t)

        for _ in range(self.max_iter):
            # weight update
            mat_w, vec_w = self.compute_weighted_term(
                terms, x, self.c, self.robust_type
            )

            # NOTE: TLS extreme outlier cases when all the data are outliers.
            if sum(vec_w) == 0:
                break

            # variable update
            x = self.solve_semidefinite_program(mat_w, initial=x)
            # stopping criteria
            cost = x.T @ mat_w @ x
            if self.check_cost_convergence(cost, prev_cost):
                break
            prev_cost = cost

        rot, t = vec_to_rot_and_t(x)

        return rot, t


class IrlsRgdSolver(RgdSolver, AbstractSolver):
    def __init__(self, max_iteration=1000, tolerance=1e-6, c=1, robust_type="GM"):
        assert robust_type == "TLS" or "GM", "Robust type must be TLS/GM."

        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c
        self.robust_type = robust_type

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Horn's approach
        rot, t = registrationHorn(pcd1, pcd2)
        res = self.compute_residuals(pcd1, pcd2, rot, t, noise_bound)
        prev_cost = np.sum(res * res)
        cost = 0.0

        terms = self.compute_terms(pcd1, pcd2, noise_bound)
        x = rot_and_t_to_vec(rot, t)

        for _ in range(self.max_iter):
            # weight update
            mat_w, vec_w = self.compute_weighted_term(
                terms, x, self.c, self.robust_type
            )

            # NOTE: TLS extreme outlier cases when all the data are outliers.
            if sum(vec_w) == 0:
                break

            # variable update
            x = self.solve_riemannian_gradient_descent(mat_w, x)
            # stopping criteria
            cost = x.T @ mat_w @ x
            if self.check_cost_convergence(cost, prev_cost):
                break
            prev_cost = cost

        rot, t = vec_to_rot_and_t(x)

        return rot, t


class IrlsSmrSolver(SmrSolver, AbstractSolver):
    def __init__(self, max_iteration=1000, tolerance=1e-6, c=1, robust_type="GM"):
        assert robust_type == "TLS" or "GM", "Robust type must be TLS/GM."

        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c
        self.robust_type = robust_type

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Horn's approach
        rot, t = registrationHorn(pcd1, pcd2)
        res = self.compute_residuals(pcd1, pcd2, rot, t, noise_bound)
        prev_cost = np.sum(res * res)
        cost = 0.0

        terms = self.compute_terms(pcd1, pcd2, noise_bound)
        x = rot_and_t_to_vec(rot, t)

        for _ in range(self.max_iter):
            # weight update
            mat_w, vec_w = self.compute_weighted_term(
                terms, x, self.c, self.robust_type
            )

            # NOTE: TLS extreme outlier cases when all the data are outliers.
            if sum(vec_w) == 0:
                break

            # variable update
            x = self.solve_stiefel_manifold(mat_w, initial=x)
            # stopping criteria
            cost = x.T @ mat_w @ x
            if self.check_cost_convergence(cost, prev_cost):
                break
            prev_cost = cost

        rot, t = vec_to_rot_and_t(x)
        rot = project(rot)

        return rot, t


class GncHornSolver(GncSolver, AbstractSolver):
    def __init__(
        self,
        max_iteration=1000,
        tolerance=1e-6,
        c=1,
        robust_type="GM",
        gnc_factor=1.4,
        weight_tolerace=1e-4,
    ):
        assert robust_type == "TLS" or "GM", "Robust type must be TLS/GM."

        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c
        self.robust_type = robust_type
        self.gnc_factor = gnc_factor
        self.weight_tol = weight_tolerace

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Horn's approach
        rot, t = registrationHorn(pcd1, pcd2)
        res = self.compute_residuals(pcd1, pcd2, rot, t, noise_bound)
        prev_cost = np.sum(res * res)
        cost = 0.0

        # initial surrogate parameter
        mu = self.compute_initial_mu(res, self.c, self.robust_type)

        for _ in range(self.max_iter):
            # weight update
            vec_w = self.compute_weights(res, self.c, self.robust_type, mu)
            # variable update
            rot, t = registrationHorn(pcd1, pcd2, weight=vec_w)
            res = self.compute_residuals(pcd1, pcd2, rot, t, noise_bound)
            # stopping criteria
            cost = np.sum(res * res)
            if (
                self.check_cost_convergence(cost, prev_cost)
                or self.check_mu_convergence(mu, self.robust_type)
                or self.check_weight_convergence(vec_w, self.robust_type)
            ):
                break
            prev_cost = cost
            # surrogate update
            mu = self.update_mu(mu, self.robust_type, self.gnc_factor)

        return rot, t


class GncLinearSolver(GncSolver, LinearRelaxationSolver, AbstractSolver):
    def __init__(
        self,
        max_iteration=1000,
        tolerance=1e-6,
        c=1,
        robust_type="GM",
        gnc_factor=1.4,
        weight_tolerace=1e-4,
    ):
        assert robust_type == "TLS" or "GM", "Robust type must be TLS/GM."

        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c
        self.robust_type = robust_type
        self.gnc_factor = gnc_factor
        self.weight_tol = weight_tolerace

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Horn's approach
        rot, t = registrationHorn(pcd1, pcd2)
        res = self.compute_residuals(pcd1, pcd2, rot, t, noise_bound)
        prev_cost = np.sum(res * res)
        cost = 0.0

        terms = self.compute_terms(pcd1, pcd2, noise_bound)
        x = rot_and_t_to_vec(rot, t)

        # initial surrogate parameter
        mu = self.compute_initial_mu(res, self.c, self.robust_type)

        for _ in range(self.max_iter):
            # weight update
            mat_w, vec_w = self.compute_weighted_term(
                terms, x, self.c, self.robust_type, mu
            )

            # NOTE: TLS extreme outlier cases when all the data are outliers.
            if sum(vec_w) == 0:
                break

            # variable update
            x = self.solve_quadratic_program(mat_w)
            # stopping criteria
            cost = x.T @ mat_w @ x
            if (
                self.check_cost_convergence(cost, prev_cost)
                or self.check_mu_convergence(mu, self.robust_type)
                or self.check_weight_convergence(vec_w, self.robust_type)
            ):
                break
            prev_cost = cost
            # surrogate update
            mu = self.update_mu(mu, self.robust_type, self.gnc_factor)

        rot, t = vec_to_rot_and_t(x)
        rot = project(rot)

        return rot, t


class GncSdpSolver(GncSolver, SdpSolver, AbstractSolver):
    def __init__(
        self,
        max_iteration=1000,
        tolerance=1e-6,
        c=1,
        robust_type="GM",
        gnc_factor=1.4,
        weight_tolerace=1e-4,
    ):
        assert robust_type == "TLS" or "GM", "Robust type must be TLS/GM."

        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c
        self.robust_type = robust_type
        self.gnc_factor = gnc_factor
        self.weight_tol = weight_tolerace

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Horn's approach
        rot, t = registrationHorn(pcd1, pcd2)
        res = self.compute_residuals(pcd1, pcd2, rot, t, noise_bound)
        prev_cost = np.sum(res * res)
        cost = 0.0

        terms = self.compute_terms(pcd1, pcd2, noise_bound)
        x = rot_and_t_to_vec(rot, t)

        # initial surrogate parameter
        mu = self.compute_initial_mu(res, self.c, self.robust_type)

        for _ in range(self.max_iter):
            # weight update
            mat_w, vec_w = self.compute_weighted_term(
                terms, x, self.c, self.robust_type, mu
            )

            # NOTE: TLS extreme outlier cases when all the data are outliers.
            if sum(vec_w) == 0:
                break

            # variable update
            x = self.solve_semidefinite_program(mat_w, initial=x)
            # stopping criteria
            cost = x.T @ mat_w @ x
            if (
                self.check_cost_convergence(cost, prev_cost)
                or self.check_mu_convergence(mu, self.robust_type)
                or self.check_weight_convergence(vec_w, self.robust_type)
            ):
                break
            prev_cost = cost
            # surrogate update
            mu = self.update_mu(mu, self.robust_type, self.gnc_factor)

        rot, t = vec_to_rot_and_t(x)

        return rot, t


class GncRgdSolver(GncSolver, RgdSolver, AbstractSolver):
    def __init__(
        self,
        max_iteration=1000,
        tolerance=1e-6,
        c=1,
        robust_type="GM",
        gnc_factor=1.4,
        weight_tolerace=1e-4,
    ):
        assert robust_type == "TLS" or "GM", "Robust type must be TLS/GM."

        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c
        self.robust_type = robust_type
        self.gnc_factor = gnc_factor
        self.weight_tol = weight_tolerace

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Horn's approach
        rot, t = registrationHorn(pcd1, pcd2)
        res = self.compute_residuals(pcd1, pcd2, rot, t, noise_bound)
        prev_cost = np.sum(res * res)
        cost = 0.0

        terms = self.compute_terms(pcd1, pcd2, noise_bound)
        x = rot_and_t_to_vec(rot, t)

        # initial surrogate parameter
        mu = self.compute_initial_mu(res, self.c, self.robust_type)

        for _ in range(self.max_iter):
            # weight update
            mat_w, vec_w = self.compute_weighted_term(
                terms, x, self.c, self.robust_type, mu
            )

            # NOTE: TLS extreme outlier cases when all the data are outliers.
            if sum(vec_w) == 0:
                break

            # variable update
            x = self.solve_riemannian_gradient_descent(mat_w, x)
            # stopping criteria
            cost = x.T @ mat_w @ x
            if (
                self.check_cost_convergence(cost, prev_cost)
                or self.check_mu_convergence(mu, self.robust_type)
                or self.check_weight_convergence(vec_w, self.robust_type)
            ):
                break
            prev_cost = cost
            # surrogate update
            mu = self.update_mu(mu, self.robust_type, self.gnc_factor)

        rot, t = vec_to_rot_and_t(x)

        return rot, t


class GncSmrSolver(GncSolver, SmrSolver, AbstractSolver):
    def __init__(
        self,
        max_iteration=1000,
        tolerance=1e-6,
        c=1,
        robust_type="GM",
        gnc_factor=1.4,
        weight_tolerace=1e-4,
    ):
        assert robust_type == "TLS" or "GM", "Robust type must be TLS/GM."

        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c
        self.robust_type = robust_type
        self.gnc_factor = gnc_factor
        self.weight_tol = weight_tolerace

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Horn's approach
        rot, t = registrationHorn(pcd1, pcd2)
        res = self.compute_residuals(pcd1, pcd2, rot, t, noise_bound)
        prev_cost = np.sum(res * res)
        cost = 0.0

        terms = self.compute_terms(pcd1, pcd2, noise_bound)
        x = rot_and_t_to_vec(rot, t)

        # initial surrogate parameter
        mu = self.compute_initial_mu(res, self.c, self.robust_type)

        for _ in range(self.max_iter):
            # weight update
            mat_w, vec_w = self.compute_weighted_term(
                terms, x, self.c, self.robust_type, mu
            )

            # NOTE: TLS extreme outlier cases when all the data are outliers.
            if sum(vec_w) == 0:
                break

            # variable update
            x = self.solve_stiefel_manifold(mat_w, initial=x)
            # stopping criteria
            cost = x.T @ mat_w @ x
            if (
                self.check_cost_convergence(cost, prev_cost)
                or self.check_mu_convergence(mu, self.robust_type)
                or self.check_weight_convergence(vec_w, self.robust_type)
            ):
                break
            prev_cost = cost
            # surrogate update
            mu = self.update_mu(mu, self.robust_type, self.gnc_factor)

        rot, t = vec_to_rot_and_t(x)
        rot = project(rot)

        return rot, t


class FracgmLinearSolver(FracgmSolver, LinearRelaxationSolver, AbstractSolver):
    def __init__(self, max_iteration=1000, tolerance=1e-6, c=1):
        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Horn's approach
        rot, t = registrationHorn(pcd1, pcd2)

        terms = self.compute_fractional_terms(pcd1, pcd2, noise_bound, self.c)
        x = rot_and_t_to_vec(rot, t)

        for _ in range(self.max_iter):
            # auxiliary variables update
            beta, mu = self.update_auxiliary_variables(terms)
            mat_w = self.compute_weighted_fractional_term(terms, beta, mu)
            # variable update
            x = self.solve_quadratic_program(mat_w)
            self.update_fractional_terms_cache(terms, x)
            # stopping criteria
            psi_norm = self.compute_psi_norm(terms, beta, mu)
            if psi_norm < self.tol:
                break

        rot, t = vec_to_rot_and_t(x)
        rot = project(rot)

        return rot, t


class FracgmSdpSolver(FracgmSolver, SdpSolver, AbstractSolver):
    def __init__(self, max_iteration=1000, tolerance=1e-6, c=1):
        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Horn's approach
        rot, t = registrationHorn(pcd1, pcd2)

        terms = self.compute_fractional_terms(pcd1, pcd2, noise_bound, self.c)
        x = rot_and_t_to_vec(rot, t)

        for _ in range(self.max_iter):
            # auxiliary variables update
            beta, mu = self.update_auxiliary_variables(terms)
            mat_w = self.compute_weighted_fractional_term(terms, beta, mu)
            # variable update
            x = self.solve_semidefinite_program(mat_w, initial=x)
            self.update_fractional_terms_cache(terms, x)
            # stopping criteria
            psi_norm = self.compute_psi_norm(terms, beta, mu)
            if psi_norm < self.tol:
                break

        rot, t = vec_to_rot_and_t(x)

        return rot, t


class FracgmRgdSolver(FracgmSolver, RgdSolver, AbstractSolver):
    def __init__(self, max_iteration=1000, tolerance=1e-6, c=1):
        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Horn's approach
        rot, t = registrationHorn(pcd1, pcd2)

        terms = self.compute_fractional_terms(pcd1, pcd2, noise_bound, self.c)
        x = rot_and_t_to_vec(rot, t)

        for _ in range(self.max_iter):
            # auxiliary variables update
            beta, mu = self.update_auxiliary_variables(terms)
            mat_w = self.compute_weighted_fractional_term(terms, beta, mu)
            # variable update
            x = self.solve_riemannian_gradient_descent(mat_w, x)
            self.update_fractional_terms_cache(terms, x)
            # stopping criteria
            psi_norm = self.compute_psi_norm(terms, beta, mu)
            if psi_norm < self.tol:
                break

        rot, t = vec_to_rot_and_t(x)

        return rot, t


class FracgmSmrSolver(FracgmSolver, SmrSolver, AbstractSolver):
    def __init__(self, max_iteration=1000, tolerance=1e-6, c=1):
        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Horn's approach
        rot, t = registrationHorn(pcd1, pcd2)

        terms = self.compute_fractional_terms(pcd1, pcd2, noise_bound, self.c)
        x = rot_and_t_to_vec(rot, t)

        for _ in range(self.max_iter):
            # auxiliary variables update
            beta, mu = self.update_auxiliary_variables(terms)
            mat_w = self.compute_weighted_fractional_term(terms, beta, mu)
            # variable update
            x = self.solve_stiefel_manifold(mat_w, x)
            self.update_fractional_terms_cache(terms, x)
            # stopping criteria
            psi_norm = self.compute_psi_norm(terms, beta, mu)
            if psi_norm < self.tol:
                break

        rot, t = vec_to_rot_and_t(x)
        rot = project(rot)

        return rot, t


class GncFracgmLinearSolver(
    GncSolver, FracgmSolver, LinearRelaxationSolver, AbstractSolver
):
    def __init__(
        self,
        max_iteration=1000,
        tolerance=1e-6,
        c=1,
        gnc_factor=1.4,
        weight_tolerace=1e-4,
    ):
        super().__init__()
        self.max_iter = max_iteration
        self.tol = tolerance
        self.c = c
        self.gnc_factor = gnc_factor
        self.weight_tol = weight_tolerace

    def solve(self, pcd1, pcd2, noise_bound=0.1):
        # initial guess by Horn's approach
        rot, t = registrationHorn(pcd1, pcd2)
        res = self.compute_residuals(pcd1, pcd2, rot, t, noise_bound)

        terms = self.compute_fractional_terms(pcd1, pcd2, noise_bound, self.c)
        x = rot_and_t_to_vec(rot, t)

        # initial surrogate parameter
        gnc_mu = self.compute_initial_mu(res, self.c, robust_type="GM")

        for _ in range(self.max_iter):
            # auxiliary variables update
            beta, mu = self.update_auxiliary_variables(terms)
            mat_w = self.compute_weighted_fractional_term(terms, beta, mu, gnc_mu)
            # variable update
            x = self.solve_quadratic_program(mat_w)
            self.update_fractional_terms_cache(terms, x)
            # stopping criteria
            psi_norm = self.compute_psi_norm(terms, beta, mu)
            if psi_norm < self.tol:
                break

            if not self.check_mu_convergence(gnc_mu, robust_type="GM"):
                gnc_mu = self.update_mu(
                    gnc_mu, robust_type="GM", gnc_factor=self.gnc_factor
                )

        rot, t = vec_to_rot_and_t(x)
        rot = project(rot)

        return rot, t
