from __future__ import annotations
 
from typing import Annotated, ClassVar
 
import numpy as np
from pydantic import Field
 
from simopt.base import (
    ConstraintType,
    Context,
    ObjectiveType,
    Problem,
    Solver,
    SolverConfig,
    VariableType,
)
def my_RBF_kernel(A, B, l=1.0):
    # squared distance between every row of A and every row of B
    diff = A[:, None, :] - B[None, :, :]   # shape (n, m, d)
    dist2 = np.sum(diff ** 2, axis=-1)      # shape (n, m)
    return np.exp(-dist2 / (2 * l ** 2))

def gp_predict(X_train, y_train, X_star, l=1.0, noise=1e-8):
    # Training kernel
    K = my_RBF_kernel(X_train, X_train, l)
    K += noise * np.eye(len(X_train))

    # K^{-1}
    K_inv = np.linalg.inv(K)

    # alpha = K^{-1} y
    alpha = K_inv @ y_train

    # Cross-kernel
    K_star = my_RBF_kernel(X_star, X_train, l)

    # Test kernel
    K_starstar = my_RBF_kernel(X_star, X_star, l)

    # Predictive mean
    mean = K_star @ alpha

    # Predictive covariance
    cov = K_starstar - K_star @ K_inv @ K_star.T

    # Predictive std
    std = np.sqrt(np.clip(np.diag(cov), 0, None))

    return mean, std

 
 
class BAYESOPTConfig(SolverConfig):
 
    sample_size: Annotated[int, Field(default=10, gt=0, description="sample size per solution")]
    n_candidates: Annotated[
        int, Field(default=1000, gt=0, description="number of random candidates per iteration")
    ]
    delta: Annotated[
        float,
        Field(default=0.1, gt=0, lt=1, description="confidence parameter for GP-UCB beta_t"),
    ]
    search_radius: Annotated[
        float,
        Field(
            default=0.5,
            gt=0,
            description=(
                "fallback half-width used to sample candidates along any dimension "
                "where the problem's bounds are unbounded (+/- inf), centered on the "
                "current incumbent"
            ),
        ),
    ]
 
 
 
class BAYESOPT(Solver):
 
    name: str = "BAYESOPT"
    config_class: ClassVar[type[SolverConfig]] = BAYESOPTConfig
    class_name_abbr: ClassVar[str] = "BAYESOPT"
    class_name: ClassVar[str] = "Bayesian Optimization"
    objective_type: ClassVar[ObjectiveType] = ObjectiveType.SINGLE
    constraint_type: ClassVar[ConstraintType] = ConstraintType.UNCONSTRAINED
    variable_type: ClassVar[VariableType] = VariableType.CONTINUOUS
    gradient_needed: ClassVar[bool] = False
 
    @staticmethod
    def beta_t(t: int, delta: float) -> float:
        #Compute the GP-UCB exploration coefficient for iteration t.
        return 2 * np.log((t**2 * np.pi**2) / (3 * delta))
 
    @staticmethod
    def gp_ucb(mean: np.ndarray, std: np.ndarray, beta: float) -> np.ndarray:
        #Evaluate the GP-UCB acquisition function
        return mean - beta * std
 
    def propose_candidate(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        t: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        incumbent_x: np.ndarray,
        search_radius: float,
        n_candidates: int,
        delta: float,
        candidate_rng,
    ) -> tuple:

        # Fall back to a finite window centered on the incumbent for any
        # dimension where the problem itself imposes no finite bound.
        effective_lower = np.where(
            np.isinf(lower_bounds), incumbent_x - search_radius, lower_bounds
        )
        effective_upper = np.where(
            np.isinf(upper_bounds), incumbent_x + search_radius, upper_bounds
        )
        # 1. Sample random candidate points within the effective bounds.
        dim = len(effective_lower)
        x_cand = np.array(
            [
                [candidate_rng.uniform(effective_lower[j], effective_upper[j]) for j in range(dim)]
                for _ in range(n_candidates)
            ]
        )
        # 2. GP predicts mean and std at every candidate.
        mean, std = gp_predict(x_train, y_train, x_cand)
        # 3. Evaluate acquisition at every candidate.
        acq = self.gp_ucb(mean, std, beta=self.beta_t(t, delta))
        # 4. x* = argmax a(x).
        x_star = x_cand[np.argmax(acq)]
        return tuple(x_star)
 
    def solve(self, problem: Problem, ctx: Context) -> None:
        # Designate random number generator for candidate sampling.
        find_next_soln_rng = self.rng_list[1]
        # Bounds for each decision variable (may be +/- inf if unconstrained).
        lower_bounds = np.array(problem.lower_bounds)
        upper_bounds = np.array(problem.upper_bounds)
        # Start at initial solution and record as best.
        new_x = problem.factors["initial_solution"]
        new_solution = ctx.evaluate(new_x, 0)
        best_solution = new_solution
        ctx.log(new_solution)
        # Prepare other variables in the loop.
        sample_size = self.factors["sample_size"]
        n_candidates = self.factors["n_candidates"]
        delta = self.factors["delta"]
        search_radius = self.factors["search_radius"]
        # GP training set, built up as solutions are actually replicated.
        x_train = np.empty((0, problem.dim))
        y_train = np.empty(0)
        # Sequentially propose solutions via GP-UCB and simulate them.
        t = 0
        while True:
            t += 1
            # Request budget first, then simulate new solution.
            new_solution = ctx.evaluate(new_solution, sample_size)
            print(new_solution.x)

        
            # Add the newly observed solution to the GP training set.
            x_train = np.vstack([x_train, new_solution.x])
            y_train = np.append(y_train, new_solution.objectives_mean[0])
            # Check for improvement relative to incumbent best solution.
            mean_diff = new_solution.objectives_mean - best_solution.objectives_mean

            if all(mean_diff < 0):
                best_solution = new_solution
                ctx.log(new_solution)
                
            # Identify new solution to simulate for next iteration via GP-UCB.
            new_x = self.propose_candidate(
                x_train,
                y_train,
                t,
                lower_bounds,
                upper_bounds,
                np.array(best_solution.x),
                search_radius,
                n_candidates,
                delta,
                find_next_soln_rng,
            )
            new_solution = ctx.evaluate(new_x, 0)

            print(ctx.recommended_solns)
           
 


