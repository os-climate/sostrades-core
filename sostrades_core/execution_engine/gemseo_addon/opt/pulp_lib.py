'''
PuLP optimization library for GEMSEO.
'''

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy import array

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem

# Try to import PuLP
try:
    import pulp
    from pulp import (
        CPLEX_CMD,
        GLPK_CMD,
        GUROBI_CMD,
        HiGHS_CMD,
        PULP_CBC_CMD,
        LpMinimize,
        LpMaximize,
        LpProblem,
        LpStatus,
        LpVariable,
        lpSum,
    )
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    # Create dummy classes to avoid import errors
    PULP_CBC_CMD = GLPK_CMD = CPLEX_CMD = GUROBI_CMD = HiGHS_CMD = None
    LpMinimize = LpMaximize = LpProblem = LpStatus = LpVariable = lpSum = None

from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.design_space_utils import get_value_and_bounds


class PuLPOpt(BaseOptimizationLibrary):
    """PuLP optimization library wrapper for GEMSEO."""

    # Algorithm descriptions
    ALGORITHM_INFOS: ClassVar[dict[str, OptimizationAlgorithmDescription]] = {
        "PULP_CBC_CMD": OptimizationAlgorithmDescription(
            algorithm_name="PULP_CBC_CMD",
            description="CBC solver via PuLP",
            internal_algorithm_name="PULP_CBC_CMD",
            library_name="PuLP",
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            for_linear_problems=False,
        ),
        "HiGHS_CMD": OptimizationAlgorithmDescription(
            algorithm_name="HiGHS_CMD", 
            description="HiGHS solver via PuLP",
            internal_algorithm_name="HiGHS_CMD",
            library_name="PuLP",
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            for_linear_problems=True,
        ),
        "GLPK_CMD": OptimizationAlgorithmDescription(
            algorithm_name="GLPK_CMD",
            description="GLPK solver via PuLP",
            internal_algorithm_name="GLPK_CMD", 
            library_name="PuLP",
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            for_linear_problems=True,
        ),
        "CPLEX_CMD": OptimizationAlgorithmDescription(
            algorithm_name="CPLEX_CMD",
            description="IBM CPLEX solver via PuLP",
            internal_algorithm_name="CPLEX_CMD",
            library_name="PuLP",
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            for_linear_problems=True,
        ),
        "GUROBI_CMD": OptimizationAlgorithmDescription(
            algorithm_name="GUROBI_CMD",
            description="Gurobi solver via PuLP",
            internal_algorithm_name="GUROBI_CMD",
            library_name="PuLP",
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            for_linear_problems=True,
        ),
    }

    def __init__(self, algo_name: str = "PULP_CBC_CMD") -> None:
        """Initialize the PuLP optimization library."""
        super().__init__(algo_name)
        if not PULP_AVAILABLE:
            msg = "PuLP is not available."
            raise ImportError(msg)

    def _run(self, problem: OptimizationProblem, **settings: Any):
        """
        Run the PuLP optimization algorithm.

        Args:
            problem: The optimization problem to solve.
            **settings: The algorithm settings.

        Returns:
            Tuple containing (message, status)
        """
        # Create PuLP problem (LpMaximize for maximization problems)
        sense = LpMaximize if getattr(problem, 'minimize_objective', True) == False else LpMinimize
        pulp_prob = LpProblem("PuLP_Optimization", sense)
        
        # Get design variables bounds using GEMSEO 6 API
        _, l_b, u_b = get_value_and_bounds(problem.design_space, normalize_ds=False)
        variable_names = problem.design_space.variable_names
        
        # Create PuLP variables
        pulp_vars = {}
        for i, var_name in enumerate(variable_names):
            lower = l_b[i] if l_b[i] is not None else None
            upper = u_b[i] if u_b[i] is not None else None
            pulp_vars[var_name] = LpVariable(
                var_name,
                lowBound=lower,
                upBound=upper,
                cat='Continuous'
            )
        
        # Get objective function coefficients
        # For linear problems, we extract coefficients from the objective gradient
        x_current = problem.design_space.get_current_value()
        
        # Evaluate objective to get the linear coefficients
        # For a linear objective f(x) = c^T * x, the gradient is c
        try:
            # Try to get analytical gradient if available
            obj_jac = problem.objective.jac(x_current)
            if obj_jac is not None:
                obj_coeffs = obj_jac.flatten()
            else:
                # Fallback: use finite differences to approximate gradient
                obj_coeffs = self._approximate_linear_coefficients(problem.objective, x_current)
        except:
            # Last resort: assume unit coefficients
            obj_coeffs = np.ones(len(variable_names))
        
        # Build objective function
        objective_expr = lpSum([obj_coeffs[i] * pulp_vars[var_name] 
                               for i, var_name in enumerate(variable_names)])
        pulp_prob += objective_expr
        
        # Add constraints
        for constraint in problem.constraints:
            try:
                # Get constraint gradient to extract linear coefficients
                constr_jac = constraint.jac(x_current)
                if constr_jac is not None:
                    constr_coeffs = constr_jac.flatten()
                else:
                    # Approximate if analytical gradient not available
                    constr_coeffs = self._approximate_linear_coefficients(constraint, x_current)
                
                # Build constraint expression
                constraint_expr = lpSum([constr_coeffs[i] * pulp_vars[var_name] 
                                       for i, var_name in enumerate(variable_names)])
                
                # Evaluate constraint at current point to get the constant term
                constraint_value = constraint.evaluate(x_current)[0]
                constant_term = constraint_value - np.dot(constr_coeffs, x_current)
                
                # Add constraint (assuming <= 0 format for inequality constraints)
                # constraint_expr + constant_term <= 0
                pulp_prob += constraint_expr + constant_term <= 0
                
            except Exception as e:
                print(f"Warning: Could not add constraint {constraint.name}: {e}")
                continue
        
        # Get solver from algorithm name
        solver_map = {
            'PULP_CBC_CMD': PULP_CBC_CMD,
            'HiGHS_CMD': HiGHS_CMD,
            'GLPK_CMD': GLPK_CMD,
            'CPLEX_CMD': CPLEX_CMD,
            'GUROBI_CMD': GUROBI_CMD
        }
        
        solver_class = solver_map.get(self._algo_name, PULP_CBC_CMD)
        solver = solver_class(msg=1)  # Enable solver messages
        
        # Solve the problem
        pulp_prob.solve(solver)
        
        # Extract results
        status_text = LpStatus[pulp_prob.status]
        success = pulp_prob.status == 1  # 1 means optimal solution found
        
        if success:
            # Get optimal solution
            x_opt = np.array([pulp_vars[var_name].varValue or 0.0 for var_name in variable_names])
            
            # Store results in the problem's database
            obj_value = problem.objective.evaluate(x_opt)
            constraint_values = {}
            
            for constraint in problem.constraints:
                try:
                    constraint_values[constraint.name] = constraint.evaluate(x_opt)
                except:
                    constraint_values[constraint.name] = np.array([0.0])
            
            # Combine all function values
            all_values = {problem.objective.name: obj_value}
            all_values.update(constraint_values)
            
            # Store in database
            problem.database.store(x_opt, all_values)
        
        return status_text, success

    def _approximate_linear_coefficients(self, function, x_current):
        """
        Approximate linear coefficients using finite differences.
        
        For a linear function f(x) = c^T * x + d, the gradient is c.
        """
        n = len(x_current)
        coeffs = np.zeros(n)
        h = 1e-8  # Step size for finite differences
        
        f_base = function.evaluate(x_current)[0]
        
        for i in range(n):
            x_pert = x_current.copy()
            x_pert[i] += h
            f_pert = function.evaluate(x_pert)[0]
            coeffs[i] = (f_pert - f_base) / h
            
        return coeffs
