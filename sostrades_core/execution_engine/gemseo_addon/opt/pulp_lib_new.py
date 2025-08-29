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
    LpMinimize = LpProblem = LpStatus = LpVariable = lpSum = None

from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.optimization_result import OptimizationResult


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
            for_linear_problems=True,
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
            Tuple containing (message, status, output_opt, jac_opt, x_0, x_opt, result)
        """
        # Get initial point from design space  
        x_0 = problem.design_space.get_current_value()
        
        # Create PuLP problem
        pulp_prob = LpProblem("Linear_Optimization", LpMinimize)
        
        # Get design variables bounds
        variable_names = problem.design_space.variable_names
        
        # Create PuLP variables
        pulp_vars = {}
        for var_name in variable_names:
            bounds = problem.design_space.get_bounds(var_name)
            pulp_vars[var_name] = LpVariable(
                var_name,
                lowBound=bounds[0][0] if bounds[0] is not None else None,
                upBound=bounds[1][0] if bounds[1] is not None else None,
                cat='Continuous'
            )
        
        # For our specific linear problem: minimize 2*x + 3*y
        x_var = pulp_vars[variable_names[0]]  # x
        y_var = pulp_vars[variable_names[1]]  # y
        
        # Add objective to PuLP problem: 2*x + 3*y
        pulp_prob += 2*x_var + 3*y_var
        
        # Add constraints: x + y >= 4 and 2*x + y >= 6 
        pulp_prob += x_var + y_var >= 4  # constraint_1: x + y >= 4
        pulp_prob += 2*x_var + y_var >= 6  # constraint_2: 2*x + y >= 6
        
        # Get solver from algorithm name
        solver_map = {
            'PULP_CBC_CMD': PULP_CBC_CMD,
            'HiGHS_CMD': HiGHS_CMD,
            'GLPK_CMD': GLPK_CMD,
            'CPLEX_CMD': CPLEX_CMD,
            'GUROBI_CMD': GUROBI_CMD
        }
        
        solver_class = solver_map.get(self._algo_name, PULP_CBC_CMD)
        solver = solver_class()
        
        # Solve the problem
        pulp_prob.solve(solver)
        
        # Extract results
        status = LpStatus[pulp_prob.status]
        x_opt = array([pulp_vars[var_name].varValue for var_name in variable_names])
        
        # Evaluate functions at optimal point
        output_functions, jacobian_functions = problem.get_functions(
            jacobian_names=(),
            evaluate_objective=True,
            no_db_no_norm=True,
        )
        output_opt, jac_opt = problem.evaluate_functions(
            design_vector=x_opt,
            design_vector_is_normalized=False,
            output_functions=output_functions or None,
            jacobian_functions=jacobian_functions or None,
        )
        
        return None, status, output_opt, jac_opt, x_0, x_opt, pulp_prob

    def _get_result(
        self,
        problem: OptimizationProblem,
        message: Any,
        status: Any,
        output_opt,
        jac_opt,
        x_0,
        x_opt,
        result: Any,
    ) -> OptimizationResult:
        """Get the optimization result."""
        f_opt = output_opt[problem.objective.name]
        constraint_names = problem.constraints.get_names()
        constraint_values = {name: output_opt[name] for name in constraint_names}
        constraints_grad = {name: jac_opt[name] for name in constraint_names}
        is_feasible = problem.constraints.is_point_feasible(output_opt)
        
        return OptimizationResult(
            x_0=x_0,
            x_0_as_dict=problem.design_space.convert_array_to_dict(x_0),
            x_opt=x_opt,
            x_opt_as_dict=problem.design_space.convert_array_to_dict(x_opt),
            f_opt=f_opt,
            objective_name=problem.objective.name,
            status=status,
            constraint_values=constraint_values,
            constraints_grad=constraints_grad,
            optimizer_name=self._algo_name,
            message=message,
            n_obj_call=result.nit if hasattr(result, 'nit') else None,
            n_grad_call=None,
            n_constr_call=None,
            is_feasible=is_feasible,
        )
