'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/19-2024/06/10 Copyright 2023 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import numpy as np
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp


class LinearProgrammingDisc(SoSWrapp):
    """Linear Programming Discipline using PuLP optimization."""

    # ontology information
    _ontology_data = {
        'label': 'Linear Programming Discipline',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Team',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': 'fas fa-calculator fa-fw',
        'version': '',
    }

    DESC_IN = {
        'objective_coeffs': {
            'type': 'array',
            'visibility': 'Shared',
            'namespace': 'ns_linear_prog',
            'description': 'Coefficients for the objective function'
        },
        'constraint_matrix': {
            'type': 'array', 
            'visibility': 'Shared',
            'namespace': 'ns_linear_prog',
            'description': 'Constraint matrix A where Ax <= b'
        },
        'constraint_bounds': {
            'type': 'array',
            'visibility': 'Shared', 
            'namespace': 'ns_linear_prog',
            'description': 'Right-hand side bounds b for constraints'
        },
        'variable_bounds': {
            'type': 'array',
            'visibility': 'Shared',
            'namespace': 'ns_linear_prog', 
            'description': 'Bounds for variables [(low1, high1), (low2, high2), ...]'
        },
        'solver_name': {
            'type': 'string',
            'default': 'PULP_CBC_CMD',
            'visibility': 'Shared',
            'namespace': 'ns_linear_prog',
            'description': 'Solver to use for optimization',
            'possible_values': ['PULP_CBC_CMD', 'HiGHS', 'GLPK_CMD', 'CPLEX_CMD', 'GUROBI_CMD']
        }
    }

    DESC_OUT = {
        'optimal_solution': {
            'type': 'array',
            'visibility': 'Shared',
            'namespace': 'ns_linear_prog',
            'description': 'Optimal values for the decision variables'
        },
        'optimal_value': {
            'type': 'float',
            'visibility': 'Shared', 
            'namespace': 'ns_linear_prog',
            'description': 'Optimal objective function value'
        },
        'solver_status': {
            'type': 'string',
            'visibility': 'Shared',
            'namespace': 'ns_linear_prog', 
            'description': 'Status of the optimization (Optimal, Infeasible, etc.)'
        }
    }

    def run(self):
        """Execute the linear programming optimization."""
        
        # Import PuLP
        try:
            import pulp
        except ImportError:
            raise ImportError("PuLP library is required for linear programming optimization")

        # Get input data
        objective_coeffs = self.get_sosdisc_inputs('objective_coeffs')
        constraint_matrix = self.get_sosdisc_inputs('constraint_matrix')
        constraint_bounds = self.get_sosdisc_inputs('constraint_bounds')
        variable_bounds = self.get_sosdisc_inputs('variable_bounds')
        solver_name = self.get_sosdisc_inputs('solver_name')

        # Validate inputs
        n_vars = len(objective_coeffs)
        if constraint_matrix.shape[1] != n_vars:
            raise ValueError("Constraint matrix columns must match number of variables")
        if len(constraint_bounds) != constraint_matrix.shape[0]:
            raise ValueError("Constraint bounds length must match constraint matrix rows")
        if len(variable_bounds) != n_vars:
            raise ValueError("Variable bounds length must match number of variables")

        # Create PuLP problem
        prob = pulp.LpProblem("LinearProgram", pulp.LpMinimize)

        # Create decision variables
        variables = []
        for i in range(n_vars):
            low_bound, up_bound = variable_bounds[i]
            var = pulp.LpVariable(f"x_{i}", 
                                lowBound=low_bound if low_bound != -np.inf else None,
                                upBound=up_bound if up_bound != np.inf else None,
                                cat='Continuous')
            variables.append(var)

        # Add objective function
        prob += pulp.lpSum([objective_coeffs[i] * variables[i] for i in range(n_vars)]), "Objective"

        # Add constraints
        for i in range(constraint_matrix.shape[0]):
            constraint = pulp.lpSum([constraint_matrix[i, j] * variables[j] for j in range(n_vars)])
            prob += constraint <= constraint_bounds[i], f"Constraint_{i}"

        # Select solver
        solver_dict = {
            'PULP_CBC_CMD': pulp.PULP_CBC_CMD,
            'HiGHS': pulp.HiGHS_CMD,
            'GLPK_CMD': pulp.GLPK_CMD,
            'CPLEX_CMD': pulp.CPLEX_CMD,
            'GUROBI_CMD': pulp.GUROBI_CMD
        }

        try:
            solver = solver_dict.get(solver_name, pulp.PULP_CBC_CMD)()
        except Exception:
            # Fallback to default solver if specified solver is not available
            solver = pulp.PULP_CBC_CMD()

        # Solve the problem
        prob.solve(solver)

        # Extract results
        status_dict = {
            pulp.LpStatusOptimal: "Optimal",
            pulp.LpStatusNotSolved: "Not Solved", 
            pulp.LpStatusInfeasible: "Infeasible",
            pulp.LpStatusUnbounded: "Unbounded",
            pulp.LpStatusUndefined: "Undefined"
        }

        solver_status = status_dict.get(prob.status, "Unknown")
        
        if prob.status == pulp.LpStatusOptimal:
            optimal_solution = np.array([var.varValue for var in variables])
            optimal_value = pulp.value(prob.objective)
        else:
            optimal_solution = np.zeros(n_vars)
            optimal_value = np.nan

        # Set output values
        self.store_sos_outputs_values({
            'optimal_solution': optimal_solution,
            'optimal_value': optimal_value,
            'solver_status': solver_status
        })
