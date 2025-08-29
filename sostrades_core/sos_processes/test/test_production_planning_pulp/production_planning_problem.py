#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Planning Problem Discipline

This module implements a production planning optimization problem for multi-product
manufacturing using linear programming formulation compatible with GEMSEO and PuLP.

The problem is a classic Operations Research case study commonly used in industry
for production planning and resource allocation optimization.
"""

import numpy as np
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp


class ProductionPlanningProblem(SoSWrapp):
    """
    Production Planning Problem - Classic Operations Research Problem
    
    Based on the multi-product manufacturing optimization problem commonly
    found in operations research literature and industrial applications.
    
    Problem formulation:
    Maximize: 40*x_A + 50*x_B + 30*x_C  (profit)
    
    Subject to:
    - Machine M1: 2*x_A + 3*x_B + 1*x_C <= 100 hours/week
    - Machine M2: 1*x_A + 2*x_B + 2*x_C <= 80 hours/week  
    - Storage:    x_A + x_B + x_C <= 50 units
    - Demand A:   x_A >= 5 units
    - Demand B:   x_B >= 10 units
    - Demand C:   x_C >= 8 units
    - x_A, x_B, x_C >= 0
    """
    
    # ontology information
    _ontology_data = {
        'label': 'Production Planning Problem',
        'description': 'Multi-product production planning optimization using linear programming',
        'category': 'Operations Research',
        'version': '',
        'validated': '',
        'source': 'Classic Operations Research Problem',
        'icon': 'fas fa-industry fa-fw',
        'type': 'Research',
    }

    DESC_IN = {
        # Production quantities (decision variables) - match design variables names
        'production_A': {'type': 'float', 'default': 20.0, 'unit': 'units', 
                        'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_production',
                        'description': 'Quantity of product A to produce'},
        'production_B': {'type': 'float', 'default': 15.0, 'unit': 'units',
                        'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_production', 
                        'description': 'Quantity of product B to produce'},
        'production_C': {'type': 'float', 'default': 10.0, 'unit': 'units',
                        'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_production',
                        'description': 'Quantity of product C to produce'},
        
        # Problem parameters
        'profit_A': {'type': 'float', 'default': 40.0, 'unit': '$/unit',
                     'description': 'Profit per unit of product A'},
        'profit_B': {'type': 'float', 'default': 50.0, 'unit': '$/unit', 
                     'description': 'Profit per unit of product B'},
        'profit_C': {'type': 'float', 'default': 30.0, 'unit': '$/unit',
                     'description': 'Profit per unit of product C'},
        
        # Machine time requirements (hours per unit)
        'time_A_M1': {'type': 'float', 'default': 2.0, 'unit': 'hours/unit',
                      'description': 'Time required on machine M1 for product A'},
        'time_B_M1': {'type': 'float', 'default': 3.0, 'unit': 'hours/unit',
                      'description': 'Time required on machine M1 for product B'},
        'time_C_M1': {'type': 'float', 'default': 1.0, 'unit': 'hours/unit',
                      'description': 'Time required on machine M1 for product C'},
        'time_A_M2': {'type': 'float', 'default': 1.0, 'unit': 'hours/unit',
                      'description': 'Time required on machine M2 for product A'},
        'time_B_M2': {'type': 'float', 'default': 2.0, 'unit': 'hours/unit',
                      'description': 'Time required on machine M2 for product B'},
        'time_C_M2': {'type': 'float', 'default': 2.0, 'unit': 'hours/unit',
                      'description': 'Time required on machine M2 for product C'},
        
        # Capacity constraints
        'capacity_M1': {'type': 'float', 'default': 100.0, 'unit': 'hours/week',
                        'description': 'Available time on machine M1 per week'},
        'capacity_M2': {'type': 'float', 'default': 80.0, 'unit': 'hours/week',
                        'description': 'Available time on machine M2 per week'},
        'storage_capacity': {'type': 'float', 'default': 50.0, 'unit': 'units',
                             'description': 'Maximum storage capacity'},
        
        # Minimum demand requirements
        'min_demand_A': {'type': 'float', 'default': 5.0, 'unit': 'units',
                         'description': 'Minimum demand for product A'},
        'min_demand_B': {'type': 'float', 'default': 10.0, 'unit': 'units',
                         'description': 'Minimum demand for product B'},
        'min_demand_C': {'type': 'float', 'default': 8.0, 'unit': 'units',
                         'description': 'Minimum demand for product C'},
    }

    DESC_OUT = {
        # Objective function (to maximize)
        'total_profit': {'type': 'float', 'unit': '$', 
                         'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_production',
                         'description': 'Total profit from production plan'},
        
        # Constraint functions (must be <= 0 for GEMSEO)
        'machine_M1_constraint': {'type': 'float', 'unit': 'hours',
                                  'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_production',
                                  'description': 'Machine M1 time constraint (usage - capacity)'},
        'machine_M2_constraint': {'type': 'float', 'unit': 'hours',
                                  'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_production', 
                                  'description': 'Machine M2 time constraint (usage - capacity)'},
        'storage_constraint': {'type': 'float', 'unit': 'units',
                              'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_production',
                              'description': 'Storage capacity constraint (usage - capacity)'},
        'demand_A_constraint': {'type': 'float', 'unit': 'units',
                               'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_production',
                               'description': 'Minimum demand constraint for product A (min_demand - production)'},
        'demand_B_constraint': {'type': 'float', 'unit': 'units',
                               'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_production',
                               'description': 'Minimum demand constraint for product B (min_demand - production)'},
        'demand_C_constraint': {'type': 'float', 'unit': 'units',
                               'visibility': SoSWrapp.SHARED_VISIBILITY, 'namespace': 'ns_production',
                               'description': 'Minimum demand constraint for product C (min_demand - production)'},
        
        # Additional monitoring outputs
        'machine_M1_utilization': {'type': 'float', 'unit': 'hours',
                                   'description': 'Total time used on machine M1'},
        'machine_M2_utilization': {'type': 'float', 'unit': 'hours',
                                   'description': 'Total time used on machine M2'},
        'storage_utilization': {'type': 'float', 'unit': 'units',
                               'description': 'Total storage space used'},
    }

    def run(self):
        """
        Execute the production planning problem.
        
        Computes the objective function and constraint values based on current
        production quantities and problem parameters.
        """
        # Get input values - using production_A, production_B, production_C names
        inputs = self.get_sosdisc_inputs([
            'production_A', 'production_B', 'production_C',
            'profit_A', 'profit_B', 'profit_C',
            'time_A_M1', 'time_B_M1', 'time_C_M1',
            'time_A_M2', 'time_B_M2', 'time_C_M2', 
            'capacity_M1', 'capacity_M2', 'storage_capacity',
            'min_demand_A', 'min_demand_B', 'min_demand_C'
        ])
        
        x_A, x_B, x_C = inputs['production_A'], inputs['production_B'], inputs['production_C']
        
        # Objective function: Total profit (to be maximized)
        total_profit = (inputs['profit_A'] * x_A + 
                       inputs['profit_B'] * x_B + 
                       inputs['profit_C'] * x_C)
        
        # Machine utilization calculations
        machine_M1_utilization = (inputs['time_A_M1'] * x_A + 
                                  inputs['time_B_M1'] * x_B + 
                                  inputs['time_C_M1'] * x_C)
        
        machine_M2_utilization = (inputs['time_A_M2'] * x_A + 
                                  inputs['time_B_M2'] * x_B + 
                                  inputs['time_C_M2'] * x_C)
        
        storage_utilization = x_A + x_B + x_C
        
        # Constraint functions (formulated as g(x) <= 0 for GEMSEO)
        # Machine capacity constraints: usage - capacity <= 0
        machine_M1_constraint = machine_M1_utilization - inputs['capacity_M1']
        machine_M2_constraint = machine_M2_utilization - inputs['capacity_M2']
        
        # Storage constraint: total production - capacity <= 0
        storage_constraint = storage_utilization - inputs['storage_capacity']
        
        # Demand constraints: min_demand - production <= 0 (i.e., production >= min_demand)
        demand_A_constraint = inputs['min_demand_A'] - x_A
        demand_B_constraint = inputs['min_demand_B'] - x_B
        demand_C_constraint = inputs['min_demand_C'] - x_C
        
        # Store outputs
        outputs = {
            'total_profit': total_profit,
            'machine_M1_constraint': machine_M1_constraint,
            'machine_M2_constraint': machine_M2_constraint,
            'storage_constraint': storage_constraint,
            'demand_A_constraint': demand_A_constraint,
            'demand_B_constraint': demand_B_constraint,
            'demand_C_constraint': demand_C_constraint,
            'machine_M1_utilization': machine_M1_utilization,
            'machine_M2_utilization': machine_M2_utilization,
            'storage_utilization': storage_utilization,
        }
        
        self.store_sos_outputs_values(outputs)

    def compute_sos_jacobian(self):
        """
        Compute analytical Jacobian for the production planning problem.
        
        Returns derivatives of outputs with respect to inputs for GEMSEO optimization.
        """
        # Get input values for Jacobian computation
        inputs = self.get_sosdisc_inputs([
            'production_A', 'production_B', 'production_C',
            'profit_A', 'profit_B', 'profit_C',
            'time_A_M1', 'time_B_M1', 'time_C_M1',
            'time_A_M2', 'time_B_M2', 'time_C_M2', 
            'capacity_M1', 'capacity_M2', 'storage_capacity',
            'min_demand_A', 'min_demand_B', 'min_demand_C'
        ])
        
        # Define input and output variable lists
        input_vars = ['production_A', 'production_B', 'production_C']
        output_vars = ['total_profit', 'machine_M1_constraint', 'machine_M2_constraint',
                      'storage_constraint', 'demand_A_constraint', 'demand_B_constraint',
                      'demand_C_constraint', 'machine_M1_utilization', 'machine_M2_utilization',
                      'storage_utilization']
        
        # Initialize Jacobian dictionary
        jacobian_dict = {}
        
        # Jacobian computation for each output with respect to production variables
        for output_var in output_vars:
            jacobian_dict[output_var] = {}
            
            if output_var == 'total_profit':
                # d(total_profit)/d(production_A) = profit_A
                jacobian_dict[output_var]['production_A'] = np.array([[inputs['profit_A']]])
                jacobian_dict[output_var]['production_B'] = np.array([[inputs['profit_B']]])
                jacobian_dict[output_var]['production_C'] = np.array([[inputs['profit_C']]])
                
            elif output_var == 'machine_M1_constraint':
                # d(machine_M1_constraint)/d(production_A) = time_A_M1
                jacobian_dict[output_var]['production_A'] = np.array([[inputs['time_A_M1']]])
                jacobian_dict[output_var]['production_B'] = np.array([[inputs['time_B_M1']]])
                jacobian_dict[output_var]['production_C'] = np.array([[inputs['time_C_M1']]])
                
            elif output_var == 'machine_M2_constraint':
                # d(machine_M2_constraint)/d(production_A) = time_A_M2
                jacobian_dict[output_var]['production_A'] = np.array([[inputs['time_A_M2']]])
                jacobian_dict[output_var]['production_B'] = np.array([[inputs['time_B_M2']]])
                jacobian_dict[output_var]['production_C'] = np.array([[inputs['time_C_M2']]])
                
            elif output_var == 'storage_constraint':
                # d(storage_constraint)/d(production_X) = 1
                jacobian_dict[output_var]['production_A'] = np.array([[1.0]])
                jacobian_dict[output_var]['production_B'] = np.array([[1.0]])
                jacobian_dict[output_var]['production_C'] = np.array([[1.0]])
                
            elif output_var == 'demand_A_constraint':
                # d(demand_A_constraint)/d(production_A) = -1
                jacobian_dict[output_var]['production_A'] = np.array([[-1.0]])
                jacobian_dict[output_var]['production_B'] = np.array([[0.0]])
                jacobian_dict[output_var]['production_C'] = np.array([[0.0]])
                
            elif output_var == 'demand_B_constraint':
                # d(demand_B_constraint)/d(production_B) = -1
                jacobian_dict[output_var]['production_A'] = np.array([[0.0]])
                jacobian_dict[output_var]['production_B'] = np.array([[-1.0]])
                jacobian_dict[output_var]['production_C'] = np.array([[0.0]])
                
            elif output_var == 'demand_C_constraint':
                # d(demand_C_constraint)/d(production_C) = -1
                jacobian_dict[output_var]['production_A'] = np.array([[0.0]])
                jacobian_dict[output_var]['production_B'] = np.array([[0.0]])
                jacobian_dict[output_var]['production_C'] = np.array([[-1.0]])
                
            elif output_var == 'machine_M1_utilization':
                # Same as machine_M1_constraint
                jacobian_dict[output_var]['production_A'] = np.array([[inputs['time_A_M1']]])
                jacobian_dict[output_var]['production_B'] = np.array([[inputs['time_B_M1']]])
                jacobian_dict[output_var]['production_C'] = np.array([[inputs['time_C_M1']]])
                
            elif output_var == 'machine_M2_utilization':
                # Same as machine_M2_constraint
                jacobian_dict[output_var]['production_A'] = np.array([[inputs['time_A_M2']]])
                jacobian_dict[output_var]['production_B'] = np.array([[inputs['time_B_M2']]])
                jacobian_dict[output_var]['production_C'] = np.array([[inputs['time_C_M2']]])
                
            elif output_var == 'storage_utilization':
                # Same as storage_constraint
                jacobian_dict[output_var]['production_A'] = np.array([[1.0]])
                jacobian_dict[output_var]['production_B'] = np.array([[1.0]])
                jacobian_dict[output_var]['production_C'] = np.array([[1.0]])
        
        self.set_partial_derivatives_for_other_types(jacobian_dict)
