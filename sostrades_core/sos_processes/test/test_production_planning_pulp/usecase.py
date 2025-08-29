#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Planning Optimization UseCase with PuLP

This usecase demonstrates a realistic production planning optimization problem
using PuLP (Python Linear Programming) as the optimization algorithm through GEMSEO.

The problem is based on classic Operations Research formulations commonly used
in manufacturing and supply chain management.

Problem Description:
A manufacturing company produces three products (A, B, C) using two machines (M1, M2).
The goal is to maximize profit while respecting:
- Machine capacity constraints
- Storage limitations  
- Minimum demand requirements

This is a reference problem in Operations Research literature.
"""

import pandas as pd
from sostrades_core.study_manager.study_manager import StudyManager


class ProductionPlanningPuLPStudy(StudyManager):
    """
    Production Planning Optimization Study with PuLP
    
    Based on the multi-product manufacturing problem commonly found
    in Operations Research textbooks and industrial applications.
    """

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        """
        Setup the production planning optimization usecase.
        
        Returns:
            list: Configuration dictionary for the study
        """
        
        # Define design space for production quantities (decision variables)
        design_space = pd.DataFrame({
            'variable': ['production_A', 'production_B', 'production_C'],
            'value': [[20.0], [15.0], [10.0]],           # Starting values  
            'lower_bnd': [[0.0], [0.0], [0.0]],          # Cannot produce negative quantities
            'upper_bnd': [[50.0], [50.0], [50.0]],       # Production capacity limits
            'enable_variable': [True, True, True],
            'activated_elem': [[True], [True], [True]]
        })
        
        # Configuration dictionary with proper namespace structure
        values_dict = {
            # Optimization algorithm configuration
            f'{self.study_name}.ProductionPlanningOptimization.algo': 'PULP_CBC_CMD',
            f'{self.study_name}.ProductionPlanningOptimization.design_space': design_space,
            f'{self.study_name}.ProductionPlanningOptimization.formulation': 'DisciplinaryOpt',
            f'{self.study_name}.ProductionPlanningOptimization.objective_name': 'total_profit',
            f'{self.study_name}.ProductionPlanningOptimization.maximize_objective': True,
            f'{self.study_name}.ProductionPlanningOptimization.max_iter': 1000,
            f'{self.study_name}.ProductionPlanningOptimization.algo_options': {},
            
            # Inequality constraints (all must be <= 0 for GEMSEO)
            f'{self.study_name}.ProductionPlanningOptimization.ineq_constraints': [
                'machine_M1_constraint',    # Machine M1 time constraint
                'machine_M2_constraint',    # Machine M2 time constraint  
                'storage_constraint',       # Storage capacity constraint
                'demand_A_constraint',      # Minimum demand for product A
                'demand_B_constraint',      # Minimum demand for product B
                'demand_C_constraint'       # Minimum demand for product C
            ],
            
            # Production planning problem parameters
            f'{self.study_name}.ProductionPlanningCoupling.profit_A': 40.0,      # $/unit profit for A
            f'{self.study_name}.ProductionPlanningCoupling.profit_B': 50.0,      # $/unit profit for B  
            f'{self.study_name}.ProductionPlanningCoupling.profit_C': 30.0,      # $/unit profit for C
            
            # Machine M1 time requirements (hours per unit)
            f'{self.study_name}.ProductionPlanningCoupling.time_A_M1': 2.0,      # Product A on M1
            f'{self.study_name}.ProductionPlanningCoupling.time_B_M1': 3.0,      # Product B on M1
            f'{self.study_name}.ProductionPlanningCoupling.time_C_M1': 1.0,      # Product C on M1
            
            # Machine M2 time requirements (hours per unit)  
            f'{self.study_name}.ProductionPlanningCoupling.time_A_M2': 1.0,      # Product A on M2
            f'{self.study_name}.ProductionPlanningCoupling.time_B_M2': 2.0,      # Product B on M2
            f'{self.study_name}.ProductionPlanningCoupling.time_C_M2': 2.0,      # Product C on M2
            
            # Resource capacity constraints
            f'{self.study_name}.ProductionPlanningCoupling.capacity_M1': 100.0,  # Hours/week available on M1
            f'{self.study_name}.ProductionPlanningCoupling.capacity_M2': 80.0,   # Hours/week available on M2
            f'{self.study_name}.ProductionPlanningCoupling.storage_capacity': 50.0,  # Maximum storage units
            
            # Minimum demand requirements  
            f'{self.study_name}.ProductionPlanningCoupling.min_demand_A': 5.0,   # Minimum units of A
            f'{self.study_name}.ProductionPlanningCoupling.min_demand_B': 10.0,  # Minimum units of B
            f'{self.study_name}.ProductionPlanningCoupling.min_demand_C': 8.0    # Minimum units of C
        }
        
        return [values_dict]


if __name__ == '__main__':
    """
    Execute the production planning optimization usecase.
    
    This will solve a realistic manufacturing optimization problem using PuLP
    linear programming solver through the GEMSEO framework in SOStrades.
    """
    
    print("=" * 80)
    print("PRODUCTION PLANNING OPTIMIZATION WITH PuLP")
    print("=" * 80)
    print()
    print("Problem: Multi-product manufacturing optimization")
    print("Products: A, B, C with different profit margins")  
    print("Resources: 2 machines (M1, M2) with limited capacity")
    print("Objective: Maximize total profit")
    print("Constraints: Machine time, storage capacity, minimum demand")
    print("Algorithm: PuLP CBC (Coin-or Branch and Cut) solver")
    print()
    print("Running optimization...")
    print()
    
    # Create and run the study
    uc_cls = ProductionPlanningPuLPStudy(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
    
    print()
    print("=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    
    # Display optimization results
    dm = uc_cls.execution_engine.dm
    
    # Get optimal production quantities
    production_A = dm.get_value('usecase.ProductionPlanningOptimization.production_A')
    production_B = dm.get_value('usecase.ProductionPlanningOptimization.production_B') 
    production_C = dm.get_value('usecase.ProductionPlanningOptimization.production_C')
    
    print(f"Optimal Production Plan:")
    print(f"  Product A: {production_A:.2f} units")
    print(f"  Product B: {production_B:.2f} units") 
    print(f"  Product C: {production_C:.2f} units")
    print()
    
    # Get objective value (total profit)
    total_profit = dm.get_value('usecase.ProductionPlanningCoupling.total_profit')
    print(f"Maximum Profit: ${total_profit:.2f}")
    print()
    
    # Get resource utilization
    machine_M1_util = dm.get_value('usecase.ProductionPlanningCoupling.machine_M1_utilization')
    machine_M2_util = dm.get_value('usecase.ProductionPlanningCoupling.machine_M2_utilization')
    storage_util = dm.get_value('usecase.ProductionPlanningCoupling.storage_utilization')
    
    print(f"Resource Utilization:")
    print(f"  Machine M1: {machine_M1_util:.2f} / 100.0 hours ({machine_M1_util:.1f}%)")
    print(f"  Machine M2: {machine_M2_util:.2f} / 80.0 hours ({machine_M2_util/0.8:.1f}%)")
    print(f"  Storage: {storage_util:.2f} / 50.0 units ({storage_util*2:.1f}%)")
    print()
    
    # Check constraint satisfaction
    print("Constraint Verification:")
    
    constraints = ['machine_M1_constraint', 'machine_M2_constraint', 'storage_constraint',
                  'demand_A_constraint', 'demand_B_constraint', 'demand_C_constraint']
    
    for constraint in constraints:
        value = dm.get_value(f'usecase.ProductionPlanningCoupling.{constraint}')
        status = "✓ Satisfied" if value <= 0.001 else "✗ Violated"
        print(f"  {constraint}: {value:.3f} {status}")
    
    print()
    print("Production planning optimization completed successfully!")
    print("=" * 80)
