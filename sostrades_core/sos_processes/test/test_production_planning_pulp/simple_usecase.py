"""
Simple test case for production planning problem without optimization
Tests the discipline evaluation directly.
"""

import logging
from sostrades_core.study_manager.base_study_manager import BaseStudyManager


if __name__ == '__main__':
    # Configure logging for better visibility
    logging.basicConfig(level=logging.INFO, 
                       format='%(levelname)s:%(name)s:%(message)s')
    
    print("="*80)
    print("PRODUCTION PLANNING PROBLEM TEST")
    print("="*80)
    print("Testing the production planning discipline without optimization")
    print()
    
    # Define study parameters
    study_name = 'ProductionPlanningTest'
    repository_name = 'sostrades_core.sos_processes.test'
    process_name = 'test_production_planning_pulp'
    
    # Initialize study manager
    print("Initializing study...")
    study = BaseStudyManager(repository_name, process_name, study_name)
    
    # Load process definition
    print("Loading process definition...")
    study.load_data()
    
    # Define input data for production planning problem
    print("Setting up input data...")
    input_data = {
        f'{study_name}.ProductionPlanningOptimization.ProductionPlanningCoupling.ProductionPlanningProblem.x_A': 20.0,
        f'{study_name}.ProductionPlanningOptimization.ProductionPlanningCoupling.ProductionPlanningProblem.x_B': 15.0,
        f'{study_name}.ProductionPlanningOptimization.ProductionPlanningCoupling.ProductionPlanningProblem.x_C': 10.0,
    }
    
    # Load input data
    study.load_data(from_input_dict=input_data)
    
    print("TreeView after loading data:")
    print(study.execution_engine.display_treeview_nodes())
    
    # Execute the discipline
    print("Executing production planning problem...")
    try:
        study.run()
        print("✓ Execution successful!")
        
        # Get results
        print("\nResults:")
        total_profit = study.execution_engine.dm.get_value(
            f'{study_name}.ProductionPlanningOptimization.ProductionPlanningCoupling.ProductionPlanningProblem.total_profit'
        )
        print(f"Total Profit: ${total_profit:,.2f}")
        
        # Get constraint values
        m1_constraint = study.execution_engine.dm.get_value(
            f'{study_name}.ProductionPlanningOptimization.ProductionPlanningCoupling.ProductionPlanningProblem.machine_M1_constraint'
        )
        m2_constraint = study.execution_engine.dm.get_value(
            f'{study_name}.ProductionPlanningOptimization.ProductionPlanningCoupling.ProductionPlanningProblem.machine_M2_constraint'
        )
        storage_constraint = study.execution_engine.dm.get_value(
            f'{study_name}.ProductionPlanningOptimization.ProductionPlanningCoupling.ProductionPlanningProblem.storage_constraint'
        )
        
        print(f"Machine M1 constraint (usage - capacity): {m1_constraint:.2f}")
        print(f"Machine M2 constraint (usage - capacity): {m2_constraint:.2f}")
        print(f"Storage constraint (usage - capacity): {storage_constraint:.2f}")
        
        # Check feasibility
        print("\nFeasibility Check:")
        if m1_constraint <= 0 and m2_constraint <= 0 and storage_constraint <= 0:
            print("✓ Solution is feasible (all constraints satisfied)")
        else:
            print("✗ Solution violates constraints:")
            if m1_constraint > 0:
                print(f"  - Machine M1 over capacity by {m1_constraint:.2f} hours")
            if m2_constraint > 0:
                print(f"  - Machine M2 over capacity by {m2_constraint:.2f} hours")
            if storage_constraint > 0:
                print(f"  - Storage over capacity by {storage_constraint:.2f} units")
        
    except Exception as e:
        print(f"✗ Execution failed: {e}")
        import traceback
        traceback.print_exc()
