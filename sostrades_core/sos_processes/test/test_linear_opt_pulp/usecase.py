'''
Copyright 2024 Capgemini

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

import pandas as pd
from numpy import array
from sostrades_core.study_manager.study_manager import StudyManager

"""
Linear programming optimization usecase with PuLP solver
"""


class Study(StudyManager):

    def __init__(self, run_usecase=True, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.optim_name = "LinearOptimScenario"
        self.coupling_name = "LinearCoupling"

    def setup_usecase(self):
        """Setup usecase for linear programming optimization using PuLP"""
        
        ns = f'{self.study_name}'
        
        # Design space definition
        # Variables: x and y
        # Problem: minimize 2*x + 3*y subject to x + y >= 4, 2*x + y >= 6, x >= 0, y >= 0
        dspace_dict = {
            'variable': ['x', 'y'],
            'value': [array([2]), array([2])],  # Initial values as arrays
            'lower_bnd': [array([0]), array([0])],  # x >= 0, y >= 0 as arrays
            'upper_bnd': [array([10]), array([10])],  # Upper bounds for feasibility as arrays
            'enable_variable': [True, True],
            'activated_elem': [[True], [True]]
        }
        dspace = pd.DataFrame(dspace_dict)

        disc_dict = {}
        
        # Optimization algorithm configuration
        disc_dict[f'{ns}.{self.optim_name}.max_iter'] = 100
        disc_dict[f'{ns}.{self.optim_name}.algo'] = "PULP_CBC_CMD"  # Use PuLP CBC solver
        disc_dict[f'{ns}.{self.optim_name}.design_space'] = dspace
        disc_dict[f'{ns}.{self.optim_name}.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{ns}.{self.optim_name}.objective_name'] = 'objective'
        
        # Define inequality constraints (note: in SOStrades, constraints <= 0)
        # Our constraints are: 4 - x - y <= 0 and 6 - 2*x - y <= 0
        disc_dict[f'{ns}.{self.optim_name}.ineq_constraints'] = ['constraint_1', 'constraint_2']
        
        # Algorithm options for PuLP  
        disc_dict[f'{ns}.{self.optim_name}.algo_options'] = {
            "msg": 1,  # Show solver output
        }

        # Initialize the optimization variables x and y
        disc_dict[f'{ns}.{self.optim_name}.{self.coupling_name}.x'] = 2
        disc_dict[f'{ns}.{self.optim_name}.{self.coupling_name}.y'] = 2

        return [disc_dict]


if __name__ == '__main__':
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()
