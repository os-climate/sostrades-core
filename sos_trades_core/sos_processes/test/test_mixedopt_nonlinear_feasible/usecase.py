'''
Copyright 2022 Airbus SAS

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
from sos_trades_core.study_manager.study_manager import StudyManager
from numpy import array
import pandas as pd
from gemseo.algos.design_space import DesignSpace


class Study(StudyManager):

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        ns = f'{self.study_name}'
        sc_name = "MixedOptimScenario"
        disc_name = "DiscMixedOpt"
        c_name = "MixedCoupling"
        dspace_dict = {'variable': ['x1', 'x2'],
                       'value': [[2], [10.]],
                       'lower_bnd': [[0], [0.]],
                       'upper_bnd': [[200], [200.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True]],
                       'variable_type' : [DesignSpace.INTEGER, DesignSpace.FLOAT]}
        dspace = pd.DataFrame(dspace_dict)

        # Optim inputs
        disc_dict = {}
        disc_dict[f'{ns}.{sc_name}.max_iter'] = 100
        disc_dict[f'{ns}.{sc_name}.algo'] = "OuterApproximation"
        disc_dict[f'{ns}.{sc_name}.design_space'] = dspace
        disc_dict[f'{ns}.{sc_name}.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{ns}.{sc_name}.objective_name'] = 'obj'
        disc_dict[f'{ns}.{sc_name}.ineq_constraints'] = ['constr']
        disc_dict[f'{ns}.{sc_name}.differentiation_method'] = 'user'
        
        algo_options_master = {}
        
        algo_options_slave = {"ftol_rel": 1e-10,
                              "ineq_tolerance": 2e-3,
                              "normalize_design_space": False}
        
        disc_dict[f'{ns}.{sc_name}.algo_options'] = {"ftol_abs": 1e-10,
                                                     "ineq_tolerance": 2e-3,
                                                     "normalize_design_space": False,
                                                     "algo_NLP": "SLSQP",
                                                     "algo_options_NLP": algo_options_slave,
                                                     "algo_options_MILP": algo_options_master}

        # subproc inputs
        disc_dict[f'{ns}.{sc_name}.{c_name}.{disc_name}.x1'] = array([2])
        disc_dict[f'{ns}.{sc_name}.{c_name}.{disc_name}.x2'] = array([4.])

        return [disc_dict]


#     def setup_usecase(self):
#         ns = f'{self.study_name}'
#         sc_name = "MixedOptimScenario"
#         disc_name = "DiscMixedOpt"
#         c_name = "MixedCoupling"
#         dspace_dict = {'variable': ['x', 'y'],
#                        'value': [1., 1.],
#                        'lower_bnd': [0., 1.],
#                        'upper_bnd': [3., 2.],
#                        'enable_variable': [True, True],
#                        'activated_elem': [[True], [True]],
#                        'variable_type' : [DesignSpace.FLOAT, DesignSpace.INTEGER]}
#         dspace = pd.DataFrame(dspace_dict)
# 
#         # Optim inputs
#         disc_dict = {}
#         disc_dict[f'{ns}.{sc_name}.max_iter'] = 100
#         disc_dict[f'{ns}.{sc_name}.algo'] = "OuterApproximation"
#         disc_dict[f'{ns}.{sc_name}.design_space'] = dspace
#         disc_dict[f'{ns}.{sc_name}.formulation'] = 'DisciplinaryOpt'
#         disc_dict[f'{ns}.{sc_name}.objective_name'] = 'obj'
#         disc_dict[f'{ns}.{sc_name}.ineq_constraints'] = ['constr']
#         disc_dict[f'{ns}.{sc_name}.algo_options'] = {"ftol_rel": 1e-10,
#                                                      "ineq_tolerance": 2e-3,
#                                                      "normalize_design_space": False}
# 
#         # subproc inputs
#         disc_dict[f'{ns}.{sc_name}.{c_name}.{disc_name}.x'] = array([1.])
#         disc_dict[f'{ns}.{sc_name}.{c_name}.{disc_name}.y'] = array([1])
# 
#         return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
