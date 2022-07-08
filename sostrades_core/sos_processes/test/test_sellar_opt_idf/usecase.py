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


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        ns = f'{self.study_name}'
        sc_name = "SellarOptimScenario.SellarCoupling"
        dspace_dict = {'variable': ['x', 'z', 'y_01', 'y_02'],
                       'value': [[1.], [5., 2.], [1.], [1.]],
                       'lower_bnd': [[0.], [-10., 0.], [-10.], [-10.]],
                       'upper_bnd': [[10.], [10., 10.], [10.], [10.]],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}
#                   'type' : ['float',['float','float'],'float','float']
        dspace = pd.DataFrame(dspace_dict)

        disc_dict = {}
        # Optim inputs
        disc_dict[f'{ns}.SellarOptimScenario.max_iter'] = 100
        # SLSQP, NLOPT_SLSQP
        disc_dict[f'{ns}.SellarOptimScenario.algo'] = "NLOPT_SLSQP"
        disc_dict[f'{ns}.SellarOptimScenario.design_space'] = dspace
        # TODO: what's wrong with IDF
        disc_dict[f'{ns}.SellarOptimScenario.formulation'] = 'DisciplinaryOpt'
        # f'{ns}.SellarOptimScenario.obj'
        disc_dict[f'{ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{ns}.SellarOptimScenario.ineq_constraints'] = ['c_1', 'c_2'
                                                                   ]
        # f'{ns}.SellarOptimScenario.c_1', f'{ns}.SellarOptimScenario.c_2']
        disc_dict[f'{ns}.SellarOptimScenario.eq_constraints'] = ['c_3', 'c_4'
                                                                 ]

        disc_dict[f'{ns}.SellarOptimScenario.algo_options'] = {
            'eq_tolerance': 1e-1, 'ineq_tolerance': 1e-6, 'ftol_abs': 1e-6,
            'xtol_abs': 1e-6, 'ctol_abs': 1e-1}

        # Sellar inputs
        disc_dict[f'{ns}.{sc_name}.x'] = 1.
#         disc_dict[f'{ns}.{sc_name}.y_1'] = array([1.])
#         disc_dict[f'{ns}.{sc_name}.y_2'] = array([1.])
        disc_dict[f'{ns}.{sc_name}.z'] = array([1., 1.])
        disc_dict[f'{ns}.{sc_name}.Sellar_Problem.local_dv'] = 10.
        disc_dict[f'{ns}.{sc_name}.y_01'] = 1.
        disc_dict[f'{ns}.{sc_name}.y_02'] = 1.

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
