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

    def __init__(self, execution_engine=None):
        super().__init__(__file__, execution_engine=execution_engine)

    def setup_usecase(self):
        ns = f'{self.study_name}'
        sc_name = "SellarOptimScenario"
        dspace_dict = {'variable': ['x', 'z', 'y_1', 'y_2'],
                       'value': [[1.], [5., 2.], [1.], [1.]],
                       'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                       'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}
#                   'type' : ['float',['float','float'],'float','float']
        dspace = pd.DataFrame(dspace_dict)

        disc_dict = {}
        # Optim inputs
        disc_dict[f'{ns}.SellarOptimScenario.max_iter'] = 100
        # SLSQP, NLOPT_SLSQP
        disc_dict[f'{ns}.SellarOptimScenario.algo'] = "SLSQP"
        disc_dict[f'{ns}.SellarOptimScenario.design_space'] = dspace
        # TODO: what's wrong with IDF
        disc_dict[f'{ns}.SellarOptimScenario.formulation'] = 'MDF'
        # f'{ns}.SellarOptimScenario.obj'
        disc_dict[f'{ns}.SellarOptimScenario.objective_name'] = 'obj'
        disc_dict[f'{ns}.SellarOptimScenario.ineq_constraints'] = [
            'c_1', 'c_2']
        # f'{ns}.SellarOptimScenario.c_1', f'{ns}.SellarOptimScenario.c_2']

        disc_dict[f'{ns}.SellarOptimScenario.algo_options'] = {"ftol_rel": 1e-10,
                                                               "ineq_tolerance": 2e-3,
                                                               "normalize_design_space": False}
        disc_dict[f'{ns}.SellarOptimScenario.differentiation_method'] = 'user'

        # Sellar inputs
        disc_dict[f'{ns}.{sc_name}.x'] = 1.
#         disc_dict[f'{ns}.{sc_name}.y_1'] = array([1.])
#         disc_dict[f'{ns}.{sc_name}.y_2'] = array([1.])
        disc_dict[f'{ns}.{sc_name}.z'] = array([1., 1.])
        disc_dict[f'{ns}.{sc_name}.Sellar_Problem.local_dv'] = 10.

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
