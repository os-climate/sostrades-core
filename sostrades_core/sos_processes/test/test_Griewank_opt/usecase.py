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
        sc_name = "GriewankOptimScenario"
        dspace_dict = {'variable': ['chromosome'],
                       'value': [[3., -0.2]],
                       'lower_bnd': [[-5., -0.5]],
                       'upper_bnd': [[5., 5]],
                       'enable_variable': [True],
                       'activated_elem': [[True, True]]}
#                   'type' : ['float',['float','float'],'float','float']
        dspace = pd.DataFrame(dspace_dict)

        disc_dict = {}
        # Optim inputs
        disc_dict[f'{ns}.GriewankOptimScenario.max_iter'] = 100000
        # SLSQP, NLOPT_SLSQP, CMAES
        disc_dict[f'{ns}.GriewankOptimScenario.algo'] = "SLSQP"
        disc_dict[f'{ns}.GriewankOptimScenario.design_space'] = dspace
        # TODO: what's wrong with IDF
        disc_dict[f'{ns}.GriewankOptimScenario.formulation'] = 'MDF'
        # f'{ns}.SellarOptimScenario.obj'
        disc_dict[f'{ns}.GriewankOptimScenario.objective_name'] = 'obj'
        # f'{ns}.SellarOptimScenario.c_1', f'{ns}.SellarOptimScenario.c_2']
        disc_dict[f'{ns}.GriewankOptimScenario.ineq_constraints'] = []
        disc_dict[f'{ns}.GriewankOptimScenario.algo_options'] = {"ftol_rel": 1e-15,
                                                                 "population_size": 900,
                                                                 "sigma": 0.1}

        # Sellar inputs
#         disc_dict[f'{ns}.{sc_name}.y_1'] = array([1.])
#         disc_dict[f'{ns}.{sc_name}.y_2'] = array([1.])
        disc_dict[f'{ns}.{sc_name}.chromosome'] = array([-3., 0.2])

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
