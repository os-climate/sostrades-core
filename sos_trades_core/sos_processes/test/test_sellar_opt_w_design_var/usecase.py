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
from numpy import array, arange
import pandas as pd
from sos_trades_core.execution_engine.func_manager.func_manager import FunctionManager
from sos_trades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc

AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX


class Study(StudyManager):

    def __init__(self, run_usecase=True, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.optim_name = "SellarOptimScenario"
        self.coupling_name = "SellarCoupling"

    def setup_usecase(self):

        INEQ_CONSTRAINT = FunctionManager.INEQ_CONSTRAINT
        OBJECTIVE = FunctionManager.OBJECTIVE

        ns = f'{self.study_name}'
        dspace_dict = {'variable': ['x_in', 'z_in'],
                       'value': [[1., 2., 3., 4.], [5., 2.]],
                       'lower_bnd': [[0., 0., 0., 0.], [-10., 0.]],
                       'upper_bnd': [[10., 10., 10., 10.], [10., 10.]],
                       'enable_variable': [True, True],
                       'activated_elem': [[True], [True, True]]}
        dspace = pd.DataFrame(dspace_dict)

        output_descriptor = {'x_in': {'out_name': 'x',
                                      'type': 'array',
                                      'out_type': 'dataframe',
                                      'key': 'value',
                                      'index': arange(0, 4, 1),
                                      'index_name': 'index',
                                      'namespace_in': 'ns_OptimSellar',
                                      'namespace_out': 'ns_OptimSellar'
                                      },
                             'z_in': {'out_name': 'z',
                                      'type': 'array',
                                      'out_type': 'array',
                                      'index': [0, 1],
                                      'index_name': 'index',
                                      'namespace_in': 'ns_OptimSellar',
                                      'namespace_out': 'ns_OptimSellar'
                                      }
                             }

        disc_dict = {}
        disc_dict[f'{ns}.{self.optim_name}.{self.coupling_name}.DesignVar.output_descriptor'] = output_descriptor

        # Optim inputs
        disc_dict[f'{ns}.{self.optim_name}.max_iter'] = 100
        disc_dict[f'{ns}.{self.optim_name}.algo'] = "L-BFGS-B"
        disc_dict[f'{ns}.{self.optim_name}.design_space'] = dspace
        disc_dict[f'{ns}.{self.optim_name}.formulation'] = 'DisciplinaryOpt'
        disc_dict[f'{ns}.{self.optim_name}.objective_name'] = 'objective_lagrangian'
        disc_dict[f'{ns}.{self.optim_name}.ineq_constraints'] = []
        disc_dict[f'{ns}.{self.optim_name}.algo_options'] = {
            #"maxls": 6,
            #"maxcor": 3,
            "ftol_rel": 1e-15,

        }

        # Sellar and design var inputs
        disc_dict[f'{ns}.{self.optim_name}.x_in'] = [1., 1., 1., 1.]
        disc_dict[f'{ns}.{self.optim_name}.y_1'] = 5.
        disc_dict[f'{ns}.{self.optim_name}.y_2'] = 1.
        disc_dict[f'{ns}.{self.optim_name}.z_in'] = array([5., 2.])
        disc_dict[f'{ns}.{self.optim_name}.{self.coupling_name}.Sellar_Problem.local_dv'] = 10.

        func_df = pd.DataFrame(
            columns=['variable', 'ftype', 'weight', AGGR_TYPE])
        func_df['variable'] = ['c_1', 'c_2', 'obj']
        func_df['ftype'] = [INEQ_CONSTRAINT, INEQ_CONSTRAINT, OBJECTIVE]
        func_df['weight'] = [200, 0.000001, 0.1]
        func_df[AGGR_TYPE] = [AGGR_TYPE_SUM, AGGR_TYPE_SUM, AGGR_TYPE_SUM]
        func_mng_name = 'FunctionManager'

        prefix = f'{self.study_name}.{self.optim_name}.{self.coupling_name}.{func_mng_name}.'
        values_dict = {}
        values_dict[prefix +
                    FunctionManagerDisc.FUNC_DF] = func_df

        disc_dict.update(values_dict)

        return [disc_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
