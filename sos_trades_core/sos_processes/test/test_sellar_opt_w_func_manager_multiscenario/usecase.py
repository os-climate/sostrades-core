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
# from os.path import join, dirname
import pandas as pd
from numpy import array
# import pandas as pd
# from sos_trades_core.execution_engine.func_manager.func_manager import FunctionManager
# from sos_trades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc
from sos_trades_core.sos_processes.test.test_sellar_opt_w_func_manager._usecase import Study as sellar_usecase
from sos_trades_core.study_manager.study_manager import StudyManager
from sos_trades_core.execution_engine.func_manager.func_manager import FunctionManager
from sos_trades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc

AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX


class Study(StudyManager):

    def __init__(self, run_usecase=False, execution_engine=None):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        # self.coupling_name = "EnergyModelEval"
        self.scenario_name = 'multi_scenarios'

    def setup_usecase(self):

        # energy_uc = energy_usecase()
        # energy_uc.study_name = f'{self.study_name}.{self.coupling_name}'
        # energy_data = energy_uc.setup_usecase()[0]
        # values_dict.update(energy_data)
        INEQ_CONSTRAINT = FunctionManager.INEQ_CONSTRAINT
        OBJECTIVE = FunctionManager.OBJECTIVE
        sc_name = "SellarOptimScenario"
        dspace_dict = {'variable': ['x', 'z', 'y_1', 'y_2'],
                       'value': [[1.], [5., 2.], [5.], [1.]],
                       'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                       'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}
#                   'type' : ['float',['float','float'],'float','float']
        dspace = pd.DataFrame(dspace_dict)

        # disc_dict = {}
        # # Optim inputs
        # disc_dict[f'{ns}.SellarOptimScenario.max_iter'] = 500
        # disc_dict[f'{ns}.SellarOptimScenario.algo'] = "L-BFGS-B"
        # disc_dict[f'{ns}.SellarOptimScenario.design_space'] = dspace
        # # TODO: what's wrong with IDF
        # disc_dict[f'{ns}.SellarOptimScenario.formulation'] = 'MDF'
        # # f'{ns}.SellarOptimScenario.obj'
        # disc_dict[f'{ns}.SellarOptimScenario.objective_name'] = 'objective_lagrangian'
        # disc_dict[f'{ns}.SellarOptimScenario.ineq_constraints'] = [
        # ]
        # # f'{ns}.SellarOptimScenario.c_1', f'{ns}.SellarOptimScenario.c_2']

        # disc_dict[f'{ns}.SellarOptimScenario.algo_options'] = {
        #     #"maxls": 6,
        #     #"maxcor": 3,
        #     "ftol_rel": 1e-15

        #     }

        values_dict = {}

        sellar_uc = sellar_usecase(execution_engine=self.execution_engine)
#        sellar_uc.study_name = f'{self.study_name}.{self.scenario_name}.SellarOptimScenario'
        sellar_uc.study_name = f'{self.study_name}.{self.scenario_name}'
        sellar_data = sellar_uc.setup_usecase()[0]
        values_dict.update(sellar_data)
        config_dict = {
            f'{self.study_name}.multi_scenarios.trade_variables': {'local_dv': 'float'},
            f'{self.study_name}.multi_scenarios.local_dv_trade': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        values_dict.update(config_dict)
        for j in range(1, 11):
            values_dict[f'{self.study_name}.multi_scenarios.scenario_{j}.SellarOptimScenario.x'] = 1.
            # values_dict[f'{self.study_name}.multi_scenarios.scenario_{j}.SellarOptimScenario.y_1'] = array([1.])
            # values_dict[f'{self.study_name}.multi_scenarios.scenario_{j}.SellarOptimScenario.y_2'] = array([1.])
            values_dict[f'{self.study_name}.multi_scenarios.scenario_{j}.SellarOptimScenario.z'] = array([
                                                                                                         1., 1.])
            values_dict[f'{self.study_name}.multi_scenarios.scenario_{j}.SellarOptimScenario.max_iter'] = 500
            values_dict[f'{self.study_name}.multi_scenarios.scenario_{j}.SellarOptimScenario.algo'] = "L-BFGS-B"
            values_dict[f'{self.study_name}.multi_scenarios.scenario_{j}.SellarOptimScenario.design_space'] = dspace
            values_dict[f'{self.study_name}.multi_scenarios.scenario_{j}.SellarOptimScenario.formulation'] = 'MDF'
            values_dict[f'{self.study_name}.multi_scenarios.scenario_{j}.SellarOptimScenario.objective_name'] = 'objective_lagrangian'
            values_dict[f'{self.study_name}.multi_scenarios.scenario_{j}.SellarOptimScenario.ineq_constraints'] = []
            values_dict[f'{self.study_name}.multi_scenarios.scenario_{j}.SellarOptimScenario.algo_options'] = {
                #     #"maxls": 6,
                #     #"maxcor": 3,
                "ftol_rel": 1e-15
            }

            func_df = pd.DataFrame(
                columns=['variable', 'ftype', 'weight', AGGR_TYPE])
            func_df['variable'] = ['c_1', 'c_2', 'obj']
            func_df['ftype'] = [INEQ_CONSTRAINT, INEQ_CONSTRAINT, OBJECTIVE]
            func_df['weight'] = [200, 0.000001, 0.1]
            func_df[AGGR_TYPE] = [AGGR_TYPE_SUM, AGGR_TYPE_SUM, AGGR_TYPE_SUM]
            values_dict[f'{self.study_name}.multi_scenarios.scenario_{j}.SellarOptimScenario.FunctionManager.function_df'] = func_df

        return [values_dict]


if '__main__' == __name__:
    uc_cls = Study()
    uc_cls.load_data()
    uc_cls.execution_engine.display_treeview_nodes(display_variables=True)
    uc_cls.run()
