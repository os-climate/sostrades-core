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
"""
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
unit test for optimization scenario
"""

import unittest
from numpy import array, set_printoptions
import pandas as pd
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sos_trades_core.sos_processes.test.test_sellar_opt.usecase import Study as study_sellar_opt
from sos_trades_core.sos_processes.test.test_Griewank_opt.usecase import Study as study_griewank
from sos_trades_core.sos_processes.test.test_sellar_opt_idf.usecase import Study as study_sellar_idf
import os
from gemseo.core.mdo_scenario import MDOScenario
from copy import deepcopy


class TestSoSOptimScenario(unittest.TestCase):
    """
    SoSOptimScenario test class
    """

    def setUp(self):
        self.study_name = 'optim'




        self.repo = 'sos_trades_core.sos_processes.test'
        self.proc_name = 'test_sellar_opt_ms'

    def test_01_ms_sellar_cleaning(self):
        print("\n Test 1 : check configure and treeview")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory

        opt_builder = factory.get_builder_from_process(repo=self.repo,
                                                       mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)
        dspace_dict = {'variable': ['x', 'z', 'y_1', 'y_2'],
                       'value': [[1.], [5., 2.], [1.], [1.]],
                       'lower_bnd': [[0.], [-10., 0.], [-100.], [-100.]],
                       'upper_bnd': [[10.], [10., 10.], [100.], [100.]],
                       'enable_variable': [True, True, True, True],
                       'activated_elem': [[True], [True, True], [True], [True]]}
        #                   'type' : ['float',['float','float'],'float','float']
        dspace = pd.DataFrame(dspace_dict)
        exec_eng.configure()
        sc_name = "SellarOptimScenario"
        ns = f'{self.study_name}'
        #-- set up disciplines in Scenario
        disc_dict = {}
        # Optim inputs

        scatter_scenario_name = 'optimization scenarios'
        scenario_list = ['a=0-1', 'a=0-2']
        disc_dict[f'{ns}.{scatter_scenario_name}.scenario_list'] = ['a=0-1', 'a=0-2']
        exec_eng.dm.set_values_from_dict(disc_dict)
        exec_eng.configure()
        for scen in scenario_list:
            disc_dict[f'{ns}.{scatter_scenario_name}.{scen}.{sc_name}.Sellar_Problem.local_dv'] = 5.
            disc_dict[f'{ns}.{scatter_scenario_name}.{scen}.{sc_name}.z'] = array([2., 1.])
            disc_dict[f'{ns}.{scatter_scenario_name}.{scen}.{sc_name}.max_iter'] = 200
            disc_dict[f'{ns}.{scatter_scenario_name}.{scen}.{sc_name}.x'] = 500.
            # SLSQP, NLOPT_SLSQP
            disc_dict[f'{ns}.{scatter_scenario_name}.{scen}.{sc_name}.algo'] = "SLSQP"
            disc_dict[f'{ns}.{scatter_scenario_name}.{scen}.{sc_name}.design_space'] = dspace
            # TODO: what's wrong with IDF
            disc_dict[f'{ns}.{scatter_scenario_name}.{scen}.{sc_name}.formulation'] = 'MDF'
            # f'{ns}.SellarOptimScenario.obj'
            disc_dict[f'{ns}.{scatter_scenario_name}.{scen}.{sc_name}.objective_name'] = 'obj'
            disc_dict[f'{ns}.{scatter_scenario_name}.{scen}.{sc_name}.ineq_constraints'] = [
                'c_1', 'c_2']

            disc_dict[f'{ns}.{scatter_scenario_name}.{scen}.{sc_name}.algo_options'] = {"ftol_rel": 1e-10,
                                                                                        "ineq_tolerance": 2e-3,
                                                                                        "normalize_design_space": False}

        exec_eng.load_study_from_input_dict(disc_dict)


        exec_eng.configure()


        print(exec_eng.display_treeview_nodes())
        len_before_clean = len(list(exec_eng.dm.disciplines_id_map.keys()))
        #delete scenario and configure
        disc_dict[f'{ns}.{scatter_scenario_name}.scenario_list'] = ['a=0-1']
        exec_eng.load_study_from_input_dict(disc_dict)
        print(exec_eng.display_treeview_nodes(display_variables=True))

        # assert there is no variables of second scenario in dm (so that cleaning went well) and number of disciplines after cleaning is correct
        scen_name = 'a=0-2'
        list_var_scen = [f'{ns}.{scatter_scenario_name}.{scen_name}.{sc_name}.Sellar_Problem.local_dv', f'{ns}.{scatter_scenario_name}.{scen_name}.{sc_name}.z',
                         f'{ns}.{scatter_scenario_name}.{scen_name}.{sc_name}.x', f'{ns}.{scatter_scenario_name}.{scen_name}.{sc_name}.y_2', f'{ns}.{scatter_scenario_name}.{scen_name}.{sc_name}.y_1']
        for var in list_var_scen:
            # assert variable is not in dm
            assert(var not in exec_eng.dm.convert_data_dict_with_full_name().keys())

        len_after_clean = len(list(exec_eng.dm.disciplines_id_map.keys()))

        # assert disciplines are not in dm
        assert(len_after_clean == 8)

if '__main__' == __name__:
    cls = TestSoSOptimScenario()
    cls.setUp()
    cls.test_01_ms_sellar_cleaning()
