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
from sos_trades_core.sos_processes.test.test_sellar_opt_ms.usecase import Study as study_sellar_opt
import os
from gemseo.core.mdo_scenario import MDOScenario
from copy import deepcopy


class TestSoSOptimScenario(unittest.TestCase):
    """
    SoSOptimScenario test class
    """

    def setUp(self):
        self.study_name = 'optim'

        self.repo = 'sostrades_core.sos_processes.test'
        self.proc_name = 'test_sellar_opt_ms'

    def test_01_ms_sellar_cleaning(self):
        print("\n Test 1 : check configure and treeview")
        exec_eng = ExecutionEngine(self.study_name)
        factory = exec_eng.factory
        sc_name = 'SellarOptimScenario'
        scatter_scenario_name = 'optimization scenarios'
        ns = self.study_name
        opt_builder = factory.get_builder_from_process(repo=self.repo,
                                                       mod_id=self.proc_name)

        exec_eng.factory.set_builders_to_coupling_builder(opt_builder)

        exec_eng.configure()

        usecase = study_sellar_opt(execution_engine=exec_eng)
        usecase.study_name = self.study_name
        values_dict = usecase.setup_usecase()
        exec_eng.load_study_from_input_dict(values_dict)
        exec_eng.configure()
        exec_eng.display_treeview_nodes()

        len_before_clean = len(list(exec_eng.dm.disciplines_id_map.keys()))
        # delete scenario and configure
        values_dict[f'{ns}.{scatter_scenario_name}.scenario_list'] = ['a=0-1']
        exec_eng.load_study_from_input_dict(values_dict)
        exec_eng.display_treeview_nodes()

        # assert there is no variables of second scenario in dm (so that cleaning went well) and number of disciplines after cleaning is correct
        scen_name = 'a=0-2'
        list_var_scen = [f'{ns}.{scatter_scenario_name}.{scen_name}.{sc_name}.Sellar_Problem.local_dv', f'{ns}.{scatter_scenario_name}.{scen_name}.{sc_name}.z',
                         f'{ns}.{scatter_scenario_name}.{scen_name}.{sc_name}.x', f'{ns}.{scatter_scenario_name}.{scen_name}.{sc_name}.y_2', f'{ns}.{scatter_scenario_name}.{scen_name}.{sc_name}.y_1']
        for var in list_var_scen:
            # assert variable is not in dm
            assert(var not in exec_eng.dm.convert_data_dict_with_full_name().keys())

        len_after_clean = len(list(exec_eng.dm.disciplines_id_map.keys()))

        # assert disciplines are not in dm
        assert(len_after_clean == 11)


if '__main__' == __name__:
    cls = TestSoSOptimScenario()
    cls.setUp()
    cls.test_01_ms_sellar_cleaning()
