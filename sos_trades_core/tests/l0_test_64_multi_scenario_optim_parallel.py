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
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from sos_trades_core.sos_processes.test.test_sellar_opt_ms.usecase import Study as study_sellar_opt
import platform
from sos_trades_core.sos_processes.compare_data_manager_tooling import compare_dict



class TestSoSOptimScenario(unittest.TestCase):
    """
    SoSOptimScenario test class
    """

    def setUp(self):
        self.study_name = 'optim'




        self.repo = 'sos_trades_core.sos_processes.test'
        self.proc_name = 'test_sellar_opt_ms'

    def test_01_ms_sellar_cleaning(self):

        if platform.system() != 'Windows' or True:
            print("\n Test 1 : check configure and treeview")
            exec_eng = ExecutionEngine(self.study_name)
            factory = exec_eng.factory

            opt_builder = factory.get_builder_from_process(repo=self.repo,
                                                           mod_id=self.proc_name)

            exec_eng.factory.set_builders_to_coupling_builder(opt_builder)

            exec_eng.configure()

            usecase = study_sellar_opt(execution_engine=exec_eng)
            usecase.study_name = self.study_name
            values_dict = {}
            for dict_item in usecase.setup_usecase():
                values_dict.update(dict_item)
            exec_eng.load_study_from_input_dict(values_dict)
            exec_eng.configure()
            exec_eng.display_treeview_nodes()
            # sequential execution
            exec_eng.execute()
            dm_sequential = exec_eng.dm.get_data_dict_values()


            # parallel execution

            values_dict[f'{self.study_name}.n_subcouplings_parallel'] = 2
            exec_eng.load_study_from_input_dict(values_dict)
            exec_eng.configure()
            exec_eng.execute()
            dm_parallel = exec_eng.dm.get_data_dict_values()
            dict_error = {}
            # compare dicts
            keys_parallel = [s for s in dm_sequential.keys() if 'n_subcouplings_parallel' in s]
            keys_residual = [s for s in dm_sequential.keys() if 'residuals_history' in s]
            [dm_sequential.pop(key) for key in keys_parallel+keys_residual]
            [dm_parallel.pop(key) for key in keys_parallel+keys_residual]
            compare_dict(dm_sequential,
                         dm_parallel, '', dict_error)
            assert dict_error == {}



if '__main__' == __name__:
    cls = TestSoSOptimScenario()
    cls.setUp()
    cls.test_01_ms_sellar_cleaning()
