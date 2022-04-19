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


class TestGSPureNewtonorGSMDA(unittest.TestCase):
    """
    SoSOptimScenario test class
    """

    def setUp(self):
        self.study_name = 'optim'
        self.repo = 'sos_trades_core.sos_processes.test'
        self.proc_name = 'test_sellar_opt_discopt'

    def _test_01_GSPureNewtonorGSMDA(self):
        '''
        TEST COMMENTED BECAUSE MDF FORMULATION BUILD A MDACHAIN INSTEAD OF SOSCOUPLING
        '''

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
        values_dict['optim.SellarOptimScenario.SellarCoupling.sub_mda_class'] = 'GSPureNewtonorGSMDA'
        # activate debug mode to raise error
        values_dict['optim.SellarOptimScenario.SellarCoupling.debug_mode_sellar'] = True
        exec_eng.load_study_from_input_dict(values_dict)

        exec_eng.configure()
        exec_eng.display_treeview_nodes()
        exec_eng.execute()
        # assert residual history of GS sequence is not empty to assert run used GSmda
        GS_sequence = exec_eng.root_process.sos_disciplines[0].sos_disciplines[0].sub_mda_list[0].mda_sequence[0]
        assert len(GS_sequence.residual_history) > 0
        # assert GSPureNR was executed 6 time (5 GS and one NR before raise in compute_sos_jacobian)
        GSPureNR_sequence = exec_eng.root_process.sos_disciplines[0].sos_disciplines[0].sub_mda_list[0].mda_sequence[1]
        assert len(GSPureNR_sequence.residual_history) == 7


if '__main__' == __name__:
    cls = TestGSPureNewtonorGSMDA()
    cls.setUp()
    cls._test_01_GSPureNewtonorGSMDA()
